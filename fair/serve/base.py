"""Model-agnostic ONNX serving app: `POST /predict` takes `model_uri`, `imagery` ({input name: tile URL}, or `image_uri`
for an `image` input), `bbox`, `zoom` and `params`, and runs `predict` of the pipeline module named by MODEL_MODULE."""

import asyncio
import importlib
import inspect
import json
import logging
import os
import tempfile
from collections.abc import Awaitable
from functools import lru_cache
from pathlib import Path
from typing import Any

from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.types import ASGIApp

from fair.stac.inputs import IMAGE_INPUT, INPUT_NAME_PATTERN

logger = logging.getLogger(__name__)

MODEL_MODULE_ENV = "MODEL_MODULE"
_ONNX_CACHE_SIZE = 8
_MIN_ZOOM = 14
_MAX_ZOOM = 22
_DISCONNECT_POLL_SECONDS = 0.5
_CLIENT_CLOSED_REQUEST = 499


class ClientDisconnectedError(Exception):
    """The caller closed the connection before the prediction finished."""


def _cors_list(name: str, default: str = "*") -> list[str]:
    raw = os.environ.get(name, default).strip()
    if raw in ("*", ""):
        return ["*"]
    return [v.strip() for v in raw.split(",") if v.strip()]


class ServeConfigError(RuntimeError):
    pass


@lru_cache(maxsize=_ONNX_CACHE_SIZE)
def load_session(model_uri: str) -> Any:
    """Download the ONNX artifact and build a cached CPU inference session.

    Intra-op threads are capped to FAIR_KNATIVE_ONNX_THREADS so the session
    doesn't oversubscribe the pod's CPU quota at concurrency 1.
    """
    from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions
    from upath import UPath

    source = UPath(model_uri)
    local_path = Path(tempfile.mkdtemp()) / (source.name or "model.onnx")
    local_path.write_bytes(source.read_bytes())

    options = SessionOptions()
    options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
    options.inter_op_num_threads = 1
    intra_op_threads = int(os.environ.get("FAIR_KNATIVE_ONNX_THREADS", "0"))
    if intra_op_threads > 0:
        options.intra_op_num_threads = intra_op_threads
    return InferenceSession(str(local_path), sess_options=options, providers=["CPUExecutionProvider"])


def _load_pipeline() -> Any:
    module_name = os.environ.get(MODEL_MODULE_ENV)
    if not module_name:
        raise ServeConfigError(f"{MODEL_MODULE_ENV} environment variable is not set")
    module = importlib.import_module(module_name)
    if not hasattr(module, "predict"):
        raise ServeConfigError(f"Pipeline module '{module_name}' does not export predict()")
    return module


async def _health(_request: Request) -> JSONResponse:
    return JSONResponse({"status": "ok"})


def _validate_request(payload: Any) -> tuple[dict[str, Any] | None, str | None]:
    if not isinstance(payload, dict):
        return None, "Request body must be a JSON object"

    model_uri = payload.get("model_uri")
    bbox = payload.get("bbox")
    zoom = payload.get("zoom")
    params = payload.get("params", {})

    if not isinstance(model_uri, str) or not model_uri:
        return None, "'model_uri' is required and must be a non-empty string"
    imagery, error = _request_imagery(payload)
    if error is not None:
        return None, error
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None, "'bbox' is required and must be a list of 4 numbers [w, s, e, n]"
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in bbox):
        return None, "'bbox' values must be numbers"
    west, south, east, north = (float(v) for v in bbox)
    if west >= east or south >= north:
        return None, "'bbox' must satisfy west<east and south<north"
    if not isinstance(zoom, int) or isinstance(zoom, bool):
        return None, "'zoom' is required and must be an integer"
    if zoom < _MIN_ZOOM or zoom > _MAX_ZOOM:
        return None, f"'zoom' must be within [{_MIN_ZOOM}, {_MAX_ZOOM}]"
    if not isinstance(params, dict):
        return None, "'params' must be an object"

    return {
        "model_uri": model_uri,
        "imagery": imagery,
        "bbox": [west, south, east, north],
        "zoom": zoom,
        "params": params,
    }, None


def _request_imagery(payload: dict[str, Any]) -> tuple[dict[str, str] | None, str | None]:
    if ("imagery" in payload) == ("image_uri" in payload):
        return None, "Give exactly one of 'imagery' ({input name: template URL}) or 'image_uri'"
    imagery = payload.get("imagery", {IMAGE_INPUT: payload.get("image_uri")})
    if not isinstance(imagery, dict) or not imagery:
        return None, "'imagery' must be a non-empty object of {input name: template URL}"
    for name, url in imagery.items():
        if not INPUT_NAME_PATTERN.fullmatch(name):
            return None, f"'imagery' input name {name!r} must use only lowercase letters, digits, '_' or '-'"
        if not isinstance(url, str) or not url:
            return None, f"'imagery' URL of input {name!r} must be a non-empty string"
    return imagery, None


async def _fetch_chips(imagery: dict[str, str], bbox: list[float], zoom: int, out_dir: str) -> dict[str, str]:
    from geomltoolkits.downloader.tms import download_tiles

    # A failed download cancels the others before the temporary directory is removed.
    try:
        async with asyncio.TaskGroup() as group:
            tasks = {
                name: group.create_task(
                    download_tiles(tms=url, zoom=zoom, out=str(Path(out_dir) / name), bbox=bbox, georeference=True)
                )
                for name, url in imagery.items()
            }
    except ExceptionGroup as failures:  # the caller reports the first download's own error
        raise failures.exceptions[0] from failures
    return {name: task.result() for name, task in tasks.items()}


async def _await_unless_disconnected(request: Request, work: Awaitable[Any]) -> Any:
    """Await `work`, cancelling it if the client disconnects first."""
    task = asyncio.ensure_future(work)
    try:
        while not task.done():
            await asyncio.wait({task}, timeout=_DISCONNECT_POLL_SECONDS)
            if not task.done() and await request.is_disconnected():
                raise ClientDisconnectedError
        return task.result()
    finally:
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                if asyncio.current_task().cancelling():
                    raise


async def _predict(request: Request, pipeline: Any, forward_bbox: bool) -> JSONResponse:
    try:
        payload = await request.json()
    except json.JSONDecodeError as exc:
        return JSONResponse({"error": f"Invalid JSON body: {exc}"}, status_code=400)

    parsed, error = _validate_request(payload)
    if error is not None:
        return JSONResponse({"error": error}, status_code=400)
    assert parsed is not None

    try:
        session = load_session(parsed["model_uri"])
        with tempfile.TemporaryDirectory(prefix="fair-serve-") as tmp:
            chips_dirs = await _await_unless_disconnected(
                request, _fetch_chips(parsed["imagery"], parsed["bbox"], parsed["zoom"], tmp)
            )
            if await request.is_disconnected():
                raise ClientDisconnectedError
            predict_kwargs = {"bbox": parsed["bbox"]} if forward_bbox else {}
            result = pipeline.predict(session, chips_dirs, parsed["params"], **predict_kwargs)
    except ClientDisconnectedError:
        logger.info("client disconnected; prediction skipped")
        return JSONResponse({"error": "client disconnected"}, status_code=_CLIENT_CLOSED_REQUEST)
    except Exception as exc:
        logger.exception("predict failed")
        return JSONResponse({"error": str(exc)}, status_code=500)

    return JSONResponse(result)


def create_app() -> ASGIApp:
    pipeline = _load_pipeline()
    forward_bbox = "bbox" in inspect.signature(pipeline.predict).parameters

    async def predict_route(request: Request) -> JSONResponse:
        return await _predict(request, pipeline, forward_bbox)

    app: ASGIApp = Starlette(
        debug=False,
        routes=[
            Route("/health", _health, methods=["GET"]),
            Route("/predict", predict_route, methods=["POST"]),
        ],
    )
    # FAIR_KNATIVE_CORS_* envs come from the ksvc manifest (fair/infra/knative-service.yaml).
    return CORSMiddleware(
        app,
        allow_origins=_cors_list("FAIR_KNATIVE_CORS_ORIGINS"),
        allow_methods=_cors_list("FAIR_KNATIVE_CORS_METHODS"),
        allow_headers=_cors_list("FAIR_KNATIVE_CORS_HEADERS"),
    )
