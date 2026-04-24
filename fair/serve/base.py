"""Model-agnostic ONNX serving runtime (Starlette app factory).

The Dockerfile copies exactly one model pipeline into the image and sets
MODEL_MODULE (via KNative service env, sourced from STAC mlm:entrypoint).
This module imports that pipeline and exposes its `predict` function
behind `POST /predict`.

Request schema:
    {
      "model_uri":  str,           # ONNX artifact URI (s3/http/file)
      "image_uri":  str,           # TMS/XYZ/WMS/WMTS/TileJSON template URL
      "bbox":       [w, s, e, n],  # EPSG:4326
      "zoom":       int,
      "params":     { ... }        # pipeline-specific inference params
    }

The server downloads imagery chips via geomltoolkits into a temporary
directory and passes that directory to the pipeline's `predict(session,
input_images, params)` function.
"""

from __future__ import annotations

import importlib
import json
import logging
import os
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

logger = logging.getLogger(__name__)

MODEL_MODULE_ENV = "MODEL_MODULE"
_ONNX_CACHE_SIZE = 8
_MIN_ZOOM = 14
_MAX_ZOOM = 22


class ServeConfigError(RuntimeError):
    pass


@lru_cache(maxsize=_ONNX_CACHE_SIZE)
def load_session(model_uri: str) -> Any:
    """Download the ONNX artifact and build a CPU inference session.

    Cached so repeated requests for the same URI reuse the session.
    """
    import onnxruntime as ort
    from upath import UPath

    source = UPath(model_uri)
    local_path = Path(tempfile.mkdtemp()) / (source.name or "model.onnx")
    local_path.write_bytes(source.read_bytes())
    return ort.InferenceSession(str(local_path), providers=["CPUExecutionProvider"])


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
    image_uri = payload.get("image_uri")
    bbox = payload.get("bbox")
    zoom = payload.get("zoom")
    params = payload.get("params", {})

    if not isinstance(model_uri, str) or not model_uri:
        return None, "'model_uri' is required and must be a non-empty string"
    if not isinstance(image_uri, str) or not image_uri:
        return None, "'image_uri' is required and must be a non-empty string"
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
        "image_uri": image_uri,
        "bbox": [west, south, east, north],
        "zoom": zoom,
        "params": params,
    }, None


async def _fetch_chips(image_uri: str, bbox: list[float], zoom: int, out_dir: str) -> str:
    from geomltoolkits.downloader.tms import download_tiles

    return await download_tiles(
        tms=image_uri,
        zoom=zoom,
        out=out_dir,
        bbox=bbox,
        georeference=True,
    )


async def _predict(request: Request, pipeline: Any) -> JSONResponse:
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
            chips_dir = await _fetch_chips(parsed["image_uri"], parsed["bbox"], parsed["zoom"], tmp)
            result = pipeline.predict(session, chips_dir, parsed["params"])
    except Exception as exc:
        logger.exception("predict failed")
        return JSONResponse({"error": str(exc)}, status_code=500)

    return JSONResponse(result)


def create_app() -> Starlette:
    pipeline = _load_pipeline()

    async def predict_route(request: Request) -> JSONResponse:
        return await _predict(request, pipeline)

    return Starlette(
        debug=False,
        routes=[
            Route("/health", _health, methods=["GET"]),
            Route("/predict", predict_route, methods=["POST"]),
        ],
    )
