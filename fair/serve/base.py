"""Model-agnostic ONNX serving runtime (Starlette app factory).

The Dockerfile copies exactly one model pipeline into the image and sets
MODEL_MODULE (via KNative service env, sourced from STAC mlm:entrypoint).
This module imports that pipeline and exposes its `predict` function
behind `POST /predict`.
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


async def _predict(request: Request, pipeline: Any) -> JSONResponse:
    try:
        payload = await request.json()
    except json.JSONDecodeError as exc:
        return JSONResponse({"error": f"Invalid JSON body: {exc}"}, status_code=400)

    model_uri = payload.get("model_uri")
    input_images = payload.get("input_images")
    params = payload.get("params") or {}

    if not isinstance(model_uri, str) or not model_uri:
        return JSONResponse({"error": "'model_uri' is required and must be a string"}, status_code=400)
    if not isinstance(input_images, str) or not input_images:
        return JSONResponse({"error": "'input_images' is required and must be a string"}, status_code=400)
    if not isinstance(params, dict):
        return JSONResponse({"error": "'params' must be an object"}, status_code=400)

    try:
        session = load_session(model_uri)
        result = pipeline.predict(session, input_images, params)
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
