# syntax=docker/dockerfile:1.7

ARG BASE_IMAGE=ghcr.io/hotosm/fair-utilities-yolo:cpu-latest

# ---------------------------------------------------------------------------
# Builder stage: install model-pack deps into the base image venv
# ---------------------------------------------------------------------------
FROM ${BASE_IMAGE} AS builder

ENV UV_LINK_MODE=copy

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /workspace

COPY pyproject.toml README.md fair_zenml_patch.pth /tmp/fair-src/
COPY fair /tmp/fair-src/fair

# The fair-utilities-yolo image already provides `hot_fair_utilities` under
# `/app/.venv`. Install torch, fAIr client libs, etc. into that same venv so
# runtime and tests share one interpreter (do not use `--system` / /usr/local).
RUN --mount=type=cache,target=/root/.cache/uv \
    SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 \
    uv pip install --python /app/.venv/bin/python \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    --index-strategy unsafe-best-match \
    torch==2.10.0 \
    "ultralytics>=8.4.0" \
    rasterio \
    pyproj \
    onnx \
    onnxscript \
    "pystac[validation]>=1.14.3" \
    "universal-pathlib" \
    "fairpredictor>=0.5.1" \
    /tmp/fair-src[k8s]

# ---------------------------------------------------------------------------
# Runtime stage: production image (no test deps)
# ---------------------------------------------------------------------------
FROM ${BASE_IMAGE} AS runtime

WORKDIR /workspace

ENV PATH="/app/.venv/bin:$PATH" \
    MPLBACKEND=Agg \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY --from=builder /app/.venv /app/.venv

COPY models/yolo_v8_segmentation models/yolo_v8_segmentation
COPY models/conftest.py models/conftest.py
COPY models/test_integration.py models/test_integration.py

ENTRYPOINT ["/app/.venv/bin/python"]

# ---------------------------------------------------------------------------
# Test stage: add pytest + ZenML server deps
# ---------------------------------------------------------------------------
FROM runtime AS test

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
COPY --from=builder /tmp/fair-src /tmp/fair-src

RUN --mount=type=cache,target=/root/.cache/uv \
    SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 \
    uv pip install --python /app/.venv/bin/python /tmp/fair-src[test]
