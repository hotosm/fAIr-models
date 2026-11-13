# Builder stage: batch prediction dependencies (ONNX on CPU, no training stack).
FROM ghcr.io/astral-sh/uv:0.11.8-python3.13-trixie-slim AS builder
ENV UV_SYSTEM_PYTHON=1 UV_LINK_MODE=copy
ARG FAIR_PY_OPS_VERSION=0.3.17

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install \
    onnxruntime==1.28.0 \
    rasterio==1.5.1 \
    pyproj==3.7.2 \
    shapely==2.1.2 \
    numpy==2.4.2 \
    "fair-py-ops[k8s,serve]==${FAIR_PY_OPS_VERSION}"

# Runtime stage: batch prediction image for the ZenML inference pipeline.
FROM python:3.13.13-slim-trixie AS runtime
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libexpat1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

COPY models/unet_tree_segmentation models/unet_tree_segmentation
COPY models/conftest.py models/conftest.py

ENTRYPOINT ["/usr/local/bin/python"]

# Test stage: pytest extras, plus onnx to build the toy model the tests run.
FROM runtime AS test
ARG FAIR_PY_OPS_VERSION=0.3.17
COPY --from=ghcr.io/astral-sh/uv:0.11.8 /uv /usr/local/bin/uv
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system "fair-py-ops[test]==${FAIR_PY_OPS_VERSION}" onnx==1.22.0

# Inference builder: distroless-ready CPU venv for live serving.
FROM ghcr.io/astral-sh/uv:0.11.8-trixie-slim AS inference-builder
ENV UV_PYTHON_INSTALL_DIR=/python UV_PYTHON_PREFERENCE=only-managed
ARG FAIR_PY_OPS_VERSION=0.3.17
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN uv python install 3.13
WORKDIR /app
RUN --mount=type=cache,target=/python/.cache/uv \
    uv venv && uv pip install \
    "fair-py-ops[serve]==${FAIR_PY_OPS_VERSION}" \
    onnxruntime==1.28.0 \
    rasterio==1.5.1 \
    pyproj==3.7.2 \
    shapely==2.1.2 \
    numpy==2.4.2
COPY models/unet_tree_segmentation/pipeline.py /app/models/unet_tree_segmentation/pipeline.py

# Inference stage: distroless live-serving runtime.
FROM gcr.io/distroless/python3-debian12:nonroot AS inference
WORKDIR /app
COPY --from=inference-builder /python /python
COPY --from=inference-builder /app/.venv /app/.venv
COPY --from=inference-builder /app/models /app/models
ENV PATH="/app/.venv/bin:$PATH" PYTHONPATH=/app
EXPOSE 8080
ENTRYPOINT ["/app/.venv/bin/python", "-m", "uvicorn"]
CMD ["fair.serve.base:create_app", "--factory", "--host", "0.0.0.0", "--port", "8080"]
