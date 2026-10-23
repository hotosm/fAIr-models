# Builder stage: training + batch dependencies.
FROM ghcr.io/astral-sh/uv:0.11.8-python3.13-trixie-slim AS builder
ENV UV_SYSTEM_PYTHON=1 UV_LINK_MODE=copy

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgdal-dev git \
    && rm -rf /var/lib/apt/lists/*

ARG ROAD_SURFACE_REF=28be8d376443fc3224133c4f9e0a1c673fdf12f0
ARG FAIR_PY_OPS_VERSION=0.3.6
# predict() fetches roads with download_osm_data(filters=..., use_st_within=...), added in geomltoolkits 2.1.1.
ARG GEOMLTOOLKITS_VERSION=2.1.2
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --no-sources \
    "git+https://github.com/GIScience/Road-surface-classification-for-FAIR@${ROAD_SURFACE_REF}" \
    onnx \
    onnxscript==0.7.2 \
    "fair-py-ops[k8s,serve]==${FAIR_PY_OPS_VERSION}" \
    "geomltoolkits==${GEOMLTOOLKITS_VERSION}"

FROM python:3.13.13-slim-trixie AS runtime
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libexpat1 libgdal36 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

COPY models/dinov3s_road_surface_classification models/dinov3s_road_surface_classification
COPY models/conftest.py models/conftest.py
COPY models/test_integration.py models/test_integration.py

ENTRYPOINT ["/usr/local/bin/python"]

# Test stage: pytest extras for CI.
FROM runtime AS test
COPY --from=ghcr.io/astral-sh/uv:0.11.8 /uv /usr/local/bin/uv
ARG FAIR_PY_OPS_VERSION=0.3.6
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system "fair-py-ops[test]==${FAIR_PY_OPS_VERSION}"

# Inference builder: distroless-ready venv.
FROM ghcr.io/astral-sh/uv:0.11.8-trixie-slim AS inference-builder
ENV UV_PYTHON_INSTALL_DIR=/python UV_PYTHON_PREFERENCE=only-managed
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN uv python install 3.13
WORKDIR /app
ARG ROAD_SURFACE_REF=28be8d376443fc3224133c4f9e0a1c673fdf12f0
ARG FAIR_PY_OPS_VERSION=0.3.6
ARG GEOMLTOOLKITS_VERSION=2.1.2
# road_surface_classifier without its deps: predict() only uses its patch cutting, not torch/lightning/timm.
RUN --mount=type=cache,target=/python/.cache/uv \
    uv venv && uv pip install --no-sources \
    "fair-py-ops[serve]==${FAIR_PY_OPS_VERSION}" \
    "geomltoolkits==${GEOMLTOOLKITS_VERSION}" \
    onnxruntime==1.28.0 \
    geopandas rasterio pillow pyarrow \
    && uv pip install --no-deps \
    "git+https://github.com/GIScience/Road-surface-classification-for-FAIR@${ROAD_SURFACE_REF}"
COPY models/dinov3s_road_surface_classification/pipeline.py /app/models/dinov3s_road_surface_classification/pipeline.py

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
