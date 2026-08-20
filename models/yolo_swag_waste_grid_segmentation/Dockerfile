# Builder stage: training + batch dependencies (GPU torch).
FROM ghcr.io/astral-sh/uv:0.11.8-python3.13-trixie-slim AS builder
ENV UV_SYSTEM_PYTHON=1 UV_LINK_MODE=copy
ARG FAIR_PY_OPS_VERSION=0.3.4

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgdal-dev gdal-bin libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install \
    --extra-index-url https://download.pytorch.org/whl/cu126 \
    --index-strategy unsafe-best-match \
    torch==2.10.0 torchvision==0.25.0
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install \
    ultralytics==8.4.124 \
    rasterio==1.5.1 \
    pyproj==3.7.2 \
    geopandas==1.1.4 \
    shapely==2.1.2 \
    Pillow==12.3.0 \
    onnx==1.22.0 \
    onnxscript==0.7.1 \
    "fair-py-ops[k8s,serve]==${FAIR_PY_OPS_VERSION}"

# Runtime stage: minimal training/batch image.
FROM python:3.13.13-slim-trixie AS runtime
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libexpat1 libgdal36 gdal-bin libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

COPY models/yolo_swag_waste_grid_segmentation models/yolo_swag_waste_grid_segmentation
COPY models/conftest.py models/conftest.py
COPY models/test_integration.py models/test_integration.py
RUN touch models/__init__.py

ENTRYPOINT ["/usr/local/bin/python"]

# Test stage: pytest extras for CI.
FROM runtime AS test
ARG FAIR_PY_OPS_VERSION=0.3.4
COPY --from=ghcr.io/astral-sh/uv:0.11.8 /uv /usr/local/bin/uv
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system "fair-py-ops[test]==${FAIR_PY_OPS_VERSION}"

# Inference builder: distroless-ready CPU venv (ONNX serving, no torch).
FROM ghcr.io/astral-sh/uv:0.11.8-trixie-slim AS inference-builder
ENV UV_PYTHON_INSTALL_DIR=/python UV_PYTHON_PREFERENCE=only-managed
ARG FAIR_PY_OPS_VERSION=0.3.4
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN uv python install 3.13
WORKDIR /app
RUN --mount=type=cache,target=/python/.cache/uv \
    uv venv && uv pip install \
    "fair-py-ops[serve]==${FAIR_PY_OPS_VERSION}" \
    onnxruntime==1.29.0 \
    rasterio==1.5.1 \
    pyproj==3.7.2 \
    shapely==2.1.2 \
    numpy==2.5.2 \
    Pillow==12.3.0
# fair-py-ops bundles a top-level `models` package; own /app/models so this model's
# pipeline resolves ahead of it (fair never imports the bundled models at runtime).
COPY models/yolo_swag_waste_grid_segmentation/__init__.py models/yolo_swag_waste_grid_segmentation/pipeline.py /app/models/yolo_swag_waste_grid_segmentation/
RUN touch /app/models/__init__.py

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
