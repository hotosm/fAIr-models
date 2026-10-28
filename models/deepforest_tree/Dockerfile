# Builder stage: training + batch dependencies (GPU torch).
# PyTorch wheels bundle the CUDA runtime, so a plain python base works on GPU nodes
# (the NVIDIA driver is injected by the device plugin) and falls back to CPU in CI.
FROM ghcr.io/astral-sh/uv:0.11.8-python3.13-trixie-slim AS builder
ENV UV_SYSTEM_PYTHON=1 UV_LINK_MODE=copy
ARG FAIR_PY_OPS_VERSION=0.3.17
ARG TD_HOT_REF=v0.2.1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgdal-dev git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install \
    --extra-index-url https://download.pytorch.org/whl/cu126 \
    --index-strategy unsafe-best-match \
    torch==2.14.1 torchvision==0.29.1
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install \
    onnx==1.23.2 \
    "td-hot[train] @ git+https://github.com/AbdelrahmanKatkat/td-hot@${TD_HOT_REF}" \
    "fair-py-ops[k8s,serve]==${FAIR_PY_OPS_VERSION}"

# Runtime stage: minimal training/batch image.
FROM python:3.13.13-slim-trixie AS runtime
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libexpat1 libgdal36 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

COPY models/deepforest_tree models/deepforest_tree
COPY models/conftest.py models/conftest.py
COPY models/test_integration.py models/test_integration.py

ENTRYPOINT ["/usr/local/bin/python"]

# Test stage: pytest extras for CI.
FROM runtime AS test
ARG FAIR_PY_OPS_VERSION=0.3.17
COPY --from=ghcr.io/astral-sh/uv:0.11.8 /uv /usr/local/bin/uv
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system "fair-py-ops[test]==${FAIR_PY_OPS_VERSION}"

# Inference builder: distroless-ready CPU venv (ONNX serving, no torch).
FROM ghcr.io/astral-sh/uv:0.11.8-trixie-slim AS inference-builder
ENV UV_PYTHON_INSTALL_DIR=/python UV_PYTHON_PREFERENCE=only-managed
ARG FAIR_PY_OPS_VERSION=0.3.17
ARG TD_HOT_REF=v0.2.1
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN uv python install 3.13
WORKDIR /app
RUN --mount=type=cache,target=/python/.cache/uv \
    uv venv && uv pip install \
    "fair-py-ops[serve]==${FAIR_PY_OPS_VERSION}" \
    onnxruntime==1.28.0 \
    numpy==2.5.3 \
    Pillow==12.3.0 \
    pyproj==3.8.0 \
    rasterio==1.5.2 \
    "td-hot @ git+https://github.com/AbdelrahmanKatkat/td-hot@${TD_HOT_REF}"
COPY models/deepforest_tree/pipeline.py /app/models/deepforest_tree/pipeline.py

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
