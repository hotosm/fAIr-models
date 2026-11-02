# Builder stage: training + batch dependencies (GPU torch).
# PyTorch wheels bundle the CUDA runtime, so a plain python base works on GPU nodes
# (the NVIDIA driver is injected by the device plugin) and falls back to CPU in CI.
# spd-hot[train] brings ultralytics, onnx, onnxslim and onnxruntime, which Ultralytics' ONNX export needs.
FROM ghcr.io/astral-sh/uv:0.11.8-python3.13-trixie-slim AS builder
ENV UV_SYSTEM_PYTHON=1 UV_LINK_MODE=copy
ARG FAIR_PY_OPS_VERSION=0.3.17
ARG SPD_HOT_REF=v0.3.0

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgdal-dev git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# torch and torchvision come first, from the CUDA wheel index, so spd-hot[train] (through Ultralytics) reuses them.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install \
    --extra-index-url https://download.pytorch.org/whl/cu126 \
    --index-strategy unsafe-best-match \
    torch==2.14.1 torchvision==0.29.1
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install \
    "spd-hot[train] @ git+https://github.com/AbdelrahmanKatkat/spd-hot@${SPD_HOT_REF}" \
    "fair-py-ops[k8s,serve]==${FAIR_PY_OPS_VERSION}"

# Runtime stage: minimal training/batch image. libgl1 and libglib2.0-0 are for the OpenCV that Ultralytics imports.
FROM python:3.13.13-slim-trixie AS runtime
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libexpat1 libgdal36 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

COPY models/yolo11m_swimming_pools models/yolo11m_swimming_pools
COPY models/conftest.py models/conftest.py
COPY models/test_integration.py models/test_integration.py

ENTRYPOINT ["/usr/local/bin/python"]

# Test stage: pytest extras for CI.
FROM runtime AS test
ARG FAIR_PY_OPS_VERSION=0.3.17
COPY --from=ghcr.io/astral-sh/uv:0.11.8 /uv /usr/local/bin/uv
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system "fair-py-ops[test]==${FAIR_PY_OPS_VERSION}"

# Inference builder: distroless-ready CPU venv (ONNX serving, no torch, no Ultralytics).
FROM ghcr.io/astral-sh/uv:0.11.8-trixie-slim AS inference-builder
ENV UV_PYTHON_INSTALL_DIR=/python UV_PYTHON_PREFERENCE=only-managed
ARG FAIR_PY_OPS_VERSION=0.3.17
ARG SPD_HOT_REF=v0.3.0
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN uv python install 3.13
WORKDIR /app
# spd-hot[infer] brings onnxruntime, and its core brings numpy, Pillow, pyproj and rasterio: nothing is pinned twice.
RUN --mount=type=cache,target=/python/.cache/uv \
    uv venv && uv pip install \
    "fair-py-ops[serve]==${FAIR_PY_OPS_VERSION}" \
    "spd-hot[infer] @ git+https://github.com/AbdelrahmanKatkat/spd-hot@${SPD_HOT_REF}"
COPY models/yolo11m_swimming_pools/pipeline.py /app/models/yolo11m_swimming_pools/pipeline.py

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
