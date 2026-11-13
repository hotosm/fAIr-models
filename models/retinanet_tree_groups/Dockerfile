# One fair-trees commit for every stage: training, the ONNX export and serving must share
# the library's preprocessing and decoding. This is the commit of the library's v0.1.0 tag;
# a commit, unlike a tag, cannot be moved.
ARG FAIR_TREES_REF=2cddcfba04e7e5c2b74a8ef0d20e187a8bc91bfd

# Builder stage: training + batch dependencies (GPU torch).
# PyTorch wheels bundle the CUDA runtime, so a plain python base works on GPU nodes
# (the NVIDIA driver is injected by the device plugin) and falls back to CPU in CI.
FROM ghcr.io/astral-sh/uv:0.11.8-python3.13-trixie-slim AS builder
ENV UV_SYSTEM_PYTHON=1 UV_LINK_MODE=copy
ARG FAIR_PY_OPS_VERSION=0.3.17

# git is for uv's fair-trees checkout; it stays in the builder stages.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgdal-dev git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# fair-trees[train] pins torch and torchvision with ==; these are the CUDA builds of those pins.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install \
    --extra-index-url https://download.pytorch.org/whl/cu126 \
    --index-strategy unsafe-best-match \
    torch==2.11.0 torchvision==0.26.0
# --no-sources: fair-trees' pyproject points torch at the PyTorch CPU index for its own dev setup,
# and uv honours that for a git dependency; without it a fresh resolve picks +cpu torch.
# The serving core is pinned as in inference-builder, so training, export and serving match.
# FAIR_TREES_REF is declared here so a ref change keeps the cached apt and torch layers.
ARG FAIR_TREES_REF
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --no-sources \
    onnx==1.23.2 \
    numpy==2.5.4 \
    Pillow==13.0.0 \
    pyproj==3.8.0 \
    rasterio==1.5.2 \
    "fair-trees[train] @ git+https://github.com/OmdenaAI/HOTOSM-fAIr-Trees@${FAIR_TREES_REF}" \
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

# The whole pack: training reads its defaults and split description from stac-item.json.
COPY models/retinanet_tree_groups models/retinanet_tree_groups
COPY models/conftest.py models/conftest.py
COPY models/test_integration.py models/test_integration.py

ENTRYPOINT ["/usr/local/bin/python"]

# Test stage: pytest extras for CI (onnx and torch come with fair-trees[train] in runtime).
FROM runtime AS test
ARG FAIR_PY_OPS_VERSION=0.3.17
COPY --from=ghcr.io/astral-sh/uv:0.11.8 /uv /usr/local/bin/uv
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system "fair-py-ops[test]==${FAIR_PY_OPS_VERSION}"

# Inference builder: distroless-ready CPU venv (ONNX serving, no torch).
# fair-trees without extras is its serving core: numpy, Pillow, rasterio, pyproj, affine.
# The serving core pins equal the builder's; numpy and Pillow are the versions fair-trees' own
# tests run with (its chip resize is Pillow's).
FROM ghcr.io/astral-sh/uv:0.11.8-trixie-slim AS inference-builder
ENV UV_PYTHON_INSTALL_DIR=/python UV_PYTHON_PREFERENCE=only-managed
ARG FAIR_PY_OPS_VERSION=0.3.17
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN uv python install 3.13
WORKDIR /app
ARG FAIR_TREES_REF
RUN --mount=type=cache,target=/python/.cache/uv \
    uv venv && uv pip install \
    "fair-py-ops[serve]==${FAIR_PY_OPS_VERSION}" \
    onnxruntime==1.28.0 \
    numpy==2.5.4 \
    Pillow==13.0.0 \
    pyproj==3.8.0 \
    rasterio==1.5.2 \
    "fair-trees @ git+https://github.com/OmdenaAI/HOTOSM-fAIr-Trees@${FAIR_TREES_REF}"
COPY models/retinanet_tree_groups/pipeline.py /app/models/retinanet_tree_groups/pipeline.py

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
