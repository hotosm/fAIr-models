# Builder stage: install dependencies
FROM ghcr.io/astral-sh/uv:python3.13-trixie-slim AS builder
ENV UV_SYSTEM_PYTHON=1 UV_LINK_MODE=copy

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md fair_zenml_patch.pth /tmp/fair-src/
COPY fair /tmp/fair-src/fair
RUN --mount=type=cache,target=/root/.cache/uv \
    SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 \
    uv pip install \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    --index-strategy unsafe-best-match \
    torch==2.10.0 \
    torchvision \
    rasterio \
    onnx \
    onnxscript \
    zenml==0.93.3 \
    /tmp/fair-src[k8s]

# Runtime stage: minimal image
FROM python:3.13-slim-trixie

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libexpat1 libgdal36 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY models/resnet18_classification models/resnet18_classification

ENTRYPOINT ["/usr/local/bin/python"]
