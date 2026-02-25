# Builder stage: install dependencies
FROM ghcr.io/astral-sh/uv:python3.13-trixie-slim AS builder
ENV UV_SYSTEM_PYTHON=1 UV_LINK_MODE=copy

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install torch==2.10.0 torchgeo==0.9.0

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install fair-py-ops

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

COPY models/example_unet models/example_unet

ENTRYPOINT ["/usr/local/bin/python"]

