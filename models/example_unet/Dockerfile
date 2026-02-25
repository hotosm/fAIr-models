# Builder stage: install dependencies
FROM ghcr.io/astral-sh/uv:python3.13-trixie-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

RUN uv venv /opt/venv
RUN . /opt/venv/bin/activate && \
    uv pip install torch==2.10.0 torchgeo==0.9.0 fair-py-ops

# Runtime stage: minimal image
FROM python:3.13-slim-trixie

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libexpat1 \
    libgdal36 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY models/example_unet models/example_unet

RUN python -c "import torch, torchgeo; from models.example_unet.pipeline import training_pipeline; print('UNet pipeline ready')"

ENTRYPOINT ["/opt/venv/bin/python"]

