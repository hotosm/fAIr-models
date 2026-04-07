# syntax=docker/dockerfile:1.7

# Base: ghcr.io/hotosm/fair-utilities-ramp:cpu-latest (or :gpu-latest via --build-arg BASE_IMAGE=...)
# Build from fAIr-models root: docker build -f models/ramp/Dockerfile -t ramp-v1:cpu .
ARG BASE_IMAGE=ghcr.io/hotosm/fair-utilities-ramp:cpu-latest
FROM ${BASE_IMAGE}

ENV MPLBACKEND=Agg \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    RAMP_HOME=/workspace \
    SM_FRAMEWORK=tf.keras

WORKDIR /workspace

RUN /app/.venv/bin/python -m pip install --no-cache-dir "fair-py-ops==0.0.6"

# Smoke-test extras only (ZenML[server], STAC, fairpredictor); drop for production slim images.
RUN /app/.venv/bin/python -m pip install --no-cache-dir \
    "zenml[server]==0.93.3" \
    "pystac[validation]>=1.14.3" \
    "universal-pathlib" \
    "fairpredictor>=0.5.1"
