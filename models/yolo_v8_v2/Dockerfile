# syntax=docker/dockerfile:1.7

# Base: ghcr.io/hotosm/fair-utilities-yolo:cpu-latest (or :gpu-latest via --build-arg BASE_IMAGE=...)
# Build from fAIr-models root: docker build -f models/yolo_v8_v2/Dockerfile -t yolo-v8-v2:cpu .
ARG BASE_IMAGE=ghcr.io/hotosm/fair-utilities-yolo:cpu-latest
FROM ${BASE_IMAGE}

ENV MPLBACKEND=Agg \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# fAIr-utilities base image creates the venv via uv, which omits pip.
# We copy uv to use 'uv pip install' into that venv.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /workspace

RUN uv pip install --python /app/.venv/bin/python "fair-py-ops==0.0.6"

# Smoke-test extras only (ZenML[server], STAC, fairpredictor, UPath); drop for production slim images.
RUN uv pip install --python /app/.venv/bin/python \
    "zenml[server]==0.93.3" \
    "pystac[validation]>=1.14.3" \
    "universal-pathlib" \
    "fairpredictor>=0.5.1"
