# syntax=docker/dockerfile:1.7

# Base image: ghcr.io/hotosm/fair-utilities-ramp:cpu-latest (or :gpu-latest via BASE_IMAGE build arg).
# Build from the fAIr-models repo root:
#   docker build -f models/ramp/Dockerfile --target test -t ramp:test .
#   docker build -f models/ramp/Dockerfile --target runtime -t ramp:runtime .
#   docker build -f models/ramp/Dockerfile --target inference -t ramp:inference .
ARG BASE_IMAGE=ghcr.io/hotosm/fair-utilities-ramp:cpu-latest

# ---------------------------------------------------------------------------
# Builder stage: install model-pack deps into the base image venv
# ---------------------------------------------------------------------------
FROM ${BASE_IMAGE} AS builder

ENV UV_LINK_MODE=copy

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /workspace

COPY pyproject.toml README.md fair_zenml_patch.pth /tmp/fair-src/
COPY fair /tmp/fair-src/fair

# The fair-utilities-ramp image already provides hot_fair_utilities, ramp-fair,
# segmentation_models, and TensorFlow under /app/.venv. Add the fAIr model-pack
# deps (fair itself, STAC validation, fairpredictor, onnx tooling) into that
# same venv so training, tests, and inference share one interpreter.
RUN --mount=type=cache,target=/root/.cache/uv \
    SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 \
    uv pip install --python /app/.venv/bin/python \
    "pystac[validation]>=1.14.3" \
    "universal-pathlib" \
    "fairpredictor>=0.5.1" \
    "onnx>=1.16" \
    "tf2onnx>=1.16" \
    /tmp/fair-src[k8s]

# ---------------------------------------------------------------------------
# Runtime stage: production image (no test deps)
# ---------------------------------------------------------------------------
FROM ${BASE_IMAGE} AS runtime

WORKDIR /workspace

ENV PATH="/app/.venv/bin:$PATH" \
    MPLBACKEND=Agg \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    RAMP_HOME=/app \
    SM_FRAMEWORK=tf.keras

COPY --from=builder /app/.venv /app/.venv

COPY models/ramp models/ramp
COPY models/conftest.py models/conftest.py
COPY models/test_integration.py models/test_integration.py

ENTRYPOINT ["/app/.venv/bin/python"]

# ---------------------------------------------------------------------------
# Test stage: add pytest + ZenML server deps for step & integration tests
# ---------------------------------------------------------------------------
FROM runtime AS test

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
COPY --from=builder /tmp/fair-src /tmp/fair-src

RUN --mount=type=cache,target=/root/.cache/uv \
    SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 \
    uv pip install --python /app/.venv/bin/python /tmp/fair-src[test]

# ---------------------------------------------------------------------------
# Inference stage: runtime + serving deps for the smoke test / live API
# ---------------------------------------------------------------------------
FROM runtime AS inference

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
COPY --from=builder /tmp/fair-src /tmp/fair-src

RUN --mount=type=cache,target=/root/.cache/uv \
    SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 \
    uv pip install --python /app/.venv/bin/python /tmp/fair-src[serve]

ENV PYTHONPATH=/workspace
EXPOSE 8080
CMD ["/app/.venv/bin/uvicorn", "fair.serve.base:create_app", "--factory", "--host", "0.0.0.0", "--port", "8080"]
