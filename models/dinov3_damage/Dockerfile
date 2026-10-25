# Builder stage: training + batch dependencies.
FROM ghcr.io/astral-sh/uv:0.11.8-python3.13-trixie-slim AS builder
ENV UV_SYSTEM_PYTHON=1 UV_LINK_MODE=copy

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgdal-dev git \
    && rm -rf /var/lib/apt/lists/*

ARG DDA_REF=c0c709df8409a4855faeeb69426ee206187a0111
ARG DINOV3_HOT_REF=dee030eec385aca489be152eea2a96ddf255a2ef
ARG DINOV3_REF=6876159a11b4df116f30f667f8c9888617df0751
ARG FAIR_PY_OPS_VERSION=0.3.16
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --no-sources \
    "dda @ git+https://github.com/hotosm/dda@${DDA_REF}" \
    "dinov3-hot @ git+https://github.com/kshitijrajsharma/dinov3-hot-ablation@${DINOV3_HOT_REF}" \
    "fair-py-ops[k8s,serve]==${FAIR_PY_OPS_VERSION}" \
    && uv pip install --no-deps "git+https://github.com/facebookresearch/dinov3@${DINOV3_REF}"

FROM python:3.13.13-slim-trixie AS runtime
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libexpat1 libgdal36 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

COPY models/dinov3_damage models/dinov3_damage
COPY models/conftest.py models/conftest.py
COPY models/test_integration.py models/test_integration.py

ENTRYPOINT ["/usr/local/bin/python"]

# Test stage: pytest extras for CI.
FROM runtime AS test
COPY --from=ghcr.io/astral-sh/uv:0.11.8 /uv /usr/local/bin/uv
ARG FAIR_PY_OPS_VERSION=0.3.16
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system "fair-py-ops[test]==${FAIR_PY_OPS_VERSION}"

# Inference builder: distroless-ready venv. dda installs with --no-deps beside the serving packages its ONNX path imports.
FROM ghcr.io/astral-sh/uv:0.11.8-trixie-slim AS inference-builder
ENV UV_PYTHON_INSTALL_DIR=/python UV_PYTHON_PREFERENCE=only-managed
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN uv python install 3.13
WORKDIR /app
ARG DDA_REF=c0c709df8409a4855faeeb69426ee206187a0111
ARG FAIR_PY_OPS_VERSION=0.3.16
RUN --mount=type=cache,target=/python/.cache/uv \
    uv venv && uv pip install \
    "fair-py-ops[serve]==${FAIR_PY_OPS_VERSION}" \
    onnxruntime==1.28.0 \
    geomltoolkits==2.1.0 \
    omegaconf==2.3.0 \
    && uv pip install --no-deps "dda @ git+https://github.com/hotosm/dda@${DDA_REF}"
COPY models/dinov3_damage/pipeline.py /app/models/dinov3_damage/pipeline.py

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
