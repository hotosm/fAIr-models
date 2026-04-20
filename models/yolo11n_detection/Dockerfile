# Builder stage: install dependencies
FROM ghcr.io/astral-sh/uv:python3.13-trixie-slim AS builder
ENV UV_SYSTEM_PYTHON=1 UV_LINK_MODE=copy

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgdal-dev libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md fair_zenml_patch.pth /tmp/fair-src/
COPY fair /tmp/fair-src/fair
RUN --mount=type=cache,target=/root/.cache/uv \
    SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 \
    uv pip install \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    --index-strategy unsafe-best-match \
    torch==2.10.0 \
    "ultralytics>=8.4.0" \
    rasterio \
    pyproj \
    onnx \
    onnxscript \
    /tmp/fair-src[k8s,serve]

# Runtime stage: minimal image
FROM python:3.13-slim-trixie AS runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libexpat1 libgdal36 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY models/yolo11n_detection models/yolo11n_detection
COPY models/conftest.py models/conftest.py
COPY models/test_integration.py models/test_integration.py

ENTRYPOINT ["/usr/local/bin/python"]

# Test stage: adds test dependencies for CI
FROM runtime AS test
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
COPY --from=builder /tmp/fair-src /tmp/fair-src
RUN --mount=type=cache,target=/root/.cache/uv \
    SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 \
    uv pip install --system /tmp/fair-src[test]

# Inference builder stage: distroless-ready venv with serving deps
FROM ghcr.io/astral-sh/uv:bookworm-slim AS inference-builder
ENV UV_PYTHON_INSTALL_DIR=/python UV_PYTHON_PREFERENCE=only-managed
RUN uv python install 3.13
WORKDIR /app
COPY pyproject.toml README.md fair_zenml_patch.pth /tmp/fair-src/
COPY fair /tmp/fair-src/fair
RUN --mount=type=cache,target=/root/.cache/uv \
    export SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 && \
    uv venv && uv pip install \
    /tmp/fair-src[serve] \
    rasterio \
    pyproj \
    numpy \
    Pillow
COPY models/yolo11n_detection/pipeline.py /app/models/yolo11n_detection/pipeline.py

# Inference stage: distroless runtime for live serving
FROM gcr.io/distroless/cc-debian12:nonroot AS inference
COPY --from=inference-builder /python /python
COPY --from=inference-builder /app/.venv /app/.venv
COPY --from=inference-builder /app/models /app/models
ENV PATH="/app/.venv/bin:$PATH" PYTHONPATH=/app
EXPOSE 8080
CMD ["uvicorn", "fair.serve.base:create_app", "--factory", "--host", "0.0.0.0", "--port", "8080"]
