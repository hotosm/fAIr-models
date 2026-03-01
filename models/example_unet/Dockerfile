# Builder stage: install dependencies
FROM ghcr.io/astral-sh/uv:python3.13-trixie-slim AS builder
ENV UV_SYSTEM_PYTHON=1 UV_LINK_MODE=copy

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install \
    torch==2.10.0 \
    torchgeo==0.9.0 \
    fair-py-ops \
    "mlflow>=2.1.1,<4" \
    "pypgstac[psycopg]>=0.9" \
    "pystac-client>=0.9" \
    "zenml[connectors-aws,connectors-kubernetes,s3fs]>=0.93.3"

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

# ZenML 0.93.3: ServerDatabaseType enum lacks "postgresql" variant, TODO : need to find a stable solution for this , can't keep patching it in the dockerfile
RUN sed -i 's/OTHER = "other"/OTHER = "other"\n    POSTGRESQL = "postgresql"/' \
    /usr/local/lib/python3.13/site-packages/zenml/models/v2/misc/server_models.py

COPY models/example_unet models/example_unet
COPY data/sample data/sample

ENTRYPOINT ["/usr/local/bin/python"]

