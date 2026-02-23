# Runtime: minimal deps only
FROM ghcr.io/astral-sh/uv:python3.13-trixie-slim

WORKDIR /app

RUN uv venv /opt/venv
RUN . /opt/venv/bin/activate && \
    uv pip install torch torchgeo "pystac[validation]" zenml

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY fair/ fair/
COPY models/example_unet models/example_unet


RUN python -c "import torch, torchgeo; from models.example_unet.pipeline import training_pipeline; print('UNet pipeline ready')"

ENTRYPOINT ["/opt/venv/bin/python"]
