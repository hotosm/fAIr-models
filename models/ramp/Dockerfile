# Build instructions (from fAIr-models repo root):
#   GPU: docker build -f models/ramp/Dockerfile --build-arg BUILD_TYPE=gpu -t ramp-v1:gpu .
#   CPU: docker build -f models/ramp/Dockerfile --build-arg BUILD_TYPE=cpu -t ramp-v1:cpu .

ARG PY_VER=3.10
ARG TF_VER=2.9.3
ARG BUILD_TYPE=gpu
ARG CUDA_TAG=11.8.0-cudnn8-runtime-ubuntu22.04

# ==============================================================================
# === CPU base image (minimal) =================================================
FROM python:${PY_VER}-slim-bookworm AS cpu-base
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git build-essential gcc g++ python3-dev python3-rtree \
    gdal-bin libgdal-dev python3-gdal python3-opencv libspatialindex-dev libgeos-dev \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ==============================================================================
# === GPU base image (CUDA + runtime-only Python & GDAL) =======================
FROM nvidia/cuda:${CUDA_TAG} AS gpu-base
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git python3 python3-pip build-essential gcc g++ python3-dev python3-rtree python-is-python3 \
    gdal-bin libgdal-dev python3-gdal python3-opencv libspatialindex-dev libgeos-dev \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* && \
    python3 -m pip install --upgrade pip

# ==============================================================================
# === Builder stage (installs everything) ======================================
FROM ${BUILD_TYPE}-base AS builder
ENV DEBIAN_FRONTEND=noninteractive \
    TF_CPP_MIN_LOG_LEVEL=2
ARG TF_VER

ENV CPLUS_INCLUDE_PATH=/usr/include/gdal \
    C_INCLUDE_PATH=/usr/include/gdal

# Use pip cache and install Python packages
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir --upgrade pip more-itertools && \
    pip install --no-cache-dir "numpy>=1.22,<2.0" && \
    pip install --no-cache-dir "GDAL==$(gdal-config --version)" && \
    pip install --no-cache-dir tensorflow==${TF_VER} && \
    pip install --no-cache-dir \
        "efficientnet==1.0.0" \
        "image-classifiers==1.0.0" \
        "segmentation-models==1.0.1" && \
    pip install --no-cache-dir \
        "geopandas==0.10.2" \
        "rasterio>=1.3.10" \
        "Shapely>=1.8.1" \
        "pyproj==3.3.0" \
        "scikit-image==0.19.2" \
        "scikit-learn==1.0.2" \
        "scipy==1.8.0" \
        "tqdm==4.62.3" \
        "pandas==1.4.1" \
        "matplotlib==3.5.1" \
        "albumentations==1.0.3" \
        "pyyaml>=6.0" \
        "requests>=2.27.1" \
        "tinydb==4.7.0" \
        "gdown>=5.0"

# Compatibility pin for geopandas/raster stack
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir "fiona>=1.8.22,<1.10"

# Install solaris from ramp-code-fair (equivalent to Dockerfile.pb local install)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir \
    git+https://github.com/hotosm/ramp-code-fair.git#subdirectory=solaris

# Install scikit-fmm and ramp-fair
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir scikit-fmm && \
    pip install --no-cache-dir "ramp-fair==0.1.2"

# Install hot-fair-utilities for shared preprocessing path used by pipeline.py
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir "opencv-python-headless<=4.7.0.68" && \
    pip install --no-cache-dir "hot-fair-utilities==2.0.12" && \
    pip install --no-cache-dir "numpy>=1.22,<2.0"

# ==============================================================================
# === Final minimal runtime image ==============================================
FROM ${BUILD_TYPE}-base AS final
ENV DEBIAN_FRONTEND=noninteractive

ENV CPLUS_INCLUDE_PATH=/usr/include/gdal \
    C_INCLUDE_PATH=/usr/include/gdal \
    RAMP_HOME=/workspace

COPY --from=builder /usr/local /usr/local
COPY --from=builder /usr/lib/python*/ /usr/lib/python*/
COPY --from=builder /usr/include/gdal /usr/include/gdal

WORKDIR /workspace
