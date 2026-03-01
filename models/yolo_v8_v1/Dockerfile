# ============================================================================
# YOLOv8-v1 building footprint segmentation — runtime image
# ============================================================================
#
# Base: tensorflow/tensorflow:2.13.0-gpu
#   - Ubuntu 20.04, Python 3.11, CUDA 11.8, cuDNN 8.6
#   - TensorFlow 2.13 is required because hot-fair-utilities depends on
#     ramp-fair==0.1.2, which imports tensorflow.keras for RAMP inference.
#     YOLO inference does NOT use TF; it is pulled in transitively.
#
# PyTorch 2.1.0 is installed on top (CUDA 11.8 wheel).
# hot-fair-utilities wraps both runtimes in a single package.
#
# Known constraints:
#   - hot-fair-utilities pins shapely==1.8.0. If a future dependency requires
#     shapely>=2.0 this Dockerfile will need a resolution strategy (e.g.
#     install hot-fair-utilities with --no-deps and pin each dep manually).
#   - TF and PyTorch each install their own CUDA runtime libraries. The image
#     will be large (~8 GB compressed). Use the base GPU instance in ZenML
#     stacks to avoid re-downloading on every run.
# ============================================================================

FROM tensorflow/tensorflow:2.13.0-gpu

# -- System dependencies --
# Required by: rasterio/geopandas/shapely/rtree + hot_fair_utilities.georeferencing (osgeo.gdal)
# Also includes minimal X11/GL libs so `cv2` can import inside a headless container.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        gdal-bin \
        libgdal-dev \
        libgeos-dev \
        libspatialindex-dev \
        libglib2.0-0 \
        libgl1 \
        libxcb1 \
        libxcb-xfixes0 \
        libxcb-shm0 \
        libxcb-randr0 \
        libxcb-render0 \
        libxcb-xinerama0 \
        libxkbcommon0 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

# -- NumPy + GDAL Python binding (must match system GDAL version) --
RUN pip install --no-cache-dir "numpy>=1.24,<2.0" && \
    pip install --no-cache-dir "GDAL==$(gdal-config --version)"

# -- PyTorch 2.1.0 (CUDA 11.8) --
# Installed before hot-fair-utilities so pip sees an already-satisfied torch
# constraint and does not downgrade to a CPU-only build.
RUN pip install --no-cache-dir \
        torch==2.1.0 \
        torchvision==0.16.0 \
        torchaudio==2.1.0 \
        --index-url https://download.pytorch.org/whl/cu118

# -- Python runtime deps used by hot_fair_utilities YOLO training --
# hot-fair-utilities imports these at runtime but does not list all of them in its pyproject.
RUN pip install --no-cache-dir \
        "requests" \
        "matplotlib" \
        "pyyaml" \
        "gdown>=5.0"

# -- Solaris (geospatial ML toolkit, vendored inside ramp-code-fair) -----------
# Required because ramp-fair (installed transitively by hot-fair-utilities) imports
# solaris at module initialisation time via:
#   ramp/utils/__init__.py  ->  eval_utils.py  ->  solaris.eval.iou
# Solaris is not on PyPI; install from the ramp-code-fair GitHub source.
# This is the same copy vendored locally at fAIr-utilities/docker/solaris/.
RUN pip install --no-cache-dir \
        git+https://github.com/hotosm/ramp-code-fair.git#subdirectory=solaris

# -- hot-fair-utilities (YOLO + geospatial pipeline functions) --
RUN pip install --no-cache-dir "hot-fair-utilities==2.0.12"

# -- Compatibility pin --
# hot_fair_utilities.preprocess -> geopandas.read_file() expects Fiona API
# that exposes fiona.path. Fiona 1.10+ can break this call path.
RUN pip install --no-cache-dir "fiona<1.10"

# -- ZenML orchestration runtime --
# RUN pip install --no-cache-dir "zenml>=0.93.3,<1.0.0"

# -- Workspace --
WORKDIR /workspace

# Set RAMP_HOME to workspace root (required by ramp-fair / hot-fair-utilities)
ENV RAMP_HOME=/workspace
