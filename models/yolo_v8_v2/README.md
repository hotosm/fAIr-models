# YOLOv8-v2 Building Footprint Segmentation

YOLOv8-Seg fine-tuned on OpenAerialMap (OAM) chips for building instance
segmentation, wrapped in the `hot-fair-utilities` processing stack.

## Model pack contents

| File | Purpose |
| --- | --- |
| `pipeline.py` | ZenML `@step` / `@pipeline` entrypoints (pre → train → infer → post) |
| `stac-item.json` | STAC MLM item — registers the model in the fAIr catalog |
| `Dockerfile` | Isolated runtime (thin layer on `fair-utilities-yolo` base image) |

Data, weights, and configs are **not** stored here:

- **Weights** → S3 / GitHub (referenced by `stac-item.json` `assets.model.href`)
- **Training data** → S3 (referenced by the dataset STAC item)
- **Hyperparameters** → `stac-item.json` `mlm:hyperparameters`

## Architecture: what comes from where

This model pack is intentionally **thin**. All heavy logic lives in `hot-fair-utilities`:

| Pipeline step | Delegates to | Module |
|---|---|---|
| Preprocessing | `hot_fair_utilities.preprocess()` | Georeference + rasterize + clip labels |
| YOLO formatting | `hot_fair_utilities.preprocessing.yolo_v8.yolo_format()` | Convert to YOLO dataset layout |
| Training | `hot_fair_utilities.training.yolo_v8.train()` | YOLOv8-Seg fine-tuning with pos-weight |
| Inference | `hot_fair_utilities.predict()` | fairpredictor-based prediction + georeferencing |
| Postprocessing | `hot_fair_utilities.polygonize()` | geomltoolkits merge + vectorize + merge polygons |

## Pipeline steps

```text
input chips (PNG/TIF) + labels.geojson
        │
        ▼
[run_preprocessing]  hot_fair_utilities.preprocess + yolo_format
        │  yolo_dir/
        ▼
[train_model]        hot_fair_utilities.training.yolo_v8.train
        │  best.pt  (IoU% logged to ZenML)
        ▼
[run_inference]      hot_fair_utilities.predict
        │  predicted GeoTIFFs
        ▼
[run_postprocessing] hot_fair_utilities.polygonize
        │
        ▼
output prediction.geojson  (EPSG:4326 building footprints)
```

## Running locally (outside ZenML)

```python
from models.yolo_v8_v2.pipeline import training_pipeline, inference_pipeline

# Training (use data/sample or your dataset)
training_pipeline(
    input_path="data/sample/yolo_work/input",
    output_path="data/sample/yolo_work/out",
)

# Inference (model_uri from STAC)
inference_pipeline(
    model_uri="https://github.com/hotosm/fAIr-utilities/raw/refs/heads/master/yolov8s_v2-seg.pt",
    input_path="data/sample/yolo_work/prediction/input",
    prediction_path="data/sample/yolo_work/prediction/output",
    output_geojson="data/sample/yolo_work/prediction/output/prediction.geojson",
    confidence=0.5,
)
```

## Building the Docker image

### Bash (Linux/macOS/WSL)

```bash
# CPU
docker build -t yolo-v8-v2:cpu \
    -f models/yolo_v8_v2/Dockerfile .

# GPU
docker build --build-arg BASE_IMAGE=ghcr.io/hotosm/fair-utilities-yolo:gpu-latest \
    -t yolo-v8-v2:gpu \
    -f models/yolo_v8_v2/Dockerfile .
```

### PowerShell (Windows)

```powershell
# CPU
docker build -t yolo-v8-v2:cpu `
    -f models/yolo_v8_v2/Dockerfile .

# GPU
docker build --build-arg BASE_IMAGE=ghcr.io/hotosm/fair-utilities-yolo:gpu-latest `
    -t yolo-v8-v2:gpu `
    -f models/yolo_v8_v2/Dockerfile .
```

## Docker runtime tests

The `tests/` folder contains an end-to-end smoke suite that runs inside
the container and validates:

- imports (`gdal`, `cv2`, `ultralytics`, `hot_fair_utilities`)
- dataset layout checks
- preprocess + YOLO dataset formatting
- short training run (checkpoint creation)
- `resolve_model_href` for local .pt files
- inference output generation
- polygonization to GeoJSON
- intermediate artifact checks

Run on Windows PowerShell:

```powershell
.\models\yolo_v8_v2\tests\run_docker_tests.ps1 -Image yolo-v8-v2:cpu
# or with build:
$env:BUILD_IMAGE = "1"; $env:CPU_ONLY = "1"; .\models\yolo_v8_v2\tests\run_docker_tests.ps1
```

Run on Linux/macOS/WSL:

```bash
BUILD_IMAGE=1 CPU_ONLY=1 ./models/yolo_v8_v2/tests/run_docker_tests.sh .
```

Tests use `data/sample`. Run from fAIr-models repo root.

Both scripts:

- build the image (when the build flag is set),
- run the smoke test container with `--gpus all` (if available),
- allocate `--shm-size 1g` to avoid PyTorch DataLoader `bus error` issues.

If you prefer to run the test manually, this is the equivalent one-liner
(from the `fAIr-models` repo root):

```powershell
docker run --rm --shm-size 1g `
  -v "${PWD}:/workspace" `
  yolo-v8-v2:cpu `
  python /workspace/models/yolo_v8_v2/tests/inside_container_smoke_test.py `
    --dataset-root /workspace/data/sample
```

Register as a ZenML Docker settings image in `stacks/` before running
pipelines on a remote stack.

## Registering in the STAC catalog

```python
from fair.stac.catalog_manager import CatalogManager

cm = CatalogManager()
cm.register_model("models/yolo_v8_v2/stac-item.json")
```

## Key design decisions

**Why `hot-fair-utilities` and not inline model code?**
All preprocessing, training, inference, and polygonization logic lives in
`hot-fair-utilities`. This pack is intentionally thin: it declares *how* to
run the model (pipeline.py) and *what* it is (stac-item.json). Upgrading
the model logic means bumping the `hot-fair-utilities` pin in the base image,
not changing this pack.

**Why lazy imports in pipeline.py?**
`hot-fair-utilities` pulls in PyTorch, Ultralytics, and GDAL. These
are not installed in the `fAIr-models` host environment (which only needs
`pystac` and `zenml`). Lazy imports inside function bodies keep the module
importable for STAC validation and catalog operations without triggering
heavy dependency errors.

**Why one Dockerfile per model?**
Each model has its own dependency graph (different TF/PyTorch/CUDA versions,
different geospatial stacks). Per-model images prevent version conflicts
between contributors and allow independent upgrades.

**Why a thin Dockerfile on top of a base image?**
The `fAIr-utilities` project publishes base images (`Dockerfile.yolo`) with
all heavy dependencies (PyTorch, Ultralytics, GDAL, hot-fair-utilities).
The model Dockerfile only adds orchestration tools (`fair-py-ops`, `zenml`,
`pystac`, `fairpredictor`). This keeps model images small and aligned with
the upstream dependency split.
