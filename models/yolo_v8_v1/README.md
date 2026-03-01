# YOLOv8-v1 Building Footprint Segmentation

YOLOv8-Seg fine-tuned on OpenAerialMap (OAM) chips for building instance
segmentation, wrapped in the `hot-fair-utilities` processing stack.

## Model pack contents

| File | Purpose |
|---|---|
| `pipeline.py` | ZenML `@step` / `@pipeline` entrypoints (pre → train → infer → post) |
| `stac-item.json` | STAC MLM item — registers the model in the fAIr catalog |
| `Dockerfile` | Isolated runtime (TF 2.13 + PyTorch 2.1 + hot-fair-utilities) |

Data, weights, and configs are **not** stored here:

- **Weights** → S3 (referenced by `stac-item.json` `assets.model.href`)
- **Training data** → S3 (referenced by the dataset STAC item)
- **Hyperparameters** → `stac-item.json` `mlm:hyperparameters`

## Pipeline steps

```text
input chips (PNG/TIF) + labels.geojson
        │
        ▼
[run_preprocessing]  hot_fair_utilities.preprocess + yolo_format
        │  yolo_dir/
        ▼
[train_model]        hot_fair_utilities.training.yolo_v8_v1.train
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
from models.yolo_v8_v1.pipeline import training_pipeline, inference_pipeline

# Training (use data/sample or your dataset)
training_pipeline(
    input_path="data/sample/yolo_work/input",
    output_path="data/sample/yolo_work/out",
    weights_path="yolov8s_v1-seg-best.pt",
    epochs=20,
    batch_size=16,
    pc=2.0,
)

# Inference (model_uri from STAC or local path)
inference_pipeline(
    model_uri="https://github.com/hotosm/fAIr-utilities/raw/refs/heads/master/yolov8s_v1-seg-best.pt",
    input_path="data/sample/yolo_work/prediction/input",
    prediction_path="data/sample/yolo_work/prediction/output",
    output_geojson="data/sample/yolo_work/prediction/output/prediction.geojson",
    confidence=0.5,
)
```

## Building the Docker image

```bash
docker build -t yolo-v8-v1:latest \
    -f models/yolo_v8_v1/Dockerfile .
```

## Docker runtime tests

The `tests/` folder contains an end-to-end smoke suite that runs inside
the container and validates:

- imports (`gdal`, `cv2`, `ultralytics`, `hot_fair_utilities`)
- dataset layout checks
- preprocess + YOLO dataset formatting
- short training run (checkpoint creation)
- inference output generation
- polygonization to GeoJSON

Run on Windows PowerShell:

```powershell
.\models\yolo_v8_v1\tests\run_docker_tests.ps1 -BuildImage
```

Run on Linux/macOS/WSL:

```bash
BUILD_IMAGE=1 ./models/yolo_v8_v1/tests/run_docker_tests.sh .
```

Tests use `data/sample` (train/oam + train/osm + predict/oam). Run from fAIr-models repo root.

Both scripts:

- build the `yolo-v8-v1:latest` image (when the build flag is set),
- run the smoke test container with `--gpus all` (if available),
- allocate `--shm-size 1g` to avoid PyTorch DataLoader `bus error` issues.

If you prefer to run the test manually, this is the equivalent one-liner
(from the `fAIr-models` repo root):

```powershell
docker run --rm --gpus all `
  --shm-size 1g `
  -v "${PWD}:/workspace" `
  yolo-v8-v1:latest `
  python /workspace/models/yolo_v8_v1/tests/inside_container_smoke_test.py `
    --dataset-root /workspace/models/yolo_v8_v1/ramp-data/yolo_v8_v1_sample
```

Register as a ZenML Docker settings image in `stacks/` before running
pipelines on a remote stack.

## Registering in the STAC catalog

```python
from fair.stac.catalog_manager import CatalogManager

cm = CatalogManager()
cm.register_model("models/yolo_v8_v1/stac-item.json")
```

## Key design decisions

**Why `hot-fair-utilities` and not inline model code?**
All preprocessing, training, inference, and polygonization logic lives in
`hot-fair-utilities`. This pack is intentionally thin: it declares *how* to
run the model (pipeline.py) and *what* it is (stac-item.json). Upgrading
the model logic means bumping the `hot-fair-utilities` pin in the Dockerfile,
not changing this pack.

**Why lazy imports in pipeline.py?**
`hot-fair-utilities` pulls in TensorFlow, PyTorch, and Ultralytics. These
are not installed in the `fAIr-models` host environment (which only needs
`pystac` and `zenml`). Lazy imports inside function bodies keep the module
importable for STAC validation and catalog operations without triggering
heavy dependency errors.

**Why one Dockerfile per model?**
Each model has its own dependency graph (different TF/PyTorch/CUDA versions,
different geospatial stacks). Per-model images prevent version conflicts
between contributors and allow independent upgrades.
