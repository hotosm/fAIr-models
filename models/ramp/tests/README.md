# RAMP Smoke Tests — In-Depth Guide

This folder contains end-to-end smoke tests that validate the RAMP Docker runtime.
Tests run **inside** the container after the repo is mounted at `/workspace`.

## Test Files Overview

| File | Purpose |
| --- | --- |
| `inside_container_smoke_test.py` | Main test script. Run inside the container. Validates 7 stages (imports → preprocessing → train → inference → polygonization). |
| `run_docker_tests.ps1` | PowerShell runner: builds image (optional), runs container, executes the smoke test. |
| `run_docker_tests.sh` | Bash runner: same as above for Linux/macOS. |
| `test_plan.yaml` | Declarative description of what each stage checks. Used for documentation and CI planning. |

## Data Source: `data/sample`

Tests use **`data/sample`** (fAIr-models shared sample) instead of model-specific `ramp-data`.

**Layout:**

```text
data/sample/
├── train/oam/          # OAM GeoTIFF tiles (OAM-{x}-{y}-{z}.tif)
├── train/osm/          # OSM building labels (*.geojson)
└── predict/oam/        # (Optional) images for inference
```

**Adapter logic:** `hot_fair_utilities.preprocess` expects `input/*.png` and `input/labels.geojson`.
The test script detects the `data/sample` layout and:

1. Creates `data/sample/ramp_work/input/`
2. Converts `train/oam/*.tif` → PNG (rasterio + PIL)
3. Merges `train/osm/*.geojson` → `labels.geojson`
4. Runs the pipeline on `ramp_work/input/`

Outputs go to `data/sample/ramp_work/preprocessed_test/` and `prediction_test/`.

## Seven Test Stages (inside_container_smoke_test.py)

### 1. Critical Imports

Verifies runtime packages: `tensorflow`, `segmentation_models`, `ramp`, `hot_fair_utilities`, `osgeo.gdal`, `solaris`.
Sets `segmentation_models` to use `tf.keras` backend.

### 2. Input Dataset Layout

- **data/sample**: Requires `train/oam/`, `train/osm/`, at least one `OAM-*.tif`, and one `*.geojson`.
- **Legacy**: Requires `input/`, `input/labels.geojson`, and `input/*.png`.

### 3. Preprocessing

Calls `hot_fair_utilities.preprocess()` with:

- `georeference_images=True` — converts PNG to GeoTIFF (EPSG:3857)
- `rasterize=True`, `rasterize_options=["binary"]`
- `multimasks=True` — 4-class masks (background, building, boundary, contact)

Checks: `chips/*.tif` and `multimasks/*.mask.tif` exist and counts match.

### 4. Train / Val Split

Shuffles chip/mask pairs, moves ~20% to `val_chips/` and `val_multimasks/`.
RAMP’s data generator needs separate train and validation directories.

### 5. Training Smoke Run

Builds EfficientNetB0 U-Net (`encoder_weights=None` to avoid 404 on pretrained weights).
Runs 2 epochs, saves SavedModel to `checkpoints/`.

### 6. Inference

Loads SavedModel, runs prediction on first 3 chips.
Produces `*.pred.tif` (4-class uint8 masks).

### 7. Polygonization

Uses `ramp.utils.mask_to_vec_utils` (GDAL Polygonize) to convert predicted masks to GeoJSON building footprints.
Checks for `*.geojson` in `prediction_test/vectors/`.

## How to Run

**Prerequisite:** Ensure `data/sample` exists with `train/oam/` and `train/osm/`.

### PowerShell (Windows)

```powershell
cd fAIr-models   # repo root

# Build image and run tests
.\models\ramp\tests\run_docker_tests.ps1 -BuildImage

# Or run tests only (image must already exist)
.\models\ramp\tests\run_docker_tests.ps1

# CPU-only (no GPU)
.\models\ramp\tests\run_docker_tests.ps1 -BuildImage -CpuOnly
```

### Bash (Linux / macOS)

```bash
cd fAIr-models

# Build and run
BUILD_IMAGE=1 ./models/ramp/tests/run_docker_tests.sh

# CPU-only
CPU_ONLY=1 BUILD_IMAGE=1 ./models/ramp/tests/run_docker_tests.sh
```

### Manually Inside Container

```bash
docker run --rm -v /path/to/fAIr-models:/workspace ramp-v1:gpu \
  python /workspace/models/ramp/tests/inside_container_smoke_test.py \
    --dataset-root /workspace/data/sample
```

### With Custom Dataset (Legacy Layout)

```bash
# Legacy layout: dataset/input/*.png and dataset/input/labels.geojson
docker run --rm -v /path/to/data:/workspace/data ramp-v1:gpu \
  python /workspace/models/ramp/tests/inside_container_smoke_test.py \
    --dataset-root /workspace/data/your_dataset
```

## What Gets Created

Running tests creates:

- `data/sample/ramp_work/input/` — prepared PNG chips + labels.geojson (data/sample only)
- `data/sample/ramp_work/preprocessed_test/` — chips, multimasks, val split, checkpoints
- `data/sample/ramp_work/prediction_test/output/` — `*.pred.tif`
- `data/sample/ramp_work/prediction_test/vectors/` — `*.geojson` building polygons

These paths are in `.gitignore`.
