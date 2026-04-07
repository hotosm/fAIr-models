# RAMP — EfficientNetB0 + U-Net Building Semantic Segmentation

RAMP (Replicable AI for Microplanning) is an EfficientNetB0 encoder + U-Net decoder
for pixel-wise 4-class building segmentation on 256 × 256 px aerial image chips.

## Architecture overview

| Component | Detail |
| --- | --- |
| Encoder | EfficientNet-B0 (ImageNet pre-trained via segmentation_models) |
| Decoder | U-Net symmetric decoder with skip connections |
| Output | 4-class sparse categorical mask (channels-last, uint8) |
| Classes | 0=background, 1=building, 2=boundary, 3=contact-point |
| Loss | Sparse categorical crossentropy |
| Metric | Sparse categorical accuracy (`val_sparse_categorical_accuracy`) |
| Framework | TensorFlow 2.15.1 / Keras |

The boundary (class 2) and contact-point (class 3) channels help the model cleanly
separate adjacent buildings at inference time, even when they share a wall.

## Key difference from YOLO packs

| Stage | YOLO v8 v1/v2 | RAMP |
| --- | --- | --- |
| Preprocessing | `hot_fair_utilities.preprocess` | same |
| Training | `hot_fair_utilities.training.yolo_v8_*` (ultralytics) | `ramp.training.*` (TF/Keras) |
| Inference | `hot_fair_utilities.predict` (ultralytics) | `fairpredictor.predictor.prediction.run_prediction` |
| Postprocessing | `hot_fair_utilities.polygonize` (AutoBFE) | `geomltoolkits` vectorization + geometry validation (writes `predictions.geojson`) |

## Model pack contents

| File | Purpose |
| --- | --- |
| `pipeline.py` | ZenML `@step` / `@pipeline` entrypoints (pre → train → infer → post); `resolve_model_href()` for URLs |
| `stac-item.json` | STAC MLM item — model weights (mlm:model href), entrypoints; weights from local path or HTTP(S) `.zip` |
| `Dockerfile` | Runtime on top of `ghcr.io/hotosm/fair-utilities-ramp` (CPU/GPU); see [Docker image composition](#docker-image-composition) |

## Docker image composition

The RAMP image is **layered**, not a single monolithic install:

1. **Base (hot-fair-utilities on GHCR)**  
   TensorFlow, GDAL, `hot-fair-utilities` RAMP extras, fairpredictor, geomltoolkits, and related geospatial/ML stack. You do not build this locally; Docker pulls `cpu-latest` or `gpu-latest`.

2. **`fair-py-ops` (pinned)**  
   Installed in this Dockerfile to match `fAIr-models/pyproject.toml` (`fair-py-ops==0.0.6`). This is the long-term registry/orchestration contract for the repo.

3. **Temporary test-only packages (remove before production merge)**  
   The Dockerfile installs **`zenml[server]==0.93.3`** and **`pystac[validation]>=1.14.3`**, aligned with `pyproject.toml` (`zenml>=0.93.3`, `pystac[validation]>=1.14.3`).  
   **Why `[server]` on ZenML:** `pipeline.py` wraps logic in `@step`. Calling a step (e.g. `run_preprocessing(...)`) runs ZenML’s single-step pipeline, which initializes the default **SQL** zen store. That code path imports **`sqlalchemy_utils`** (and related SQL stack). Those are **not** included in a bare `pip install zenml==…` — they are part of ZenML’s **`server`** optional extra (same idea as the repo’s `[dependency-groups] local` → `zenml[server]`). Without `[server]`, you get `ModuleNotFoundError: No module named 'sqlalchemy_utils'`.  
   **`fAIr-utilities/docker/Dockerfile.ramp`** does not install ZenML, PySTAC, `fair-py-ops`, or this SQL stack; those are added only in this model Dockerfile layer.

   **Before you push production-oriented code**, delete the **second** `RUN pip install …` block in `models/ramp/Dockerfile` (the one that installs `zenml[server]` and `pystac`). Keep the `fair-py-ops` `RUN` unless your platform injects it another way.  
   After removal, in-container smoke tests that import `pipeline.py` will fail unless you refactor (e.g. split core vs ZenML) or run tests only on infrastructure where ZenML is pre-installed.

**STAC in code vs `pystac` on PyPI:**  
Training hyperparameters in this pack are read from **`stac-item.json` with plain `json`** — you do not need the `pystac` Python package for that path. The Dockerfile adds `pystac[validation]` **only for testing parity** with the repo’s declared dependencies; production can rely on STAC-as-JSON plus platform tooling.

**`SM_FRAMEWORK=tf.keras`:**  
The `segmentation_models` / `efficientnet` stack reads `SM_FRAMEWORK` when the package is **first imported**. The default path uses standalone `keras` and `efficientnet.keras`, which rely on removed Keras 2 APIs (`keras.utils.generic_utils`) and fail on TensorFlow 2.15+ (bundled Keras 3). The Dockerfile sets `SM_FRAMEWORK=tf.keras` so `efficientnet.tfkeras` is used. Do not unset this in the RAMP container unless you pin an older TensorFlow/Keras stack.

## Data directory layout

**Option A: Use shared `data/sample`** (recommended for tests)

The smoke tests use `data/sample` (train/oam + train/osm). The test script
auto-converts OAM GeoTIFFs to PNG and merges OSM labels into the RAMP format.

**Option B: Legacy layout**

```text
dataset/
├── input/
│   ├── *.png                     # OAM chips (PNG, no geo-reference)
│   └── labels.geojson            # combined building polygon labels
├── preprocessed/                 # created by preprocess (hot_fair_utilities)
│   ├── chips/                    # georeferenced .tif chips (EPSG:3857)
│   ├── labels/                   # per-chip .geojson labels
│   ├── multimasks/               # 4-class .mask.tif targets
│   └── (no training outputs here) # training uses a separate work dir
├── ramp_training_work/           # created by train_ramp_model
│   ├── chips/                    # training chips (copied from preprocessed)
│   ├── multimasks/               # training masks (copied from preprocessed)
│   ├── val-chips/                # validation split chips (hyphenated)
│   ├── val-multimasks/           # validation split masks (hyphenated)
│   └── model-checkpts/           # SavedModel checkpoints + best model selection
└── prediction/
    ├── input/                    # chips for inference (GeoTIFFs; typically copy from preprocessed/chips/)
    ├── output/                   # fairpredictor outputs (includes georeference/*.tif)
    └── vectors/                  # merged GeoJSON: predictions.geojson (+ merged_prediction_mask.tif, tmp/)
```

## Pipeline steps

```text
input/  (PNG chips + labels.geojson)
        │
        ▼
[run_preprocessing]  hot_fair_utilities.preprocess  (georeference + multimask; returns chip/mask arrays)
        │  preprocessed/chips/  +  preprocessed/multimasks/
        ▼
[train_model]        hot_fair_utilities.training.ramp.*  (EfficientNetB0 U-Net, TF/Keras)
        │  best checkpoint (SavedModel) loaded as tf.keras.Model + val_sparse_categorical_accuracy
        ▼
[run_inference]      fairpredictor.run_prediction  (georeferenced prediction rasters)
        │  prediction/output/georeference/*.tif
        ▼
[run_postprocessing] geomltoolkits vectorization + validation  (writes predictions.geojson; returns dict)
        │
        ▼
prediction/vectors/predictions.geojson  (merged building footprints)
```

## Running locally (outside ZenML)

```python
from models.ramp.pipeline import infer_ramp_model, train_ramp_model

# Training (expects you've already run preprocessing to produce chips/ + multimasks/)
trained_model = train_ramp_model(
    data_base_path="data/sample/ramp_work",
    preprocessed_path="data/sample/ramp_work/preprocessed_test",
    stac_item_path="models/ramp/stac-item.json",
)

# Inference run (model_uri can be a local SavedModel dir, an HTTP(S) .zip, or a tf.keras.Model)
final_geojson = infer_ramp_model(
    model_uri=trained_model,
    input_path="data/sample/ramp_work/preprocessed_test/chips",
    prediction_path="data/sample/ramp_work/prediction_test/output",
    output_dir="data/sample/ramp_work/prediction_test/vectors",
)
# final_geojson is a dict (FeatureCollection-like). predictions.geojson is also written under output_dir.
```

## Building the Docker image

The `fAIr-utilities` team publishes pre-built base images to GitHub Container Registry (GHCR) on
every push to `master`. You **do not need to build or clone the `fAIr-utilities` repo yourself**.
Docker pulls the image automatically.

| Flavour | GHCR image |
| --- | --- |
| CPU (default) | `ghcr.io/hotosm/fair-utilities-ramp:cpu-latest` |
| GPU (CUDA) | `ghcr.io/hotosm/fair-utilities-ramp:gpu-latest` |

Each build adds (see Dockerfile comments):

- `fair-py-ops==0.0.6` (from `pyproject.toml`)
- **Temporary:** `zenml[server]==0.93.3` and `pystac[validation]>=1.14.3` — the `[server]` extra pulls the SQL stack (`sqlalchemy-utils`, etc.) required when `@step` runs; remove the dedicated `RUN` before a production merge if slim images must not carry orchestration deps (see [Docker image composition](#docker-image-composition)).

```bash
# CPU image (base image pulled from GHCR automatically — no --build-arg needed)
docker build -t ramp-v1:cpu -f models/ramp/Dockerfile .

# GPU image (override the ARG to select the GPU base)
docker build -t ramp-v1:gpu \
    --build-arg BASE_IMAGE=ghcr.io/hotosm/fair-utilities-ramp:gpu-latest \
    -f models/ramp/Dockerfile .
```

```powershell
# PowerShell (Windows) — do NOT use "\" for line continuation
docker build -t ramp-v1:cpu -f models/ramp/Dockerfile .
docker build -t ramp-v1:gpu --build-arg BASE_IMAGE=ghcr.io/hotosm/fair-utilities-ramp:gpu-latest -f models/ramp/Dockerfile .
```

> **Testing unreleased `fAIr-utilities` code?**  If GHCR doesn't yet contain a feature you need,
> add a `pip install` from git inside your Dockerfile layer:
> ```dockerfile
> RUN pip install --no-cache-dir \
>     "hot-fair-utilities @ git+https://github.com/hotosm/fAIr-utilities.git@<branch-or-sha>"
> ```
> You do not need to build the base image yourself.

## Running the smoke tests

The in-container script `models/ramp/tests/inside_container_smoke_test.py` imports `pipeline.py`,
which loads **ZenML** at import time. For that to work inside Docker, the image must either include
the temporary **ZenML + PySTAC** `RUN` in the Dockerfile (current approach for local/CI testing) or
you must refactor tests / pipeline imports (see [Docker image composition](#docker-image-composition)).

```powershell
# PowerShell (Windows)
$env:BUILD_IMAGE = "1"
$env:CPU_ONLY = "1"
.\models\ramp\tests\run_docker_tests.ps1
```

```bash
# Bash (Linux / macOS / Git Bash)
BUILD_IMAGE=1 CPU_ONLY=1 ./models/ramp/tests/run_docker_tests.sh
```

> **Note**: The smoke tests use `data/sample` (train/oam OAM tiles + train/osm labels).
> Run from the fAIr-models repo root so `/workspace/data/sample` is available in the container.

### Running the smoke script directly (after building the image)

If you already built the image (see “Building the Docker image”), you can run the smoke test script directly:
`models/ramp/tests/inside_container_smoke_test.py`.

```bash
# CPU image
docker run --rm -v "$(pwd):/workspace" ramp-v1:cpu \
  python /workspace/models/ramp/tests/inside_container_smoke_test.py \
    --dataset-root /workspace/data/sample \
    --epochs 2 --batch-size 4 --backbone efficientnetb0

# GPU image
docker run --rm --gpus all -v "$(pwd):/workspace" ramp-v1:gpu \
  python /workspace/models/ramp/tests/inside_container_smoke_test.py \
    --dataset-root /workspace/data/sample \
    --epochs 2 --batch-size 4 --backbone efficientnetb0
```

```powershell
# PowerShell (Windows) — same idea, no Bash "\" line continuation
docker run --rm -v "${PWD}:/workspace" ramp-v1:cpu python /workspace/models/ramp/tests/inside_container_smoke_test.py --dataset-root /workspace/data/sample --epochs 2 --batch-size 4 --backbone efficientnetb0

docker run --rm --gpus all -v "${PWD}:/workspace" ramp-v1:gpu python /workspace/models/ramp/tests/inside_container_smoke_test.py --dataset-root /workspace/data/sample --epochs 2 --batch-size 4 --backbone efficientnetb0
```

If you want to test a different dataset layout, point `--dataset-root` at a directory that contains either:

- `train/oam/*.tif` + `train/osm/*.geojson` (sample-style), or
- `input/*.png` + `input/labels.geojson` (legacy-style)

## Model weights (STAC mlm:model asset)

The STAC Item's `assets.model.href` points to pretrained weights. Supported sources:

| Source | Example |
| --- | --- |
| Local SavedModel directory | `/workspace/ramp-data/baseline` |
| HTTP(S) `.zip` containing a SavedModel | `https://example.com/ramp_model.zip` |

 For remote weights, publish an HTTP(S) `.zip` that contains a SavedModel directory with `saved_model.pb` and `variables/`. Downloaded zips are cached under `/workspace/.ramp_model_cache/`.

## Registering in the STAC catalog

```python
from fair.stac.catalog_manager import CatalogManager

cm = CatalogManager()
cm.register_model("models/ramp/stac-item.json")
```

## Dependencies from hot_fair_utilities

This pack uses **both preprocessing and training** paths from `hot-fair-utilities`:

| Used | Not used |
| --- | --- |
| `hot_fair_utilities.preprocess` (georeference + multimasks) | `hot_fair_utilities.predict` (YOLO / ultralytics) |
| `hot_fair_utilities.training.ramp.*` (RAMP_CONFIG, split/train helpers) | `hot_fair_utilities.polygonize` (AutoBFE, YOLO output) |

The `ultralytics` and `torch` packages are installed transitively by
`hot-fair-utilities` but are never imported at runtime for RAMP.

## Key design decisions

**Why ramp-fair and not inline model code?**
`ramp-fair` provides the complete EfficientNetB0 U-Net + training utilities.
This pack is thin: it declares *how* to run the model (pipeline.py) and
*what* it is (stac-item.json). Upgrading means bumping the `ramp-fair` pin.

**Where do “ramp” + “solaris” come from?**
They are pulled in via the `hot-fair-utilities[ramp]` / `[ramp-gpu]` extras in the Dockerfile. The RAMP runtime is intentionally installed as a consistent bundle inside the image so local machines don’t need to compile the full stack.

**Why is validation split handled in training?**
RAMP training expects explicit train/val directories. `train_ramp_model` calls `split_training_2_validation`, creating `ramp_training_work/val-chips` and `ramp_training_work/val-multimasks` under a dedicated training work directory.

**Why one Dockerfile per model?**
The RAMP stack depends on TensorFlow + GDAL + geospatial libs with tight version coupling. Keeping a per-model image prevents version conflicts and makes the runtime reproducible.
