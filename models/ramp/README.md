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
| Framework | TensorFlow 2.9.3 / Keras |

The boundary (class 2) and contact-point (class 3) channels help the model cleanly
separate adjacent buildings at inference time, even when they share a wall.

## Key difference from YOLO packs

| Stage | YOLO v8 v1/v2 | RAMP |
| --- | --- | --- |
| Preprocessing | `hot_fair_utilities.preprocess` | same |
| Training | `hot_fair_utilities.training.yolo_v8_*` (ultralytics) | `ramp.training.*` (TF/Keras) |
| Inference | `hot_fair_utilities.predict` (ultralytics) | `tf.keras.models.load_model` |
| Postprocessing | `hot_fair_utilities.polygonize` (AutoBFE) | `ramp.utils.mask_to_vec_utils` (GDAL) |

## Model pack contents

| File | Purpose |
| --- | --- |
| `pipeline.py` | ZenML `@step` / `@pipeline` entrypoints (pre → train → infer → post); `resolve_model_href()` for URLs |
| `stac-item.json` | STAC MLM item — model weights (mlm:model href), entrypoints; weights from Google Drive/S3 |
| `Dockerfile` | Isolated runtime (CUDA 11.8 + TF 2.9.3 + ramp-fair + hot-fair-utilities + gdown) |

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
├── preprocessed/                 # created by run_preprocessing
│   ├── chips/                    # georeferenced .tif chips (EPSG:3857)
│   ├── labels/                   # per-chip .geojson labels
│   ├── multimasks/               # 4-class .mask.tif targets
│   ├── val_chips/                # validation split chips
│   ├── val_multimasks/           # validation split masks
│   └── checkpoints/<timestamp>/  # Keras SavedModel checkpoints
└── prediction/
    ├── input/                    # chips for inference (GeoTIFFs; typically copy from preprocessed/chips/)
    ├── output/                   # .pred.tif predicted masks
    └── vectors/                  # per-chip .geojson building polygons
```

## Pipeline steps

```text
input/  (PNG chips + labels.geojson)
        │
        ▼
[run_preprocessing]  hot_fair_utilities.preprocess  (georeference + multimask)
        │  preprocessed/chips/  +  preprocessed/multimasks/
        ▼
[train_model]        ramp.training.*  (EfficientNetB0 U-Net, TF/Keras)
        │  best checkpoint (.tf SavedModel dir)  +  val_sparse_categorical_accuracy
        ▼
[run_inference]      tf.keras.models.load_model  (chip-by-chip predict)
        │  prediction/output/*.pred.tif
        ▼
[run_postprocessing] ramp.utils.mask_to_vec_utils  (GDAL Polygonize)
        │
        ▼
prediction/vectors/*.geojson  (per-chip building footprints)
```

## Running locally (outside ZenML)

```python
from models.ramp.pipeline import training_pipeline, inference_pipeline

# Full training run (use your dataset path)
# Hyperparameters are loaded from models/ramp/stac-item.json
training_pipeline(
    input_path="data/sample/ramp_work/input",  # or your dataset/input
    output_path="data/sample/ramp_work",       # or your dataset
)
# Use a different STAC item (e.g. versioned layout):
# training_pipeline(..., stac_item_path="models/ramp/1/stac-item.json")

# Inference run (model_uri from STAC or local path)
inference_pipeline(
    model_uri="data/sample/ramp_work/preprocessed_test/checkpoints/<timestamp>/smoke_<ts>.tf",
    input_path="data/sample/ramp_work/preprocessed_test/chips",
    prediction_path="data/sample/ramp_work/prediction_test/output",
    output_dir="data/sample/ramp_work/prediction_test/vectors",
)
# model_uri can also be: Google Drive folder URL, HTTP URL to .zip
```

## Building the Docker image

```bash
# GPU image (default)
docker build -t ramp-v1:gpu \
    --build-arg BUILD_TYPE=gpu \
    -f models/ramp/Dockerfile .

# CPU-only (development / CI)
docker build -t ramp-v1:cpu \
    --build-arg BUILD_TYPE=cpu \
    -f models/ramp/Dockerfile .
```

## Running the smoke tests

```powershell
# PowerShell (Windows)
.\models\ramp\tests\run_docker_tests.ps1 -BuildImage
```

```bash
# Bash (Linux / macOS)
BUILD_IMAGE=1 ./models/ramp/tests/run_docker_tests.sh
```

> **Note**: The smoke tests use `data/sample` (train/oam OAM tiles + train/osm labels).
> Run from the fAIr-models repo root so `/workspace/data/sample` is available in the container.

## Model weights (STAC mlm:model asset)

The STAC Item's `assets.model.href` points to pretrained weights. Supported sources:

| Source | Example |
| --- | --- |
| Local path | `/workspace/checkpoints/model.tf` |
| Google Drive folder | `https://drive.google.com/drive/folders/FOLDER_ID` |
| HTTP .zip | `https://example.com/ramp_model.zip` |

**Google Drive**: Upload the **full** Keras SavedModel directory (saved_model.pb + variables/ + assets/). The pipeline downloads via gdown and caches to `/workspace/.ramp_model_cache/`.

## Registering in the STAC catalog

```python
from fair.stac.catalog_manager import CatalogManager

cm = CatalogManager()
cm.register_model("models/ramp/stac-item.json")
```

## Dependencies from hot_fair_utilities

Only the **preprocessing** path of `hot-fair-utilities` is used in this pack:

| Used | Not used |
| --- | --- |
| `hot_fair_utilities.preprocess` (+ `multimasks_from_polygons`) | `hot_fair_utilities.predict` (YOLO / ultralytics) |
| `ramp.utils.multimask_utils` via transitive import | `hot_fair_utilities.polygonize` (AutoBFE, YOLO output) |

The `ultralytics` and `torch` packages are installed transitively by
`hot-fair-utilities` but are never imported at runtime for RAMP.

## Key design decisions

**Why ramp-fair and not inline model code?**
`ramp-fair` provides the complete EfficientNetB0 U-Net + training utilities.
This pack is thin: it declares *how* to run the model (pipeline.py) and
*what* it is (stac-item.json). Upgrading means bumping the `ramp-fair` pin.

**Why solaris from GitHub source?**
Solaris is a geospatial ML toolkit vendored inside `ramp-code-fair`. It is
not on PyPI. The Dockerfile installs it directly from the
`hotosm/ramp-code-fair` GitHub repository's `solaris/` subdirectory.

**Why separate val split in pipeline.py?**
RAMP's `data_generator` requires explicitly separate `train_img_dir` and
`val_img_dir`. The pipeline handles this automatically by shuffling a
configurable fraction (`val_fraction`, default 15%) of chips out of the
training set after preprocessing.

**Why one Dockerfile per model?**
TF 2.9.3 required for RAMP is incompatible with the TF 2.13 needed by the
YOLO packs' base image. Per-model images prevent version conflicts.
