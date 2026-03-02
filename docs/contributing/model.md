# Contributing a Model

Guide for contributing a base model to fAIr. A base model is a reusable ML
blueprint that users can finetune on their own datasets through the fAIr
platform. See [`models/example_unet/`](https://github.com/hotosm/fAIr-models/tree/master/models/example_unet) for a complete
reference implementation.

## Model Scope

fAIr targets feature extraction from **very high resolution (VHR) aerial and
satellite imagery** ; typically ~ > 30 cm ground sample distance (GSD), RGB only.
All imagery is sourced from [OpenAerialMap](https://openaerialmap.org/).

### Supported Tasks

| Task | STAC value | Label mapping | Typical output |
| --- | --- | --- | --- |
| Semantic segmentation | `semantic-segmentation` | `segmentation` | polygons |
| Instance segmentation | `instance-segmentation` | `segmentation` | polygons |
| Object detection | `object-detection` | `detection` | boxes or polygons |
| Classification | `classification` | `classification` | existing geometries with attributes |

Your `mlm:tasks` must use one or more of these exact values. CI rejects
anything else.

### Supported Feature Categories

fAIr is a humanitarian mapping platform. Models should prioritise features
that support disaster response, infrastructure mapping, and environmental
monitoring. Core categories:

| Keyword | Examples |
| --- | --- |
| `building` | Residential, commercial, industrial footprints; damaged vs. undamaged assessment |
| `road` | Highway classification (primary, secondary, tertiary); paved vs. unpaved surface detection |
| `tree` | Individual canopy, tree cover areas |
| `water` | Rivers, lakes, ponds, reservoirs |

Other OpenStreetMap feature categories (`landuse`, `bridge`, etc.) are
welcome as long as they are compatible with the platform's RGB input and
vector output constraints. To add a new keyword, include it in
[`keywords.json`](https://github.com/hotosm/fAIr-models/blob/master/fair/schemas/keywords.json) as part of your PR.

### Input Requirements

All models receive **3-band RGB GeoTIFF chips** as input. The expected tensor
layout for the `mlm:input` specification:

| Field | Value |
| --- | --- |
| Bands | `red`, `green`, `blue` (3 channels, RGB) |
| Shape | `[-1, 3, H, W]` where H and W are the chip size |
| Dimension order | `["batch", "bands", "height", "width"]` |
| Data type | `float32` |

Models must normalize the uint8 pixel values (0-255) to float32 (0.0-1.0)
in their `preprocess` function. The platform does **not** accept non-RGB
inputs (e.g. multispectral, SAR, DEM).

### Output Requirements

fAIr only supports **vector output**. Your model's final output must produce
GeoJSON geometries of one of these types:

| Geometry type | Keyword | Typical task |
| --- | --- | --- |
| `Polygon` | `polygon` | Building footprints, land parcels |
| `LineString` | `line` | Roads, waterways |
| `Point` | `point` | Tree detection, POI extraction |

Your `stac-item.json` must declare exactly which geometry type the model
produces via the `keywords` array. CI enforces that at least one of `polygon`,
`line`, or `point` is present.

Raster-only output (e.g. raw segmentation masks without vectorization) is
acceptable as an intermediate step, but the `post_processing_function` must
ultimately convert to one of the supported geometry types for downstream
consumption.

### Sample Data Layout

The sample data in `data/sample/` demonstrates the expected layout:

```text
data/sample/
  train/
    oam/             # RGB GeoTIFF chips (OAM-{x}-{y}-{z}.tif, ≥30cm GSD)
    osm/             # GeoJSON labels (osm_features_*.geojson)
  predict/
    oam/             # Input chips for inference
    predictions/     # Output directory (model writes here)
```

Chip filenames follow the pattern `OAM-{x}-{y}-{z}.tif` where x, y, z are
tile coordinates. Your model must accept these as input during both training
and inference.

## Prerequisites

Before starting, ensure you have:

- A working ML model for geospatial feature extraction (buildings, roads, trees, etc.)
- Pretrained weights that are publicly downloadable or distributable
- Familiarity with Docker and Python packaging

## License

Your model **must** use one of these open-source licenses:

| License | SPDX identifier |
| --- | --- |
| GNU GPL v3 | `GPL-3.0-only` |
| MIT | `MIT` |
| Apache 2.0 | `Apache-2.0` |
| BSD 3-Clause | `BSD-3-Clause` |

The license is declared in your `stac-item.json` under `properties.license`.
CI rejects any other license value.

## Directory Structure

Create a subdirectory under `models/` named after your model (lowercase,
hyphens for spaces):

```text
models/your-model/
  pipeline.py          # ZenML pipeline with training + inference
  Dockerfile           # Self-contained runtime environment
  stac-item.json       # STAC MLM item (model metadata)
```

## pipeline.py

This is the core of your contribution. It must export two `@pipeline`-decorated
functions that the platform discovers and dispatches automatically.

### Required Exports

```python
from zenml import pipeline

@pipeline
def training_pipeline(...) -> None:
    """Finetune the model on a dataset."""
    ...

@pipeline
def inference_pipeline(...) -> None:
    """Run prediction on input imagery."""
    ...
```

CI validates these exports via AST parsing (`scripts/validate_model.py`) --
no runtime dependencies are needed for the check to pass.

### Required Functions

Your `pipeline.py` must also define:

| Function | Role | Referenced by |
| --- | --- | --- |
| `preprocess` | Normalize/transform input data before the model | `stac-item.json` `mlm:input[].pre_processing_function` |
| `postprocess` | Convert raw model output to usable predictions | `stac-item.json` `mlm:output[].post_processing_function` |

These are referenced as Python entrypoints in the STAC item (e.g.
`models.your_model.pipeline:preprocess`). The platform calls them dynamically
-- your model owns its own pre/post processing logic entirely.

### Training Pipeline

The `training_pipeline` receives its parameters from a generated YAML config
(STAC `mlm:hyperparameters` merged with user overrides via
`fair.zenml.config.generate_training_config`). Typical parameters:

- `dataset_chips` / `dataset_labels` -- S3 or local paths to training data
- `base_model_weights` -- pretrained weight reference (URL, enum, local path)
- `epochs`, `batch_size`, `learning_rate`, `weight_decay` -- training hyperparameters
- `chip_size`, `num_classes` -- model-specific configuration

All hyperparameters must have validation constraints in the function
signature (see [Hyperparameters](#hyperparameters)). The platform rejects
invalid values at submission time, before any pod is scheduled.

Use `mlflow.log_params()` and `mlflow.log_metrics()` for experiment tracking.
Use `zenml.log_metadata()` to attach metrics to the ZenML model version.

### Inference Pipeline

The `inference_pipeline` loads weights and runs prediction. It must support
both base model weights (pretrained) and finetuned weights (from ZenML
artifact store). Use `fair.zenml.steps.load_model` to load finetuned weights:

```python
from fair.zenml.steps import load_model

@pipeline
def inference_pipeline(
    model_uri: str,
    input_images: str,
    chip_size: int,
    num_classes: int,
    zenml_artifact_version_id: str = "",
    use_base_model: bool = False,
) -> None:
    if use_base_model:
        model = load_base_model(model_uri=model_uri, num_classes=num_classes)
    else:
        model = load_model(model_uri=model_uri, zenml_artifact_version_id=zenml_artifact_version_id)
    run_inference(model=model, input_images=input_images, chip_size=chip_size, num_classes=num_classes)
```

### Data Resolution

Training data lives in S3 (production) or local filesystem (dev). Use the
helpers from `fair.utils.data` to handle both transparently:

```python
from fair.utils.data import resolve_directory, resolve_path

local_chips = str(resolve_directory(chips_path, "OAM-*.tif"))
local_labels = resolve_path(labels_path)
```

Never hardcode paths. Never bake data into Docker images.

## Dockerfile

Your Dockerfile must be **self-contained**: building and running the image
alone should be sufficient to execute both training and inference pipelines.
No external dependencies beyond what is installed in the image.

Requirements:

1. Multi-stage build recommended (builder + slim runtime)
2. Install all Python dependencies including `fair-py-ops` and your ML framework
3. Copy your model code into the image at `models/your_model/`
4. Set `ENTRYPOINT ["/usr/local/bin/python"]`

Reference Dockerfile structure:

```dockerfile
FROM ghcr.io/astral-sh/uv:python3.13-trixie-slim AS builder
ENV UV_SYSTEM_PYTHON=1 UV_LINK_MODE=copy

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgdal-dev && rm -rf /var/lib/apt/lists/*

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install \
    your-ml-framework \
    fair-py-ops 

FROM python:3.13-slim-trixie
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libexpat1 libgdal36 && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY models/your_model models/your_model
ENTRYPOINT ["/usr/local/bin/python"]
```

The image is built from the repository root (not from `models/your_model/`),
so paths in `COPY` are relative to the repo root.

## stac-item.json

The STAC item is your model's metadata card. It follows the
[STAC MLM Extension v1.5.1](https://github.com/stac-extensions/mlm) and is
validated by CI against the platform's requirements schema.

### Required Extensions

```json
"stac_extensions": [
    "https://stac-extensions.github.io/mlm/v1.5.1/schema.json",
    "https://stac-extensions.github.io/version/v1.2.0/schema.json",
    "https://stac-extensions.github.io/classification/v2.0.0/schema.json",
    "https://stac-extensions.github.io/file/v2.1.0/schema.json",
    "https://stac-extensions.github.io/raster/v1.1.0/schema.json"
]
```

### Required Properties

| Property | Type | Description |
| --- | --- | --- |
| `mlm:name` | string | Model identifier (matches directory name) |
| `mlm:architecture` | string | Architecture name (e.g. `UNet`, `YOLOv8`) |
| `mlm:tasks` | string[] | One or more of: `semantic-segmentation`, `instance-segmentation`, `object-detection`, `classification` |
| `mlm:framework` | string | `PyTorch` or `TensorFlow` |
| `mlm:framework_version` | string | Framework version |
| `mlm:pretrained` | boolean | Whether pretrained weights are used |
| `mlm:pretrained_source` | string | Origin of pretrained weights (paper, dataset) |
| `mlm:input` | object[] | Input specification with `pre_processing_function` |
| `mlm:output` | object[] | Output specification with `post_processing_function` and `classification:classes` |
| `mlm:hyperparameters` | object | Default training hyperparameters |
| `keywords` | string[] | Feature tags + task + output geometry type |
| `version` | string | Semantic version (start with `"1"`) |
| `license` | string | SPDX license identifier |

### Keywords

The `keywords` array must include:

1. **At least one feature keyword**: `building`, `road`, `tree`, `water`, `landuse`
2. **At least one task keyword**: matches `mlm:tasks` values
3. **Exactly one geometry type**: `polygon`, `line`, or `point`

Example: `["building", "semantic-segmentation", "polygon"]`

### Hyperparameters

The `mlm:hyperparameters` object in your STAC item declares the **default
training configuration**. When users finetune your model, the platform reads
these defaults and merges any user overrides into a generated YAML config
(via `fair.zenml.config.generate_training_config`). This YAML is then passed
to your `training_pipeline`.

Every key in `mlm:hyperparameters` becomes a pipeline parameter. Your
`training_pipeline` signature **must** accept all of them and apply
validation constraints using `typing.Annotated` and `typing.Literal`:

```python
from typing import Annotated, Literal
from annotated_types import Ge, Le

@pipeline
def training_pipeline(
    # ...dataset and model params...
    epochs: Annotated[int, Ge(1), Le(1000)],
    batch_size: Annotated[int, Ge(1), Le(64)],
    learning_rate: Annotated[float, Ge(1e-6), Le(1.0)],
    weight_decay: Annotated[float, Ge(0.0), Le(1.0)],
    chip_size: Annotated[int, Ge(64), Le(2048)],
    num_classes: Annotated[int, Ge(2), Le(256)],
    optimizer: Literal["Adam", "AdamW", "SGD"] = "AdamW",
    loss: Literal["CrossEntropyLoss", "BCEWithLogitsLoss"] = "CrossEntropyLoss",
) -> None:
    ...
```

This serves two purposes:

1. **ZenML validates inputs** at submission time — invalid overrides are
   rejected before any pod is scheduled
2. **STAC item documents the contract** — users and the platform know
   exactly what hyperparameters your model accepts and their valid ranges

Example `mlm:hyperparameters`:

```json
"mlm:hyperparameters": {
    "epochs": 15,
    "batch_size": 4,
    "learning_rate": 0.0001,
    "weight_decay": 0.0001,
    "chip_size": 512,
    "optimizer": "AdamW",
    "loss": "CrossEntropyLoss"
}
```

The platform auto-extracts `chip_size` from `mlm:input[0].input.shape[-1]`
and `num_classes` from `classification:classes` length, so those don't need
to be duplicated in `mlm:hyperparameters` unless your defaults differ.

### Input Specification

Each entry in `mlm:input` must declare exactly 3 RGB bands and include a
`pre_processing_function` with `format` and `expression` fields:

```json
"mlm:input": [{
    "name": "RGB chips",
    "bands": [{"name": "red"}, {"name": "green"}, {"name": "blue"}],
    "input": {
        "shape": [-1, 3, 512, 512],
        "dim_order": ["batch", "bands", "height", "width"],
        "data_type": "float32"
    },
    "pre_processing_function": {
        "format": "python",
        "expression": "models.your_model.pipeline:preprocess"
    }
}]
```

### Output Specification

Each entry in `mlm:output` must include `post_processing_function` and
`classification:classes`:

```json
"mlm:output": [{
    "name": "segmentation mask",
    "tasks": ["semantic-segmentation"],
    "result": {
        "shape": [-1, 2, 512, 512],
        "dim_order": ["batch", "channel", "height", "width"],
        "data_type": "float32"
    },
    "classification:classes": [
        {"name": "background", "value": 0},
        {"name": "building", "value": 1}
    ],
    "post_processing_function": {
        "format": "python",
        "expression": "models.your_model.pipeline:postprocess"
    }
}]
```

### Required Assets

| Asset key | Purpose | Required fields |
| --- | --- | --- |
| `model` | Pretrained weights | `mlm:artifact_type` (e.g. `torch.save`, `onnx`) |
| `source-code` | Link to model code | `mlm:entrypoint` (e.g. `models.your_model.pipeline:training_pipeline`) |
| `mlm:training` | Training Docker image | `href` = Docker image reference |
| `mlm:inference` | Inference Docker image | `href` = Docker image reference |

Model weights must be downloadable from the `model` asset `href`. This can be
a direct URL, S3 path, or a framework-specific weight enum (e.g.
`torchgeo.models.Unet_Weights.OAM_RGB_RESNET50_TCD`). Your `pipeline.py` is
responsible for resolving and loading the weights at runtime.

## PR Checklist

Before submitting your pull request:

- [ ] Directory created at `models/your-model/` with `pipeline.py`, `Dockerfile`, `stac-item.json`
- [ ] `pipeline.py` exports `training_pipeline` and `inference_pipeline` as `@pipeline`-decorated functions
- [ ] `pipeline.py` defines `preprocess` and `postprocess` functions matching STAC entrypoints
- [ ] Pipeline parameters use `Annotated` bounds and `Literal` for constrained choices
- [ ] `mlm:hyperparameters` in STAC item matches pipeline parameter names and defaults
- [ ] `mlm:input` declares exactly 3 RGB bands with `float32` data type
- [ ] Dockerfile builds successfully and is self-contained
- [ ] `stac-item.json` passes `make validate-stac`
- [ ] Model passes `make validate-models`
- [ ] License is one of: `GPL-3.0-only`, `MIT`, `Apache-2.0`, `BSD-3-Clause`
- [ ] Keywords include a feature category, task, and geometry type (`polygon`, `line`, or `point`)
- [ ] Model weights are publicly accessible or included in the weight loading code
- [ ] Model can run training on sample data in `data/sample/train/`
- [ ] Model can run inference on sample data in `data/sample/predict/`

## CI Checks

On PR submission, CI will:

1. **Validate pipeline exports** -- `scripts/validate_model.py` checks for `training_pipeline` and `inference_pipeline` via AST parsing
2. **Validate STAC item** -- `scripts/validate_stac_items.py` checks all required properties, extensions, assets, keywords (including geometry type), and license
3. **Build Docker image** -- verifies the Dockerfile builds successfully
4. **Run tests with sample data** -- executes against `data/sample/`

All checks must pass before the PR is reviewed.

## Local Development

```bash
# Install dependencies
uv sync --group local --group example

# Initialize ZenML
make init

# Validate your model
make validate-models
make validate-stac

# Run tests
make test

# Run the full example pipeline to see the expected workflow
python examples/unet/run.py all
```

## Reference

- [STAC MLM Extension](https://github.com/stac-extensions/mlm)
- [MLM Best Practices](https://github.com/stac-extensions/mlm/blob/main/best-practices.md)
- [Example UNet model](https://github.com/hotosm/fAIr-models/tree/master/models/example_unet) -- complete reference implementation
- [Example UNet STAC item](https://github.com/hotosm/fAIr-models/blob/master/models/example_unet/stac-item.json) -- valid STAC item template
