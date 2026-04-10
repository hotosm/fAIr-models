---
icon: lucide/box
---

# Contributing a Model

Guide for contributing a base model to fAIr. A base model is a reusable ML
blueprint that users can finetune on their own datasets through the fAIr
platform.

### Reference Implementations

| Model | Task | Architecture | Directory |
|---|---|---|---|
| UNet segmentation | Semantic segmentation | UNet (torchgeo) | [`models/unet_segmentation/`](https://github.com/hotosm/fAIr-models/tree/master/models/unet_segmentation) |
| ResNet18 classification | Binary classification | ResNet18 (torchvision) | [`models/resnet18_classification/`](https://github.com/hotosm/fAIr-models/tree/master/models/resnet18_classification) |
| YOLOv11n detection | Object detection | YOLOv11 nano (ultralytics) | [`models/yolo11n_detection/`](https://github.com/hotosm/fAIr-models/tree/master/models/yolo11n_detection) |

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

!!! danger "RGB only"

    All models receive **3-band RGB GeoTIFF chips** as input. The platform does
    **not** accept non-RGB inputs (e.g. multispectral, SAR, DEM).

| Field | Value |
| --- | --- |
| Bands | `red`, `green`, `blue` (3 channels, RGB) |
| Shape | `[-1, 3, H, W]` where H and W are the chip size |
| Dimension order | `["batch", "bands", "height", "width"]` |

Models must normalize the uint8 pixel values (0-255) in
their `preprocess` function.

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

```text title="data/sample/"
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

!!! danger "Required: Open-source license"

    Your model **must** use one of these open-source licenses:

    | License | SPDX identifier |
    | --- | --- |
    | GNU AGPL v3 | `AGPL-3.0-only` |
    | MIT | `MIT` |
    | Apache 2.0 | `Apache-2.0` |
    | BSD 3-Clause | `BSD-3-Clause` |

    The license is declared in your `stac-item.json` under `properties.license`.
    CI rejects any other license value.

## Directory Structure

Create a subdirectory under `models/` named after your model (lowercase,
hyphens for spaces):

```text title="Model directory structure"
models/your-model/
  pipeline.py          # ZenML pipeline with training + inference
  Dockerfile           # Self-contained runtime environment
  stac-item.json       # STAC MLM item (model metadata)
  README.md            # Model overview, limitations, citation
  tests/
    conftest.py        # generate_toy_dataset fixture
    test_steps.py      # Step-level tests
```

## pipeline.py

This is the core of your contribution. It must export two `@pipeline`-decorated
functions that the platform discovers and dispatches automatically.

### Required Exports

```python title="pipeline.py"
from zenml import pipeline, step

@pipeline
def training_pipeline(...) -> None:
    """Finetune the model on a dataset."""
    ...

@pipeline
def inference_pipeline(...) -> None:
    """Run prediction on input imagery."""
    ...

@step
def split_dataset(...) -> Annotated[dict[str, Any], "split_info"]:
    """Split data into train/val sets and log split metadata."""
    ...
```

CI validates these exports via AST parsing (`scripts/validate_model.py`).
Both `@pipeline` functions and the `@step split_dataset` function are
required. No runtime dependencies are needed for the check to pass.

### Required Functions

Your `pipeline.py` must also define:

| Function | Role | Referenced by |
| --- | --- | --- |
| `preprocess` | Normalize/transform input data before the model | `stac-item.json` `mlm:input[].pre_processing_function` |
| `postprocess` | Convert raw model output to usable predictions | `stac-item.json` `mlm:output[].post_processing_function` |
| `resolve_weights` | Download pretrained weights to a local checkpoint file | Required when model asset href is not a URL |

These are referenced as Python entrypoints in the STAC item (e.g.
`models.your_model.pipeline:preprocess`). The platform calls them dynamically
; your model owns its own pre/post processing logic entirely.

### resolve_weights

If your model's `stac-item.json` `model` asset `href` is **not** a direct URL
(e.g. a framework weight enum like `torchvision.models.ResNet18_Weights.IMAGENET1K_V1`
or a short name like `yolo11n.pt`), your `pipeline.py` **must** define a
`resolve_weights` function. CI enforces this via AST parsing.

```python title="resolve_weights contract"
def resolve_weights(weight_id: str) -> Path:
    """Download pretrained weights and return the local checkpoint path."""
    ...
```

The function receives the raw `href` string from the STAC item and must
return a `pathlib.Path` to a locally saved checkpoint file. The platform
calls this when `upload_artifacts=True` to download the weights, upload them
to S3, and update the STAC item href to the S3 URL.

If the `model` asset `href` is already a URL (`https://...`, `s3://...`),
`resolve_weights` is not required. When a URL is provided, CI validates
that it is accessible via HTTP HEAD request.

Examples from reference implementations:

```python title="torchvision (ResNet18)"
def resolve_weights(weight_id: str) -> Path:
    import torch
    from torchvision.models import ResNet18_Weights

    enum_name = weight_id.rsplit(".", 1)[-1]
    weights = ResNet18_Weights[enum_name]
    checkpoint_path = Path(tempfile.mkdtemp()) / "resnet18_pretrained.pth"
    torch.hub.download_url_to_file(weights.url, str(checkpoint_path))
    return checkpoint_path
```

```python title="ultralytics (YOLO)"
def resolve_weights(weight_id: str) -> Path:
    from ultralytics import YOLO

    checkpoint_dir = Path(tempfile.mkdtemp())
    model = YOLO(weight_id)
    checkpoint_path = checkpoint_dir / weight_id
    model.save(str(checkpoint_path))
    return checkpoint_path
```

### Training Pipeline

The `training_pipeline` must follow this step sequence:

```python title="Required pipeline shape"
@pipeline
def training_pipeline(
    base_model_weights: str,
    dataset_chips: str,
    dataset_labels: str,
    num_classes: int,
    hyperparameters: dict[str, Any],
) -> None:
    split_info = split_dataset(dataset_chips, dataset_labels, hyperparameters)
    trained_model = train_model(..., split_info=split_info)
    evaluate_model(trained_model, ..., split_info=split_info)
    export_onnx(trained_model, ...)
```

| Step | Required | Purpose |
| --- | --- | --- |
| `split_dataset` | Yes | Split data into train/val, log split metadata to ZenML |
| `train_model` | Yes | Train on train split only |
| `evaluate_model` | Yes | Evaluate on val split only |
| `export_onnx` | Yes | Export ONNX model with `onnx.checker.check_model()` validation |

The `split_info` dict returned by `split_dataset` is passed as a dependency
to both `train_model` and `evaluate_model`. This enforces that both steps
use the same split and that the split step runs first.

The pipeline receives its parameters from a generated YAML config
(STAC `mlm:hyperparameters` merged with user overrides via
`fair.zenml.config.generate_training_config`). Typical parameters:

- `dataset_chips` / `dataset_labels` : S3 or local paths to training data
- `base_model_weights` : pretrained weight reference (URL, enum, local path)
- `epochs`, `batch_size`, `learning_rate`, `weight_decay` : training hyperparameters
- `chip_size`, `num_classes` : model-specific configuration
- `val_ratio`, `split_seed` : train/val split configuration

All hyperparameters must have validation constraints in the function
signature (see [Hyperparameters](#hyperparameters)). The platform rejects
invalid values at submission time, before any pod is scheduled.

Use `mlflow.log_params()` and `mlflow.log_metrics()` for experiment tracking.
Use `zenml.log_metadata()` to attach metrics to the ZenML model version.

### Auto-injected Parameters

The platform automatically injects several parameters into your
`training_pipeline` from the STAC items. Your function signature **must**
accept them, but you do **not** declare them in `mlm:hyperparameters`:

| Parameter | Source | Description |
|---|---|---|
| `model_name` | User input | ZenML model name for the finetuned model |
| `base_model_id` | Base model STAC item ID | Identifies which base model is being finetuned |
| `dataset_id` | Dataset STAC item ID | Identifies which dataset is used |
| `num_classes` | `len(classification:classes)` | Extracted from STAC output spec |
| `class_names` | `classification:classes[].name` | Class name list from STAC output spec |
| `chip_size` | `mlm:input[0].input.shape[-1]` | Chip dimension from STAC input spec |
| `dataset_chips` | Dataset `chips` asset href | Path to training images |
| `dataset_labels` | Dataset `labels` asset href | Path to training labels |

### Train/Val Split (split_dataset step)

Every training pipeline **must** include a `split_dataset` step. This step
is the single source of truth for how data is divided into training and
validation sets. CI enforces its presence via AST parsing.

Your `split_dataset` step must:

1. Accept `dataset_chips`, `dataset_labels`, and `hyperparameters`
2. Read `val_ratio` and `split_seed` from hyperparameters
3. Perform the split (strategy is model-specific)
4. Log split metadata to ZenML via `log_metadata(metadata={"fair/split": split_info})`
5. Return a `split_info` dict

The `split_info` dict must contain:

| Key | Type | Description |
| --- | --- | --- |
| `strategy` | string | Split strategy: `"random"`, `"spatial"`, or custom |
| `val_ratio` | float | Actual validation ratio used |
| `seed` | int | Random seed for reproducibility |
| `train_count` | int | Number of training samples |
| `val_count` | int | Number of validation samples |
| `description` | string | Human-readable explanation of how the split works |

Example implementation:

```python title="split_dataset step"
@step
def split_dataset(
    dataset_chips: str,
    dataset_labels: str,
    hyperparameters: dict[str, Any],
) -> Annotated[dict[str, Any], "split_info"]:
    val_ratio = hyperparameters.get("val_ratio", 0.2)
    seed = hyperparameters.get("split_seed", 42)

    # Your split logic here
    train_samples, val_samples = do_split(dataset_chips, dataset_labels, val_ratio, seed)

    split_info = {
        "strategy": "random",
        "val_ratio": val_ratio,
        "seed": seed,
        "train_count": len(train_samples),
        "val_count": len(val_samples),
        "description": "Random split by seeded shuffle of sorted filenames",
    }
    log_metadata(metadata={"fair/split": split_info})
    return split_info
```

The split metadata flows through the promotion pipeline into the local model
STAC item as `fair:split`, giving users full visibility into how each
finetuned model was trained.

!!! warning "Train on train, evaluate on val"

    `train_model` must only see training data. `evaluate_model` must only see
    validation data. Both steps receive `split_info` and must use it to
    reconstruct the same split deterministically. Evaluating on training data
    produces inflated metrics that do not reflect real-world performance.

### Non-serializable Model Pattern (YOLO)

Some ML frameworks produce model objects that are not pickle-serializable
(e.g. ultralytics YOLO). In these cases, your `train_model` step should
**return the file path to the saved checkpoint** instead of the model object
itself. ZenML will materialize the `.pt` file into the artifact store.

See `models/yolo11n_detection/pipeline.py` for a working example of this
pattern.

### Inference Pipeline

The `inference_pipeline` loads weights and runs prediction. It must support
both base model weights (pretrained) and finetuned weights (from ZenML
artifact store). Use `fair.zenml.steps.load_model` to load finetuned weights:

```python title="Inference pipeline example"
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

```python title="Data resolution helpers"
from fair.utils.data import resolve_directory, resolve_path

local_chips = str(resolve_directory(chips_path, "OAM-*.tif"))
local_labels = resolve_path(labels_path)
```

!!! warning

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

```dockerfile title="Dockerfile"
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

## Testing

Each model needs a `tests/` directory with two files:

- `conftest.py` : a `generate_toy_dataset` fixture that creates toy chips + labels at test time
- `test_steps.py` : four test functions : `test_split_dataset`, `test_train_model`, `test_evaluate_model`, `test_export_onnx`

The shared `models/conftest.py` provides common fixtures (`toy_chips`,
`toy_labels`, `base_hyperparameters`, etc.) automatically. You only write
`generate_toy_dataset`.

See `models/resnet18_classification/tests/`, `models/yolo11n_detection/tests/`,
or `models/unet_segmentation/tests/` for complete working examples.

```bash
just test-models your-model   # Build Docker image + run tests
```

## stac-item.json

The STAC item is your model's metadata card. It follows the
[STAC MLM Extension v1.5.1](https://github.com/stac-extensions/mlm) and is
validated by CI against the platform's requirements schema.

??? note "Required Extensions"

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
| `title` | string | Human-readable model name (shown in catalog UI) |
| `description` | string | One-paragraph summary of the model and its intended use |
| `mlm:name` | string | Model identifier (matches directory name) |
| `mlm:architecture` | string | Architecture name (e.g. `UNet`, `YOLOv8`) |
| `mlm:tasks` | string[] | One or more of: `semantic-segmentation`, `instance-segmentation`, `object-detection`, `classification` |
| `mlm:framework` | string | `PyTorch` or `TensorFlow` |
| `mlm:framework_version` | string | Framework version |
| `mlm:pretrained` | boolean | Whether pretrained weights are used |
| `mlm:pretrained_source` | string | URL of the paper or dataset the weights come from |
| `mlm:input` | object[] | Input specification with `pre_processing_function` |
| `mlm:output` | object[] | Output specification with `post_processing_function` and `classification:classes` |
| `mlm:hyperparameters` | object | Default training hyperparameters |
| `keywords` | string[] | Feature tags + task + output geometry type |
| `version` | string | Semantic version (start with `"1"`) |
| `license` | string | SPDX license identifier |
| `fair:metrics_spec` | object[] | Evaluation metrics vocabulary (see below) |
| `fair:split_spec` | object | Train/val split specification (see below) |

### fair:metrics_spec

The MLM extension does not define evaluation metrics semantics. `fair:metrics_spec`
fills this gap by declaring the meaning and storage location of each evaluation
metric your model produces during `evaluate_model`. Users need this to understand
what `"accuracy"` means (pixel accuracy? per-class? mean IoU?).

Each entry must declare:

| Field | Type | Description |
| --- | --- | --- |
| `key` | string | Property key where the metric is stored on the local model STAC item (e.g. `fair:accuracy`) |
| `name` | string | Human-readable metric name |
| `description` | string | Precise definition including averaging strategy |

Example:

```json title="fair:metrics_spec example"
"fair:metrics_spec": [
    {
        "key": "fair:accuracy",
        "name": "Pixel Accuracy",
        "description": "Fraction of correctly classified pixels across all classes"
    },
    {
        "key": "fair:mean_iou",
        "name": "Mean IoU (macro)",
        "description": "Macro-averaged IoU across classes; each class weighted equally"
    },
    {
        "key": "fair:per_class_iou",
        "name": "Per-class IoU",
        "description": "IoU per class, stored as object keyed by class name from classification:classes"
    }
]
```

When `evaluate_model` logs metrics via `log_metadata(infer_model=True)`, the platform
copies those values to the promoted local model STAC item. Class IoU keys use the
`classification:classes` names, e.g. `iou_background`, `iou_building` (not numeric
indices like `iou_class_0`).

### fair:split_spec

The `fair:split_spec` property declares how your model expects training data
to be split into train and validation sets. This is a **required** property
on base model STAC items. CI validates its presence and structure.

| Field | Type | Description |
| --- | --- | --- |
| `strategy` | string | Split strategy: `"random"`, `"spatial"`, or custom |
| `default_ratio` | float | Recommended validation ratio (0 < ratio < 1) |
| `seed` | int | Default random seed for reproducibility |
| `description` | string | Explanation of how the split works for this model |

The split strategy depends on the task type:

| Task | Strategy | Description |
| --- | --- | --- |
| Classification | `random` | Seeded shuffle of sorted filenames, split at ratio boundary |
| Segmentation | `spatial` | `RandomGeoSampler` for train, `GridGeoSampler` for val (non-overlapping tiles) |
| Detection | `random` | Last N% of sorted image IDs held out for validation |

Example:

```json title="fair:split_spec example"
"fair:split_spec": {
    "strategy": "random",
    "default_ratio": 0.2,
    "seed": 42,
    "description": "Random split by seeded shuffle of sorted filenames. Deterministic given the same seed."
}
```

Contributors can define custom split strategies as long as they document the
approach in `description` and implement the corresponding `split_dataset`
step. The `val_ratio` and `split_seed` hyperparameters allow users to
override the defaults at finetuning time.

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

```python title="Hyperparameter validation example"
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

In addition to model-specific hyperparameters, you **must** include these
split and training parameters:

| Parameter | Type | Required | Description |
| --- | --- | --- | --- |
| `val_ratio` | float | Yes | Fraction of data held out for validation (default 0.2) |
| `split_seed` | int | Yes | Random seed for reproducible train/val split (default 42) |
| `scheduler` | string | Recommended | LR scheduler: `"cosine"` or `"none"` |
| `max_grad_norm` | float | Recommended | Maximum gradient norm for clipping (default 1.0) |

Example `mlm:hyperparameters`:

```json title="mlm:hyperparameters example"
"mlm:hyperparameters": {
    "epochs": 15,
    "batch_size": 4,
    "learning_rate": 0.0001,
    "weight_decay": 0.0001,
    "chip_size": 512,
    "optimizer": "AdamW",
    "loss": "CrossEntropyLoss",
    "val_ratio": 0.2,
    "split_seed": 42,
    "scheduler": "cosine",
    "max_grad_norm": 1.0
}
```

The platform auto-extracts `chip_size` from `mlm:input[0].input.shape[-1]`
and `num_classes` from `classification:classes` length, so those don't need
to be duplicated in `mlm:hyperparameters` unless your defaults differ.

??? note "Input Specification"

    Each entry in `mlm:input` must declare exactly 3 RGB bands and include a
    `pre_processing_function` with `format` and `expression` fields:

    ```json title="mlm:input example"
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

??? note "Output Specification"

    Each entry in `mlm:output` must include `post_processing_function` and
    `classification:classes`:

    ```json title="mlm:output example"
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
| `source-code` | Link to model source code (git URL) | `mlm:entrypoint` (e.g. `models.your_model.pipeline:training_pipeline`) |
| `mlm:training` | Training Docker image | `href` = Docker image reference |
| `mlm:inference` | Inference Docker image | `href` = Docker image reference |
| `readme` | Model documentation (README.md) | _(none)_ |

Model weights must be downloadable from the `model` asset `href`. This can be
a direct URL, S3 path, or a framework-specific weight enum (e.g.
`torchgeo.models.Unet_Weights.OAM_RGB_RESNET50_TCD`). Your `pipeline.py` is
responsible for resolving and loading the weights at runtime.

!!! info "Weight upload to S3"

    When `upload_artifacts=True`, the platform downloads the pretrained weights
    via your `resolve_weights` function, uploads them to S3, and updates the
    STAC item href to the S3 URL. If your model asset href is already a URL,
    CI validates it is accessible. If it is a framework weight reference,
    CI validates that `resolve_weights` exists in `pipeline.py`.

The `readme` asset `href` must be an **absolute URL** to the raw file, not a
relative path. Use the GitHub raw URL pattern:

```json
"readme": {
    "href": "https://raw.githubusercontent.com/hotosm/fAIr-models/refs/heads/main/models/your_model/README.md",
    "type": "text/markdown",
    "roles": ["metadata"],
    "title": "Model README"
}
```

Relative paths such as `./README.md` are not accessible from deployed STAC
catalogs and will be flagged by validation.

The `source-code` asset `href` must point to the git repository (or tree URL)
where the model's source code lives. This is validated by CI and displayed on
the model's catalog page.

### cite-as Link

If your model or its pretrained weights come from a published paper, add a
`cite-as` link pointing to the canonical DOI or arXiv URL:

```json title="cite-as link example"
{
    "rel": "cite-as",
    "href": "https://arxiv.org/abs/2407.11743",
    "type": "text/html",
    "title": "Paper title"
}
```

This link is displayed in the catalog UI. Use the canonical DOI URL when
available (`https://doi.org/...`).

## README.md

Every model **must** include a `README.md` in its directory. This is the
human-readable documentation for your model ; it covers context that the STAC
MLM item cannot express.

The README is referenced as a `readme` asset in `stac-item.json` with an
**absolute raw GitHub URL** (see [Required Assets](#required-assets) above).
Validation checks that the README file exists locally and that the asset is
present in the STAC item.

### What to include

| Section | Content |
| --- | --- |
| **Overview** | One-paragraph summary: what the model does, target geography, intended use |
| **Architecture** | Model type, backbone, input/output shapes, key design choices |
| **Pretrained source** | Training dataset, paper reference, data license |
| **Limitations** | Known failure modes, geographic bias, resolution constraints |
| **Usage** | How to run training/inference locally, example commands |
| **Citation** | BibTeX or reference if the model or weights come from published work |
| **License** | License name (must match `properties.license` in `stac-item.json`) |

Keep it concise. The STAC item already captures hyperparameters, input/output
specs, and keywords ; the README is for everything else.

## PR Checklist

Before opening a PR, make sure:

- [ ] `models/your-model/` includes `pipeline.py`, `Dockerfile`, `stac-item.json`, and `README.md`
- [ ] `tests/test_steps.py` defines `test_split_dataset`, `test_train_model`, `test_evaluate_model`, `test_export_onnx`
- [ ] `tests/conftest.py` defines `generate_toy_dataset` fixture returning `{"chips", "labels", "dataset_stac_item"}`
- [ ] `README.md` explains the model clearly enough for another developer to use it
- [ ] `just validate` passes for the model and STAC item
- [ ] `just test-models your-model` passes inside Docker

The full requirements are described in the sections above, especially the STAC metadata, pipeline structure, assets, and README guidance. CI checks the detailed metadata, pipeline exports, Docker build, and consistency rules for you.

## CI Checks

On PR submission, CI will:

1. **Validate pipeline exports** : `scripts/validate_model.py` checks for `training_pipeline` and `inference_pipeline` (`@pipeline`), `split_dataset` (`@step`), and required test functions via AST parsing
2. **Validate STAC item** : `scripts/validate_stac_items.py` checks all required properties (including `fair:split_spec`), extensions, assets, keywords (including geometry type), and license
3. **Build Docker image** : verifies the Dockerfile builds successfully
4. **Run step tests** : `python -m pytest models/<name>/tests/` inside Docker validates all 4 pipeline steps with toy data

All checks must pass before the PR is reviewed.

## Local Development

```bash title="Local dev workflow"
just setup                             # Install deps + ZenML init
just validate                          # Validate STAC items + model pipelines
just test                              # Run tests
just example                           # Run full example pipeline
```

## Reference

- [STAC MLM Extension v1.5.1](https://github.com/stac-extensions/mlm) -- MLM fields spec
- [MLM Best Practices](https://github.com/stac-extensions/mlm/blob/main/best-practices.md)
- [UNet segmentation model](https://github.com/hotosm/fAIr-models/tree/master/models/unet_segmentation) -- segmentation reference
- [ResNet18 classification model](https://github.com/hotosm/fAIr-models/tree/master/models/resnet18_classification) -- classification reference
- [YOLOv11n detection model](https://github.com/hotosm/fAIr-models/tree/master/models/yolo11n_detection) -- detection reference
- [UNet STAC item](https://github.com/hotosm/fAIr-models/blob/master/models/unet_segmentation/stac-item.json) -- STAC item template
