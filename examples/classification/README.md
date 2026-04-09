# Building Classification Example

End-to-end finetuning of `resnet18-classification` on [Banepa Municipality, Nepal](https://www.openstreetmap.org/relation/6285773)
OAM imagery with binary building/no_building labels derived from OSM segmentation data.

## Prerequisites

- zenml
- Sample data in `data/sample/` (OAM tiles + OSM labels, including pre-generated `classification_labels.csv`)

## Quick Start

```bash
uv sync --group example --group local
just setup
python examples/classification/run.py all
```

## Commands

The `run.py` script provides a unified CLI for all workflow steps:

```bash
python examples/classification/run.py <command>
```

### Available Commands

- `init` - Initialize ZenML and STAC catalog
- `register` - Register base model + dataset to STAC
- `finetune` - Train model (1 epoch for CI/testing)
- `promote` - Promote latest model version to production + publish to STAC
- `predict` - Run inference with promoted model
- `all` - Execute full pipeline: init -> register -> finetune -> promote -> predict
- `clean` - Remove generated artifacts (stac_catalog, artifacts, configs, predictions)

### CI Usage

```bash
uv run python examples/classification/run.py clean
uv run python examples/classification/run.py all
```

## Output

| Command | Artifacts |
|---|---|
| `init` | `stac_catalog/` (3 collections: base-models, datasets, local-models) |
| `register` | STAC items: `base-models/resnet18-classification`, `datasets/buildings-banepa-classification` |
| `finetune` | `artifacts/` (ZenML artifact store + trained weights) |
| `promote` | STAC item: `local-models/resnet18-classification-finetuned-banepa-vN` |
| `predict` | `data/sample/predict/predictions/*.csv` (classification results) |
