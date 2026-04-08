# Building Segmentation Example

End-to-end finetuning of `unet-segmentation` on [Banepa Municipality, Nepal](https://www.openstreetmap.org/relation/6285773)
OAM imagery with OSM building labels.

## Prerequisites

- zenml
- Sample data in `data/sample/` (OAM tiles + OSM labels)

## Quick Start

```bash
uv sync --group example --group local
just setup
python examples/segmentation/run.py all
```

## Commands

The `run.py` script provides a unified CLI for all workflow steps:

```bash
python examples/segmentation/run.py <command>
```

### Available Commands

- `init` - Initialize ZenML and STAC catalog
- `register` - Register base model + dataset to STAC
- `finetune` - Train model (1 epoch for CI/testing)
- `promote` - Promote latest model version to production + publish to STAC
- `predict` - Run inference with promoted model
- `all` - Execute full pipeline: init → register → finetune → promote → predict
- `clean` - Remove generated artifacts (stac_catalog, artifacts, configs, predictions)

### CI Usage

```bash
uv run python examples/segmentation/run.py clean
uv run python examples/segmentation/run.py all
```

## Output

| Command | Artifacts |
|---|---|
| `init` | `stac_catalog/` (3 collections: base-models, datasets, local-models) |
| `register` | STAC items: `base-models/unet-segmentation`, `datasets/buildings-banepa-segmentation` |
| `finetune` | `artifacts/` (ZenML artifact store + trained weights) |
| `promote` | STAC item: `local-models/unet-segmentation-finetuned-banepa-vN` |
| `predict` | `data/sample/predict/predictions/*.tif` (segmentation masks) |
