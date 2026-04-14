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
uv run python examples/segmentation/run.py
```

## Workflow

The script runs the full workflow in one execution:

1. Initialize ZenML and local STAC context
2. Register the base model item
3. Register the dataset item
4. Finetune the model
5. Promote the finetuned model
6. Run prediction on sample imagery

### CI Usage

```bash
FAIR_FORCE_CPU=1 uv run python examples/segmentation/run.py
```

## Output

| Artifact | Location |
|---|---|
| STAC items | `stac_catalog/` |
| Trained artifacts | `artifacts/` |
| Predictions | `data/sample/predict/predictions/*.tif` |
