# Building Segmentation Example

End-to-end finetuning of `example-unet` on [Banepa Municipality, Nepal](https://www.openstreetmap.org/relation/6285773)
OAM imagery with OSM building labels.

## Prerequisites

- `uv` >= 0.4 and Python 3.12+
- Internet access for OAM/OSM download and ZenML ZenHub (or local ZenML server)

## Steps

```bash
# install
uv sync --group example --group local

# setup
make init

# download datasets
uv run python examples/unet/download.py

# register your dataset to stac
uv run python examples/unet/register_dataset.py

uv run python examples/unet/register_basemodel.py

uv run python examples/unet/finetune.py

# promote finetuned model & publish to STAC local-models collection
uv run python examples/unet/promote.py

uv run python examples/unet/predict.py
```

## Output

| After step | Artifact |
|---|---|
| `download.py` | `data/banepa/oam/` (image tiles) + `data/banepa/osm/` (label GeoJSON) |
| `register_dataset.py` | `stac_catalog/datasets/buildings-banepa/buildings-banepa.json` |
| `register_basemodel.py` | `stac_catalog/base-models/example-unet/example-unet.json` |
| `finetune.py` | `artifacts/finetuned_weights.pth` + ZenML run logged |
| `promote.py` | `stac_catalog/local-models/example-unet-finetuned-banepa-v1/` |
| `predict.py` | `data/banepa/predictions/*.png` |
