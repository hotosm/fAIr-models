# Building Segmentation Example

End-to-end finetuning of `example-unet` on [Banepa Municipality, Nepal](https://www.openstreetmap.org/relation/6285773)
OAM imagery with OSM building labels.

## Prerequisites

- `uv` >= 0.4 and Python 3.12+
- Internet access for OAM/OSM download and ZenML ZenHub (or local ZenML server)

## Steps

```bash
# 1. Install dependencies
uv sync --group example --group local

# 2. Initialize ZenML (local SQLite stack)
make init

# 3. Download OAM imagery and OSM building labels (~Banepa area)
uv run python examples/building_segmentation/download.py

# 4. Register the dataset in the STAC catalog
uv run python examples/building_segmentation/register_dataset.py

# 5. Register the example-unet base model in the STAC catalog
uv run python examples/building_segmentation/register_basemodel.py

# 6. Finetune (1 epoch by default -- edit OVERRIDES in finetune.py to change)
uv run python examples/building_segmentation/finetune.py

# 7. Promote model to production and publish to STAC local-models collection
uv run python examples/building_segmentation/promote.py

# 8. Run inference on the downloaded OAM images
uv run python examples/building_segmentation/predict.py
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

## Adapting to Other Areas

Edit the constants at the top of each script:

- `download.py`: `BBOX`, `ZOOM`, `OAM_IMAGE_ID`, `CHIP_SIZE`
- `register_dataset.py`: `ITEM_ID`, `CHIPS_HREF`, `OSM_DIR`
- `finetune.py`: `BASE_MODEL_ID`, `DATASET_ID`, `MODEL_NAME`, `OVERRIDES`
- `promote.py` / `predict.py`: `MODEL_NAME` / `LOCAL_MODEL_ID`
