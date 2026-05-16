# Building Segmentation Example

End-to-end finetuning of `unet-segmentation` on [Banepa Municipality, Nepal](https://www.openstreetmap.org/relation/6285773)
OAM imagery with OSM building labels.

## Prerequisites

- Docker, [uv](https://docs.astral.sh/uv/), [just](https://just.systems/).
- Sample data in `data/sample/` (OAM tiles + OSM labels).

## Quick Start

```bash
just setup                        # install deps + bring up the stack
just build unet_segmentation      # build the model container
just example segmentation         # run this pipeline only
```

## Workflow

The script submits the pipeline to ZenML. Each step runs inside the model's
docker image (`ghcr.io/hotosm/fair-models/unet_segmentation:latest`):

1. Register the base model item with the STAC catalog
2. Register the dataset item
3. Finetune the model
4. Promote the finetuned model
5. Run prediction on sample imagery

## Output

| Artifact | Location |
| --- | --- |
| STAC items | PgStAC API at <http://localhost:8082> |
| Trained artifacts | `s3://zenml/` in MinIO at <http://localhost:9001> |
| MLflow runs | <http://localhost:5000> |
| Predictions | `data/sample/predict/predictions/*.tif` |
