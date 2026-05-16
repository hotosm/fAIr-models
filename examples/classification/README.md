# Building Classification Example

End-to-end finetuning of `resnet18-classification` on [Banepa Municipality, Nepal](https://www.openstreetmap.org/relation/6285773)
OAM imagery with binary building/no_building labels derived from OSM segmentation data.

## Prerequisites

- Docker, [uv](https://docs.astral.sh/uv/), [just](https://just.systems/).
- Sample data in `data/sample/` (OAM tiles + OSM building polygons; per-chip binary labels are derived at runtime).

## Quick Start

```bash
just setup                              # install deps + bring up the stack
just build resnet18_classification      # build the model container
just example classification             # run this pipeline only
```

## Workflow

The script submits the pipeline to ZenML. Each step runs inside the model's
docker image (`ghcr.io/hotosm/fair-models/resnet18_classification:latest`):

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
| Predictions | `data/sample/predict/predictions/*.csv` |
