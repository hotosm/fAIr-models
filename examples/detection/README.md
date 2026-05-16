# Building Detection Example

End-to-end finetuning of `yolo11n-detection` on [Banepa Municipality, Nepal](https://www.openstreetmap.org/relation/6285773)
OAM imagery with COCO-format building detection labels derived from OSM segmentation data.

## Prerequisites

- Docker, [uv](https://docs.astral.sh/uv/), [just](https://just.systems/).
- Sample data in `data/sample/` (OAM tiles + OSM building polygons; per-chip COCO bounding boxes are derived at runtime).

## Quick Start

```bash
just setup                          # install deps + bring up the stack
just build yolo11n_detection        # build the model container
just example detection              # run this pipeline only
```

## Workflow

The script submits the pipeline to ZenML. Each step runs inside the model's
docker image (`ghcr.io/hotosm/fair-models/yolo11n_detection:latest`):

1. Register the base model item with the STAC catalog
2. Register the dataset item
3. Finetune the model
4. Promote the finetuned model
5. Run prediction on sample imagery

## Notes

- YOLO uses [ultralytics](https://docs.ultralytics.com/) for training and ONNX export.
- The `train_model` step returns a file path (not a model object) because YOLO
  objects are not pickle-serializable. ZenML materializes the `.pt` file into
  the artifact store automatically.
- ONNX export uses `model.export(format="onnx")` via ultralytics, not `torch.onnx.export`.

## Output

| Artifact | Location |
| --- | --- |
| STAC items | PgStAC API at <http://localhost:8082> |
| Trained artifacts | `s3://zenml/` in MinIO at <http://localhost:9001> |
| MLflow runs | <http://localhost:5000> |
| Predictions | `data/sample/predict/predictions/*.geojson` |
