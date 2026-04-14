# Building Detection Example

End-to-end finetuning of `yolo11n-detection` on [Banepa Municipality, Nepal](https://www.openstreetmap.org/relation/6285773)
OAM imagery with COCO-format building detection labels derived from OSM segmentation data.

## Prerequisites

- zenml
- Sample data in `data/sample/` (OAM tiles + OSM labels, including pre-generated `detection_labels.json`)

## Quick Start

```bash
uv sync --group example --group local
just setup
uv run python examples/detection/run.py
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
FAIR_FORCE_CPU=1 uv run python examples/detection/run.py
```

## Notes

- YOLO uses [ultralytics](https://docs.ultralytics.com/) for training and ONNX export.
- The `train_model` step returns a file path (not a model object) because YOLO
  objects are not pickle-serializable. ZenML materializes the `.pt` file into
  the artifact store automatically.
- ONNX export uses `model.export(format="onnx")` via ultralytics, not `torch.onnx.export`.

## Output

| Artifact | Location |
|---|---|
| STAC items | `stac_catalog/` |
| Trained artifacts | `artifacts/` |
| Predictions | `data/sample/predict/predictions/*.geojson` |
