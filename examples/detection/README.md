# Building Detection Example

End-to-end finetuning of `yolo11n-detection` on [Banepa Municipality, Nepal](https://www.openstreetmap.org/relation/6285773)
OAM imagery with COCO-format building detection labels derived from OSM segmentation data.

## Prerequisites

- zenml
- Sample data in `data/sample/` (OAM tiles + OSM labels)
- Detection labels generated from segmentation labels (see below)

## Label Conversion

Detection labels are derived from the segmentation GeoJSON labels. Run the
conversion script before training:

```bash
python scripts/convert_segmentation_to_detection.py
```

This produces `data/sample/train/detection_labels.json` in COCO format.

## Quick Start

```bash
uv sync --group example --group local
just setup
python scripts/convert_segmentation_to_detection.py
python examples/detection/run.py all
```

## Commands

The `run.py` script provides a unified CLI for all workflow steps:

```bash
python examples/detection/run.py <command>
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
python scripts/convert_segmentation_to_detection.py
uv run python examples/detection/run.py clean
uv run python examples/detection/run.py all
```

## Notes

- YOLO uses [ultralytics](https://docs.ultralytics.com/) for training and ONNX export.
- The `train_model` step returns a file path (not a model object) because YOLO
  objects are not pickle-serializable. ZenML materializes the `.pt` file into
  the artifact store automatically.
- ONNX export uses `model.export(format="onnx")` via ultralytics, not `torch.onnx.export`.

## Output

| Command | Artifacts |
|---|---|
| `init` | `stac_catalog/` (3 collections: base-models, datasets, local-models) |
| `register` | STAC items: `base-models/yolo11n-detection`, `datasets/buildings-banepa-detection` |
| `finetune` | `artifacts/` (ZenML artifact store + trained weights) |
| `promote` | STAC item: `local-models/yolo11n-detection-finetuned-banepa-vN` |
| `predict` | `data/sample/predict/predictions/*.geojson` (detection results) |
