# YOLOv11n Building Detection

Object detection model for building bounding box extraction from aerial imagery.

## Architecture

- **Model**: YOLOv11 nano (ultralytics)
- **Pretrained**: COCO dataset
- **Task**: Single-class (building) object detection

## Dataset

Derived from the same Banepa OAM+OSM dataset used for segmentation. Building polygons from the GeoJSON labels are converted to COCO format bounding boxes per chip. Pre-generated labels are stored in `data/sample/train/detection_labels.json`.

## Training

```bash
uv run python examples/detection/run.py
```

## Metrics

- mAP50 (Mean Average Precision at IoU 0.50)
- mAP50-95 (Mean Average Precision at IoU 0.50:0.95)
- Precision
- Recall
