# YOLOv11m Swimming Pool Detection

Object detection model for swimming pool bounding box extraction from aerial imagery.

## Architecture

- **Model**: YOLOv11 medium (ultralytics)
- **Pretrained**: [mozilla-ai/swimming-pool-detector](https://huggingface.co/mozilla-ai/swimming-pool-detector)
- **Task**: Single-class (swimming_pool) object detection

## Inference

Served via the model-agnostic ONNX runtime in `fair.serve.base`. Predictions are returned as a GeoJSON `FeatureCollection` of axis-aligned bounding box polygons (EPSG:4326), one feature per detected pool with a `confidence` property. Recommended imagery zoom is 19 (~0.3 m/px), matching the training scale.

## Metrics

- mAP50 (Mean Average Precision at IoU 0.50)
- mAP50-95 (Mean Average Precision at IoU 0.50:0.95)
- Precision
- Recall

## License

AGPL-3.0-only, inherited from the upstream Ultralytics YOLO weights.
