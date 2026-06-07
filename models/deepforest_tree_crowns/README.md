# Detect small tree crowns

Object detection model for individual tree crowns in RGB aerial imagery, emitting one bounding-box polygon per detected crown.

## Architecture

- **Model**: RetinaNet with a ResNet50 backbone (DeepForest)
- **Pretrained**: NEON airborne forest dataset
- **Task**: Single-class (tree) object detection on 256x256 chips

## Dataset

Finetuned on OAM imagery chips paired with tree labels. Label polygons from the GeoJSON labels are clipped to each chip and converted to DeepForest annotation boxes (`image_path,xmin,ymin,xmax,ymax,label`). A seeded random shuffle holds out a fraction of the labeled chips for validation.

## Training

```bash
uv run python examples/detection/run.py
```

## Inference

Predictions are returned as a GeoJSON FeatureCollection of points (one per detected crown), each carrying a confidence score. The portable ONNX model takes a single RGB chip tensor and returns boxes, scores, and labels; the recipe converts each box centroid to a geographic point.

## Metrics

- Precision (box precision on the validation split)
- Recall (box recall on the validation split)
