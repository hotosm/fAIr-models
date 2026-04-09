# ResNet18 Building Classification

Binary classification model for detecting building presence in aerial imagery chips.

## Architecture

- **Backbone**: ResNet18 pretrained on ImageNet (torchvision)
- **Head**: Replaced FC layer with Linear(512, 1) for binary classification
- **Transfer learning**: Backbone frozen, only classification head trained

## Task

Given an aerial imagery chip (256x256 RGB), predict whether the chip contains a building (1) or not (0).

## Dataset

Derived from the same Banepa OAM+OSM dataset used for segmentation. Building polygons from the GeoJSON labels are intersected with each chip to produce per-chip binary labels (building / no_building). Pre-generated labels are stored in `data/sample/train/classification_labels.csv`.

## Training

```bash
python examples/classification/run.py all
```

## Metrics

- Accuracy
- Precision
- Recall
- F1 Score
