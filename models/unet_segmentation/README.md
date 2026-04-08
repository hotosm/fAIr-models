# UNet Building Segmentation

Semantic segmentation model for building footprint extraction from aerial imagery (OAM).

## Architecture

- **Model**: UNet
- **Framework**: PyTorch (torchgeo)
- **Task**: Semantic segmentation
- **Input**: RGB chips (512×512, float32)
- **Classes**: 2 (background, building)

## Pretrained Source

OAM-TCD (NeurIPS 2024, [arxiv.org/abs/2407.11743](https://arxiv.org/abs/2407.11743))

## Pipeline

The training and evaluation pipeline is defined in `pipeline.py` with ZenML steps:

- `train_model` — fine-tunes the UNet on labeled chip/mask pairs
- `evaluate_model` — computes accuracy and IoU metrics
- `predict` — runs inference on OAM tiles and produces GeoJSON polygons

## Usage

See [examples/segmentation](../../examples/segmentation/) for a full end-to-end workflow.
