# UNet Building Segmentation

Semantic segmentation model for building footprint extraction from aerial imagery (OAM).

## Architecture

- **Model**: UNet (ResNet34 encoder)
- **Framework**: PyTorch (torchgeo)
- **Task**: Semantic segmentation
- **Input**: RGB chips (256x256, float32)
- **Classes**: 2 (background, building)

## Pretrained Source

OAM-TCD via torchgeo `Unet_Weights.OAM_RGB_RESNET34_TCD` (NeurIPS 2024, [arxiv.org/abs/2407.11743](https://arxiv.org/abs/2407.11743))

## Pipeline

Training pipeline steps (ZenML) defined in `pipeline.py`:

- `split_dataset` - spatial split via torchgeo samplers
- `train_model` - fine-tunes the UNet on labeled chip/mask pairs
- `evaluate_model` - computes accuracy, mean IoU and per-class IoU
- `export_onnx` - exports trained weights to ONNX

Inference pipeline runs `segment` over input imagery and produces GeoJSON polygons.

## Usage

See [examples/segmentation](../../examples/segmentation/) for a full end-to-end workflow.
