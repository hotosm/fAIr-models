# UNet Building Segmentation

Semantic segmentation model for building footprint extraction from aerial imagery (OAM).

## Architecture

- **Model**: UNet (ResNet34 encoder)
- **Framework**: PyTorch (torchgeo)
- **Task**: Semantic segmentation
- **Input**: RGB chips (256x256, float32)
- **Classes**: 2 (background, building)

## Pretrained Source

Building-segmentation base checkpoint trained on
[hotosm/vhr-building-segmentation](https://huggingface.co/datasets/hotosm/vhr-building-segmentation)
(57,890 train / 7,237 val chips).

### Base checkpoint provenance

The previous base (`unet_resnet34_oam_rgb_tcd`) is a **tree-crown delineation**
model: it scores F1 ≈ 0 on buildings zero-shot, and fine-tuning from it reached
50.0 pooled test F1 at 32 chips — below training the decoder from random
initialization (52.4; 6 regions × 5 folds). The building-pretrained base of
identical architecture reaches 59.3 pooled (+9.3) under the same recipe.

Training config of the replacement base (mirrors this pipeline's fine-tune
recipe, seed 1337):

- Architecture: `torchgeo`/`segmentation_models_pytorch` UNet, ResNet34
  encoder (ImageNet init, frozen); decoder + 2-class head from random init
- Data: `hotosm/vhr-building-segmentation`, 256×256 chips, inputs scaled `/255`
- Loss/optim: 2-class cross-entropy, AdamW lr 1e-3 (sweep-selected for
  from-scratch decoder training), weight decay 1e-4, batch 32, cosine schedule
  (T_max 50), gradient clip 1.0, fp16 autocast
- Early stopping patience 5 on validation cross-entropy; best checkpoint kept
  (val CE 0.2351)

Zero-shot on the dataset's own held-out splits (building-class pixel metrics,
argmax operating point; val n=7,237 / test n=7,236 chips):

| Base | Split | Building IoU | Building F1 |
|---|---|---|---|
| tree-crown (previous) | val / test | 0.000 / 0.000 | 0.000 / 0.000 |
| buildings (this) | val / test | 0.252 / 0.342 | 0.402 / 0.510 |

Both checkpoints load into the pipeline's model with 0 missing / 0 unexpected
keys (`scripts/eval_bases_dataset_val.py`).

## Pipeline

Training pipeline steps (ZenML) defined in `pipeline.py`:

- `split_dataset` - spatial split via torchgeo samplers
- `train_model` - fine-tunes the UNet on labeled chip/mask pairs
- `evaluate_model` - computes accuracy, mean IoU and per-class IoU
- `export_onnx` - exports trained weights to ONNX

Inference pipeline runs `segment` over input imagery and produces GeoJSON polygons.

## Usage

See [examples/segmentation](../../examples/segmentation/) for a full end-to-end workflow.
