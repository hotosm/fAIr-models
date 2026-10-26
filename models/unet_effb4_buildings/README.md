# unet-effb4-buildings

Building footprint instance segmentation for HOT's fAIr. A U-Net with an
EfficientNet-B4 encoder predicts four channels per pixel; buildings are
separated by a watershed seeded from the centreness channel.

Weights and config: [nilsho01/unet-buildings-autorun-v2](https://huggingface.co/nilsho01/unet-buildings-autorun-v2)

## Lineage

This model builds on
[nilsho01/unet-effb4-dist3-buildings](https://huggingface.co/nilsho01/unet-effb4-dist3-buildings),
whose model card compares it with the shipped `dinov3s_upernet_hot` on the
reviewed benchmark (PQ 47.79 ± 0.23 over five folds against 33.92), together
with the label-quality findings behind its training pool. This model keeps that
architecture and data basis, adds a centreness channel, trains longer and adds
Open Cities chips. It improves on the `dist3` model by 1.8 PQ on the same
benchmark (table below).

## Summary

| | |
|---|---|
| Task | Building footprint instance segmentation |
| Input | 3-band RGB GeoTIFF chips, VHR (~30 cm GSD), 256 × 256 |
| Output | GeoJSON `FeatureCollection` of `Polygon` features in EPSG:4326 |
| Training pool | 19,199 chips from 21 countries (4,379 with buildings, 96,848 building instances) |
| License | Apache-2.0 (code and weights); data attribution below |

## Results

Panoptic quality (PQ = SQ × RQ) on chips that contain buildings. Ground-truth
instances are connected components of the label mask, instances under 30 px are
ignored, predictions are averaged over the 8 flips/transposes (d4 TTA), and the
decode settings were selected on validation. Nothing was tuned on the benchmark.

| | val PQ | val SQ | val RQ | bench PQ | bench SQ | bench RQ |
|---|---|---|---|---|---|---|
| previous version of this model (`dist3` head) | 56.06 | 79.03 | 70.93 | 49.30 | 76.28 | 64.63 |
| **this model** | **57.30** | **79.53** | **72.06** | **51.08** | **76.69** | **66.60** |

Validation is a spatially blocked cross-validation fold (2,651 chips, 1,532 with
buildings). The benchmark is
[nilsho01/vhr-buildings-benchmark-v1](https://huggingface.co/datasets/nilsho01/vhr-buildings-benchmark-v1)
(400 chips with reviewed labels). Both rows are single training runs (one seed).
A from-scratch rerun of the `dist3` recipe reproduced its score to the second
decimal, which suggests run-to-run noise below the 1.2–1.8 PQ gap shown here,
but that was not measured for this model.

fAIr runs inference **without** TTA by default because TTA costs 8 forward
passes per chip. On the predecessor of this model, dropping TTA cost about
1 PQ. Set `tta` to `true` to reproduce the table.

### Breakdown (PQ, this model)

| stratum | val | bench |
|---|---|---|
| size 0–8 px | 12.64 | 11.58 |
| size 8–12 px | 33.02 | 29.67 |
| size 12–16 px | 46.27 | 43.30 |
| size 16–24 px | 58.53 | 54.59 |
| size 24–40 px | 67.54 | 59.14 |
| size 40+ px | 69.40 | 62.11 |
| density 1–9 buildings | 60.78 | 56.76 |
| density 10–49 | 59.84 | 51.61 |
| density 50+ | 49.13 | 49.54 |
| Africa | 57.07 | 51.49 |
| Myanmar | 57.36 | 52.08 |
| other regions | 58.23 | 43.32 |
| pixel IoU | 77.75 | 74.41 |
| empty chips with ≥ 1 false positive | 10.2 % | 10.9 % |

Size is the side of the square of equal area (√area in px); a 40 px side is a
building of about 1,600 px.

## Known Limitations

- **Very small buildings** (under 8 px side) are weak: PQ 12.6 on validation and
  11.6 on the benchmark.
- **False positives on empty chips**: 10.9 % of empty benchmark chips receive at
  least one spurious detection (10.2 % on validation).
- **Merges versus splits**: centreness seeds split fewer large buildings than the
  previous distance-transform seeds but merge more touching ones. This was
  measured on the predecessor of this model, which uses the same recipe.
- **Fine-tuning supervises only the mask channel.** The decoder is shared by all
  four channels, so fine-tuning may also shift the centreness channel that seeds
  the watershed. How well touching buildings are separated after fine-tuning has
  not been measured.

## Why Instance Metrics

Pixel IoU cannot tell a correctly separated building from several merged into
one: a single blob over a terrace of four scores pixel IoU 86 and PQ 0, while
the same four separated but traced one pixel too wide score pixel IoU 86 and PQ
86. PQ is reported as the primary metric throughout.

## Usage

Four output channels: mask, normalised distance transform, boundary,
centreness. Instances come from a marker-controlled watershed seeded by the
h-maxima of the centreness channel inside the mask. The decode settings are in
`best_threshold.json["decode"]` (mask threshold 0.5, h 0.4, minimum 30 px, d4 TTA).

```python
import json, torch, numpy as np, segmentation_models_pytorch as smp
from huggingface_hub import hf_hub_download
from scipy import ndimage
from skimage.morphology import h_maxima
from skimage.segmentation import watershed

repo = "nilsho01/unet-buildings-autorun-v2"
dec = json.load(open(hf_hub_download(repo, "best_threshold.json")))["decode"]

net = smp.Unet(encoder_name="efficientnet-b4", encoder_weights=None, in_channels=3, classes=4)
net.load_state_dict(torch.load(hf_hub_download(repo, "best.pth"), map_location="cpu"))
net.eval()

# x: float32 tensor [1, 3, 256, 256], ImageNet-normalised RGB
with torch.no_grad():
    prob = torch.sigmoid(net(x))[0].numpy()

mask = prob[0] > dec["mask_threshold"]
cent = prob[3].copy()
cent[~mask] = cent[mask].min() if mask.any() else 0
seeds = ndimage.label(h_maxima(cent, dec["h"]).astype(bool) & mask)[0]
inst = watershed(-ndimage.distance_transform_edt(mask), markers=seeds, mask=mask)
sizes = np.bincount(inst.ravel())
inst[np.isin(inst, np.flatnonzero(sizes < dec["min_instance_px"]))] = 0
```

`model.onnx` in the same repository is the same network at opset 18 with a
dynamic batch dimension; it matches the PyTorch output to 6e-5.

## Architecture and Training

`smp.Unet`, EfficientNet-B4 encoder (ImageNet), `dist4` head (4 output
channels). 60 epochs, batch 32, Adam at 2e-3 (decoder) / 2e-4 (encoder), EMA
0.99, 256 × 256, seed 42. The published checkpoint is the epoch with the best
validation PQ (60 of 60). Folds are split by spatial tile blocks, because
adjacent OAM tiles share buildings and a per-chip split leaks.

Training data is the pool from
[hotosm/vhr-building-segmentation](https://huggingface.co/datasets/hotosm/vhr-building-segmentation)
plus 3,293 chips from the Open Cities AI Challenge (tier 1). Most of the Open
Cities chips from Zanzibar are empty rural imagery and were kept as negative
examples.

## Fine-tuning

The encoder is frozen; only the decoder is trained. fAIr's fine-tuning pipeline
supervises the mask channel (BCE + Dice loss). Default budget: 15 epochs, batch
8, learning rate 5e-5. Spatial block split with `block_size=4` on OAM tile
coordinates to prevent data leakage.

## Licence and Attribution

Model weights: Apache-2.0. Available at
[nilsho01/unet-buildings-autorun-v2](https://huggingface.co/nilsho01/unet-buildings-autorun-v2).

Training data:
- [hotosm/vhr-building-segmentation](https://huggingface.co/datasets/hotosm/vhr-building-segmentation):
  imagery © OpenAerialMap contributors; labels © OpenStreetMap contributors,
  ODbL-1.0.
- Open Cities AI Challenge tier 1 (GFDRR Labs, Radiant MLHub, 2020;
  [doi 10.34911/rdnt.f94cxb](https://doi.org/10.34911/rdnt.f94cxb)): imagery
  CC BY 4.0, labels ODbL-1.0. Please attribute "Open Cities AI Challenge,
  GFDRR Labs / Radiant MLHub, 2020".

No CC BY-NC data (e.g. RAMP) was used for this model.
