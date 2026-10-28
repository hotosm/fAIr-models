# unet-effb4-buildings

## Overview

Building footprint instance segmentation for HOT's fAIr. A U-Net with an
EfficientNet-B4 encoder predicts four maps per pixel; individual buildings are
separated by a watershed seeded from the centreness map and returned as GeoJSON
polygons. The model is pretrained on very high resolution aerial and drone
imagery (about 30 cm per pixel, OpenAerialMap zoom 19) from 21 countries and is
meant as a general starting point that mappers fine-tune on a few labelled
chips of their own area.

**Intended use.** Assisting humanitarian and community mappers: it proposes
building outlines that a person reviews before they enter OpenStreetMap. It is
not meant for unreviewed import, for counting people, or for imagery much
coarser than 50 cm per pixel.

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

## Architecture

`smp.Unet` with an EfficientNet-B4 encoder and a four-channel output head. The
input is a 3-band RGB chip of 256 × 256 pixels, normalised with ImageNet
statistics. The four output channels are a building mask, a normalised distance
transform, a building boundary and an instance centreness map, all as logits of
the same 256 × 256 size. Inference uses the mask and the centreness map: peaks
of the centreness map inside the mask seed a marker-controlled watershed, and
instances smaller than 30 pixels are dropped. The distance and boundary
channels are auxiliary training targets. Each output polygon carries `class`
(1 = building) and `score` (mean mask probability over the polygon).

## Pretrained Source

The weights were trained for 60 epochs (batch 32, Adam at 2e-3 for the decoder
and 2e-4 for the encoder, EMA 0.99, seed 42) from ImageNet-initialised encoder
weights. The training pool has 19,199 chips: chips from
[hotosm/vhr-building-segmentation](https://huggingface.co/datasets/hotosm/vhr-building-segmentation)
(OpenAerialMap imagery, OpenStreetMap labels, ODbL-1.0) plus 3,293 chips from
the Open Cities AI Challenge tier 1 (imagery CC BY 4.0, labels ODbL-1.0). Most
Open Cities chips from Zanzibar are empty rural imagery and were kept as
negative examples. Folds are split by spatial tile blocks, because adjacent
OpenAerialMap tiles share buildings and a per-chip split leaks. The published
checkpoint is the epoch with the best validation PQ (60 of 60).

## Results

Panoptic quality (PQ = SQ × RQ) on chips that contain buildings. Ground-truth
instances are connected components of the label mask, instances under 30 px are
ignored, predictions are averaged over the 8 flips and transposes of each chip,
and the decode settings were selected on validation. Nothing was tuned on the
benchmark.

| | val PQ | val SQ | val RQ | bench PQ | bench SQ | bench RQ |
|---|---|---|---|---|---|---|
| previous version of this model (`dist3` head) | 56.06 | 79.03 | 70.93 | 49.30 | 76.28 | 64.63 |
| **this model** | **57.30** | **79.53** | **72.06** | **51.08** | **76.69** | **66.60** |

Validation is a spatially blocked cross-validation fold (2,651 chips, 1,532 with
buildings). The benchmark is described under Evaluation Dataset below. Both
rows are single training runs (one seed). A from-scratch rerun of the `dist3`
recipe reproduced its score to the second decimal, which suggests run-to-run
noise below the 1.2–1.8 PQ gap shown here, but that was not measured for this
model.

fAIr runs inference without the flip and transpose averaging by default because
it costs eight forward passes per chip. On the predecessor of this model,
dropping it cost about 1 PQ. Setting `tta` to true reproduces the table.

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

### Why instance metrics

Pixel IoU cannot tell a correctly separated building from several merged into
one: a single blob over a terrace of four scores pixel IoU 86 and PQ 0, while
the same four separated but traced one pixel too wide score pixel IoU 86 and PQ
86. PQ is therefore the primary metric.

## Evaluation Dataset

[nilsho01/vhr-buildings-benchmark-v1](https://huggingface.co/datasets/nilsho01/vhr-buildings-benchmark-v1)
has 400 chips of 256 × 256 pixels whose labels were reviewed twice by hand. It
is stratified into four density bands of 100 chips (empty, 1–9, 10–49 and
50–149 buildings), each split across Myanmar, Africa and other regions, so a
model cannot score well by handling only sparse scenes and false positives on
empty chips stay visible. Each record holds `tile_id`, the RGB `image`, a
binary building `mask`, the density `band`, `region`, `country`,
`project_name`, `num_buildings` and `mask_source`. The licence is ODbL,
inherited from `hotosm/vhr-building-segmentation` (OpenAerialMap imagery,
OpenStreetMap labels).

## Limitations

- **Very small buildings** (under 8 px side, about 2.4 m at zoom 19) are weak:
  PQ 12.6 on validation and 11.6 on the benchmark.
- **False positives on empty chips**: 10.9 % of empty benchmark chips receive
  at least one spurious detection (10.2 % on validation).
- **Geographic bias**: the training pool is dominated by Myanmar and Africa.
  On the benchmark, chips from other regions score 43.3 PQ against 51–52 for
  those two, so new regions should be fine-tuned before use.
- **Resolution**: trained at about 30 cm per pixel. Coarser imagery, such as
  most satellite basemaps, is outside the training distribution.
- **Merges versus splits**: centreness seeds split fewer large buildings than
  distance-transform seeds but merge more touching ones. This was measured on
  the predecessor of this model, which uses the same recipe.
- **Fine-tuning supervises only the mask channel.** The decoder is shared by
  all four channels, so fine-tuning may also shift the centreness channel that
  separates touching buildings. How well buildings are separated after
  fine-tuning has not been measured.

## Ethical Considerations

Predictions are proposals for human review, not finished map data. Importing
them unreviewed would carry the model's errors into OpenStreetMap, and those
errors are not spread evenly: small structures and regions outside the
training distribution are missed more often, which in practice tends to mean
informal settlements and places that are already under-mapped. Counts derived
from the output would under-represent those communities in any planning or aid
allocation built on them.

The labels come from volunteer OpenStreetMap mapping and carry its conventions
and gaps. Building maps can also be used for purposes other than humanitarian
response, including surveillance or targeting in conflict settings; the model
adds no information beyond what the imagery shows, but it makes extraction
faster. Decisions about where to map, and who receives the results, should
follow the responsible-data practices of the organisation running the mapping.

## Usage

**Installation.** The model runs inside the Docker images built from the
`Dockerfile` in this directory, as all fAIr models do; nothing has to be
installed on the host besides Docker. The `runtime` image is used for
fine-tuning and batch prediction, and the distroless `inference` image serves
the ONNX model for live prediction. With the repository's tooling,
`just build unet_effb4_buildings` builds the training image and
`just example unet_effb4_buildings` runs fine-tuning and prediction end to end
against the local stack.

**Inference.** fAIr passes a directory of GeoTIFF chips under the `image` input.
Each chip is resized to 256 × 256 if needed, run through the ONNX model,
decoded into instances and returned as one GeoJSON `FeatureCollection` of
polygons in EPSG:4326. The parameters are `confidence_threshold` (mask
threshold, default 0.5), `seed_h` (minimum height of a centreness peak, default
0.4; higher merges more), `min_instance_px` (default 30) and `tta` (flip and
transpose averaging, default off).

**Fine-tuning.** fAIr supplies OpenAerialMap chips and a GeoJSON of building
polygons. The pipeline loads the published weights, freezes the encoder and
trains the decoder with a BCE plus Dice loss on the mask channel, using a
spatial block split (4 × 4 OpenAerialMap tiles) so that neighbouring chips do
not leak between training and validation. The default budget is 15 epochs,
batch 8, learning rate 5e-5. Validation reports pixel accuracy, mean IoU and
per-class IoU, and the result is exported to ONNX.

## Hardware Requirements

Inference runs on CPU. On an Intel i9-10900KF (20 threads), one 256 × 256 chip
takes about 29 ms through the ONNX model and a few milliseconds to decode;
with `tta` enabled it takes about 350 ms. The ONNX file is 78 MB. Fine-tuning
runs on CPU for small smoke tests, but a CUDA GPU is recommended for real
datasets.

## Third-party Components

| Component | Licence |
|---|---|
| PyTorch | BSD-3-Clause |
| segmentation-models-pytorch | MIT |
| timm | Apache-2.0 |
| TorchGeo | MIT |
| ONNX | Apache-2.0 |
| ONNX Script | MIT |
| ONNX Runtime | MIT |
| NumPy | BSD-3-Clause |
| SciPy | BSD-3-Clause |
| scikit-image | BSD-3-Clause |
| rasterio | BSD-3-Clause |
| pyproj | MIT |
| Pillow | MIT-CMU |
| huggingface_hub | Apache-2.0 |
| ZenML | Apache-2.0 |
| fair-py-ops (fAIr platform library) | AGPL-3.0-only |
| EfficientNet-B4 ImageNet weights (`smp-hub/efficientnet-b4.imagenet`, encoder initialisation) | listed as "other" on the hub; ported from Google's EfficientNet release |

Exact versions are pinned in the `Dockerfile`.

## Citation

- Ronneberger, Fischer and Brox (2015). U-Net: Convolutional Networks for
  Biomedical Image Segmentation. [arXiv:1505.04597](https://arxiv.org/abs/1505.04597)
- Tan and Le (2019). EfficientNet: Rethinking Model Scaling for Convolutional
  Neural Networks. [arXiv:1905.11946](https://arxiv.org/abs/1905.11946)
- Open Cities AI Challenge, GFDRR Labs / Radiant MLHub (2020).
  [doi:10.34911/rdnt.f94cxb](https://doi.org/10.34911/rdnt.f94cxb)

## License

Code and model weights: Apache-2.0 (matches `properties.license` in
`stac-item.json`).

Training data attribution:
- [hotosm/vhr-building-segmentation](https://huggingface.co/datasets/hotosm/vhr-building-segmentation):
  imagery © OpenAerialMap contributors; labels © OpenStreetMap contributors,
  ODbL-1.0.
- Open Cities AI Challenge tier 1 (GFDRR Labs, Radiant MLHub, 2020): imagery
  CC BY 4.0, labels ODbL-1.0. Please attribute "Open Cities AI Challenge,
  GFDRR Labs / Radiant MLHub, 2020".

No CC BY-NC data (e.g. RAMP) was used for this model.
