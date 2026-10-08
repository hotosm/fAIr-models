# unet-effb4-buildings

Building footprint instance segmentation for HOT's fAIr. Trained on a pool of
**18,557 chips** filtered from
[hotosm/vhr-building-segmentation](https://huggingface.co/datasets/hotosm/vhr-building-segmentation)
by a label-quality classifier, after manual review established that **27.2% of
that dataset's labels** (n = 6,459 reviewed) are unusable.

Weights and config: [nilsho01/unet-effb4-dist3-buildings](https://huggingface.co/nilsho01/unet-effb4-dist3-buildings)

## Summary

| | |
|---|---|
| Task | Building footprint instance segmentation |
| Input | 3-band RGB GeoTIFF chips, VHR (~30 cm GSD), 256 × 256 |
| Output | GeoJSON `FeatureCollection` of `Polygon` features in EPSG:4326 |
| Coverage | Global; training pool covers 19 countries across 3 regions (Myanmar 61%, Africa 21%, other 18%) |
| License | Apache-2.0 |

## Results

Mean ± sd over five cross-validation folds. Threshold selected on each fold's
own validation split, never on a test set. The published fold (fold 3) was
selected by highest validation PQ.

### nilsho01/vhr-buildings-benchmark-v1 (400 chips, reviewed and corrected labels)

| model | PQ | SQ | RQ | pixel F1 | pixel IoU |
|---|---|---|---|---|---|
| **this model (5 folds)** | **47.79 ± 0.23** | **75.44 ± 0.17** | **63.35 ± 0.27** | **84.43 ± 0.11** | **73.05 ± 0.16** |
| shipped dinov3s_upernet_hot | 33.92 | 72.08 | 47.06 | 83.03 | 70.98 |

This benchmark is disadvantageous to this model: all 400 chips lie in
vhr-buildings' train split, so the shipped baseline trained on every chip it is
scored on, and this model never saw one.

### hotosm/vhr-building-segmentation test (7,236 chips, raw OSM labels)

Neither model trained on this split.

| metric | this model (5 folds) | shipped DINOv3 |
|---|---|---|
| **PQ** | **30.02 ± 0.26** | 27.81 |
| **SQ** | **78.45 ± 0.29** | 76.82 |
| **RQ** | **38.28 ± 0.30** | 36.20 |
| pixel F1 | 60.46 ± 0.48 | **61.50** |
| pixel IoU | 43.33 ± 0.49 | **44.41** |

Ahead on instance metrics, behind on pixel metrics by ~1 point. The shipped
model predicts 75% of the true object count against this model's ~116%: a model
that under-predicts is rewarded by an area-weighted metric and penalised by an
object-weighted one.

### How the baseline was decoded

Both families emit three channels but order them differently — this model as
`[mask, interior, boundary]`, `dinov3_hot` as `[mask, boundary, signed distance]`.
The baseline's channels were reordered and its signed distance mapped from
[-1, 1] to [0, 1] so that one decoder and one threshold protocol apply to both.

The baseline was decoded with that shared decoder, **not** with its own tuned
post-processing (`h_maxima_depth` 0.2, `seed_min_distance` 6, `min_area_m2`,
Douglas-Peucker simplification, regularisation). Holding the decoder fixed
isolates what the network predicts, but it means these numbers are not the
baseline's shipped operating point and are not comparable to figures published
elsewhere for it.

### What the held-out split measures

The `test` split is drawn from the same OpenStreetMap-derived labelling process
as `train` and was not reviewed, so there is no reason to expect its labels to
be cleaner. Manual review of 6,459 `train` chips found 27.2% carrying errors
severe enough to make them unusable — shifted footprints, missing buildings, and
annotations with no building under them. Applied to the two splits, the
label-quality classifier passes `test` chips at close to the rate it passes
`train` chips, which is consistent with a comparable noise level.

The errors are not distributed uniformly. They concentrate by country and by
background type, which makes them learnable: a model trained on the unfiltered
pool can acquire a region's labelling convention, including its systematic
shift, and is then rewarded for reproducing it here. A model trained on filtered
labels predicts the building rather than the convention and is penalised for the
difference. The comparison on this split therefore does not isolate segmentation
quality from agreement with a particular annotation style.

This split is reported because it is the one the shipped model is evaluated on
and excluding it would be a convenient omission. The reviewed benchmark above is
the comparison this work puts forward.

### Per fold

| fold | val PQ | benchmark PQ | hotosm test PQ |
|---|---|---|---|
| 0 | 43.50 | 47.90 | 29.98 |
| 1 | 44.48 | 47.67 | 30.33 |
| 2 | 43.58 | 47.68 | 29.62 |
| **3** | **45.73** (selected) | 47.57 | 30.16 |
| 4 | 44.78 | **48.14** | 30.03 |

Selecting on the benchmark instead would have given fold 4 (48.14 PQ) — that
would be selection on the test set. The published weights use fold 3.

## Known Limitations

**False-positive rate on empty chips**: 37.4% ± 2.3 of chips containing no
buildings receive at least one spurious detection on the hotosm test split,
against 8.3% for the shipped model (15.8–19.8% on the reviewed benchmark). The
rate rose as training data was added. Likely a domain-coverage effect: the
training pool is ~61% Myanmar, and vhr-buildings' validation and test splits
contain no Myanmar at all.

**Touching buildings**: the watershed step separates most touching instances,
but dense informal settlements with very small gaps may still be merged.

## Why Instance Metrics

Pixel IoU cannot tell a correctly separated building from several merged into
one: a single blob over a terrace of four scores pixel IoU 86 and PQ 0, while
the same four separated but traced one pixel too wide score pixel IoU 86 and PQ
86. PQ is reported as the primary metric throughout.

## Usage

Three output channels: mask logit, per-instance normalised distance transform
(EDT), instance boundary logit. Instances come from marker-controlled watershed
seeded by channel 1 — connected components on channel 0 alone merges roughly
2.3 buildings per blob.

```python
import json, torch, numpy as np, segmentation_models_pytorch as smp
import scipy.ndimage, skimage.segmentation
from huggingface_hub import hf_hub_download

repo = "nilsho01/unet-effb4-dist3-buildings"
thr  = json.load(open(hf_hub_download(repo, "best_threshold.json")))["threshold"]  # 0.50

net = smp.Unet(encoder_name="efficientnet-b4", encoder_weights=None,
               in_channels=3, classes=3)
net.load_state_dict(torch.load(hf_hub_download(repo, "best.pth"),
                               map_location="cpu")["model"])
net.eval()

# x: float32 tensor [1, 3, 256, 256], ImageNet-normalised
with torch.no_grad():
    prob = torch.sigmoid(net(x))[0].numpy()

mask = prob[0] > thr
edt  = prob[1]
seeds = scipy.ndimage.label((edt > 0.5) & mask)[0]
dist  = scipy.ndimage.distance_transform_edt(mask)
instances = skimage.segmentation.watershed(-dist, markers=seeds, mask=mask)
```

`best_threshold.json` matters — folds selected 0.40–0.50, and the operating
point is not interchangeable between folds.

## Architecture

`smp.Unet`, EfficientNet-B4 encoder (ImageNet), decoder channels 256/128/64/32/16,
`dist3` head (3 output channels). 40 epochs, batch 32, Adam at 2e-3 (decoder) /
2e-4 (encoder), 256 × 256, seed 42. Folds split by 4 × 4 spatial tile blocks —
adjacent OAM tiles share buildings, so a per-chip split leaks.

## Label Quality

6,459 chips were reviewed manually; 27.2% were unusable — 1,471 needing
correction (label shifted relative to imagery) and 287 with labels too poor to
correct. The surviving 4,701 reviewed chips formed the quality-classifier
training set. The classifier was then applied to the full ~57k hotosm dataset;
the training pool of 18,557 chips has estimated label quality > 90%.

The quality classifier uses:
1. A frozen EfficientNet-B4 UNet run on each chip to extract prediction features
   (IoU, coverage, etc.)
2. A small XGBoost classifier trained on reviewed chips

## Fine-tuning

The encoder is frozen; only the decoder is trained. fAIr's fine-tuning pipeline
supervises the mask channel (BCE + Dice loss). Default budget: 15 epochs, batch
8, learning rate 5e-5. Spatial block split with `block_size=4` on OAM tile
coordinates to prevent data leakage.

## Licence and Attribution

Model weights: Apache-2.0. Available at
[nilsho01/unet-effb4-dist3-buildings](https://huggingface.co/nilsho01/unet-effb4-dist3-buildings).

Training data: [hotosm/vhr-building-segmentation](https://huggingface.co/datasets/hotosm/vhr-building-segmentation)
— OpenAerialMap imagery + OpenStreetMap labels, ODbL. No other footprint source
contributed to the weights or to the filtering.
