# yolo11m-swimming-pools

Detects outdoor swimming pools in very high resolution RGB aerial imagery and returns one box polygon per pool. The base weights are the
YOLO11m detector [`mozilla-ai/swimming-pool-detector`](https://huggingface.co/mozilla-ai/swimming-pool-detector), unchanged and exported to
ONNX at 512 x 512, the size it was trained at, so it can be used directly on OpenAerialMap (OAM) chips or finetuned on OAM chips with OSM pool
labels. The model never predicts a pool outline, only an axis-aligned box, which fAIr exports as a rectangular polygon.

## How it works

```
OAM chip  ->  resize to 512  ->  ONNX network  ->  confidence filter + NMS  ->  EPSG:4326 box polygons
OAM-x-y-z.tif    / 255          output0 (1,5,5376)     (0.40, 0.45, cap 300)        GeoJSON features
```

- Ultralytics YOLO11m, one class (swimming pool), anchor-free head.
- The ONNX graph takes one chip as `images` (float32, 1 x 3 x 512 x 512, RGB in [0, 1]) and returns `output0` (1 x 5 x 5376): for each
  anchor the box centre, width and height in model pixels and one pool score. The graph holds no normalisation, so the division by 255 is the
  whole of it.
- Everything else lives in the pinned [`spd-hot`](https://github.com/AbdelrahmanKatkat/spd-hot) helper library: the resize, the confidence
  filter, NMS and the per-chip cap, mapping the boxes back to chip pixels, georeferencing to EPSG:4326, the label boxes, training and
  scoring. There is no sliding window: a 256 px chip is one pass.
- The pack itself is a thin adapter: `pipeline.py` wires the library into the fAIr steps and renames the output properties.

## Inputs and outputs

- **Input:** RGB chips named `OAM-<x>-<y>-<z>.<ext>`. The chip is resized to 512 whatever its size, and the resize does not read the zoom.
  **Use zoom 18**: the model was trained on 512 px Mapbox tiles at zoom 18, about `0.2986 x cos(latitude)` m per pixel, and a 256 px chip of
  a zoom-18 tile covers the same ground as one training image. At another zoom the scale is wrong (at zoom 20 pools look about four times
  larger than the model saw), so the STAC preview declares zoom 18 as the only working zoom. Scaling each chip from its zoom, as other fAIr
  models do, would widen that range; it is not implemented.
- **Output:** one polygon per pool in EPSG:4326, with the properties `class` (always 0), `label` (`swimming_pool`), `confidence` and `source`
  (the chip file name).
- **Training labels:** a GeoJSON of pool polygons (see Labels).

## Labels

Training labels are expected to come from OpenStreetMap through fAIr: a GeoJSON of `leisure=swimming_pool` polygons. The detector needs one
box per pool in chip pixels, so the labels are converted each time a step runs, by `spd_hot`'s `build_chip_labels` (the dataset itself is
never changed): each polygon becomes its bounding box, reprojected from the label CRS into the chip CRS and clipped to the chip.

- The library does not filter by feature properties, so the file must hold outdoor pool polygons only (no buildings or other features).
- A chip with no pool left stays in training and validation as a background example.
- OSM is incomplete: pools nobody mapped are missing from the labels, and the model's boxes on them count as false positives.

## Pretrained source

The base weights are `model.pt` of [`mozilla-ai/swimming-pool-detector`](https://huggingface.co/mozilla-ai/swimming-pool-detector),
unchanged. Mozilla AI fine-tuned `yolo11m.pt` on
[`mozilla-ai/osm-swimming-pools`](https://huggingface.co/datasets/mozilla-ai/osm-swimming-pools) (512 x 512 Mapbox tiles at zoom 18, labels
from OpenStreetMap `leisure=swimming_pool`, north-west Iberia) for 10 epochs at `imgsz=512`. The checkpoint and the ONNX file exported
from it at 512 are both published at
[`AbdelrahmanU3/swimming-pool-detector-OAM`](https://huggingface.co/AbdelrahmanU3/swimming-pool-detector-OAM), and the STAC item pins
their commit. The weights are AGPL-3.0 licensed, and this model is AGPL-3.0-only to match; the labels are OpenStreetMap data (ODbL).

## Training

Finetuning continues from the base checkpoint (the Mozilla weights, or a checkpoint of this model) with Ultralytics, through
`spd_hot.train`. The defaults are Mozilla's training settings. One default differs on purpose: the first 10 layers (the backbone) are
frozen, so it keeps what it learned and cannot forget it, while the neck and head adapt. A visible GPU is used, else the CPU. The weights
of the best epoch are kept.

The step rebuilds the labels from the chips and the GeoJSON it was given, so nothing is shared between steps. The folder Ultralytics needs
holds links to the chips and small label files, and it is created in a temporary folder of the training step and removed when it ends, so
no dataset or run accumulates between trainings.

The train and validation chips are split in spatial blocks on OAM tile coordinates, because neighbouring chips share pools and
surroundings. A chip whose name is not `OAM-<x>-<y>-<z>` is rejected. Reported metrics are box precision, recall and F1 at IoU 0.5 on the
validation split, measured through the exported ONNX and the same decoding that serves predictions, and mAP50 and mAP50-95 computed as
Ultralytics validates (confidence 0.001, NMS IoU 0.7).

### Hyperparameters

Declared in the STAC item (`fair:hyperparameters_spec`) with their ranges; the defaults are:

| Training          | Default | Meaning                                                          |
| ----------------- | ------- | ---------------------------------------------------------------- |
| `epochs`          | 10      | finetuning epochs                                                |
| `batch_size`      | 16      | chips per batch                                                  |
| `learning_rate`   | 0.01    | initial learning rate of AdamW                                   |
| `patience`        | 3       | epochs without improvement before training stops, 0 disables it  |
| `cos_lr`          | true    | cosine learning-rate schedule                                    |
| `flipud`          | 0.5     | probability of a vertical flip                                   |
| `scale`           | 0.0     | scale augmentation                                               |
| `translate`       | 0.0     | translation augmentation                                         |
| `freeze_encoder`  | true    | keep the first 10 layers frozen                                  |
| `val_ratio`       | 0.2     | share of the chips held out as spatial validation               |
| `split_seed`      | 42      | seed of the block order in the split                             |
| `block_size`      | 4       | side of a spatial block, in tile units                           |

| Inference              | Default | Meaning                                                   |
| ---------------------- | ------- | --------------------------------------------------------- |
| `confidence_threshold` | 0.40    | minimum score of a kept box                               |
| `nms_threshold`        | 0.45    | IoU above which overlapping boxes of one chip are removed |

The model input size is fixed at 512 by the ONNX graph and is not a hyperparameter.

## Limitations

- Trained on Mapbox imagery of north-west Iberia (Galicia and Viana do Castelo). Other regions, sensors, seasons and roof types are
  unmeasured.
- Only zoom 18 has the training scale; other zooms need the chip rescaled first.
- Boxes, not outlines. A pool cut by a chip border is returned as a box in each chip.
- OSM labels are incomplete: pools nobody mapped count as false positives when the model finds them, and as background in training.
- The default confidence (0.40) is not tuned on OAM imagery; choose it on labelled chips of the target area.
- Training is slow on CPU. A GPU is requested by the STAC item; without one, training falls back to the CPU.

## License

AGPL-3.0-only
