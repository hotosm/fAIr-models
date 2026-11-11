# deepforest-tree

Detects individual tree crowns in very high resolution RGB aerial imagery and returns one box polygon per crown. The base weights are
`NEON_OAM`: the [`weecology/deepforest-tree`](https://huggingface.co/weecology/deepforest-tree) detector (DeepForest, NEON checkpoint)
fine-tuned on OpenAerialMap (OAM) chips of Freetown, so it can be used directly on OAM chips or finetuned further on OAM chips with OSM
tree labels. The model never predicts a crown outline, only an axis-aligned box, which fAIr exports as a rectangular polygon.

## Sample results

![Three Freetown test tiles: the OAM chip, the OSM tree labels and the NEON_OAM predictions](https://huggingface.co/AbdelrahmanU3/NEON_OAM/resolve/367506c984369015337a100528b03d55902ac854/sample_results.jpg)

Three tiles of the Freetown test split (zoom 20), one per row. Left: the OAM chip. Middle: the OSM tree labels (green). Right: the boxes
of `NEON_OAM` at the default confidence of 0.10, with their scores (red). The top row is a good case, the middle row a typical dense one, and
the bottom row a hard one: vegetation without a label gives several boxes against one label, which counts as false positives. The labels are
OSM trees mapped as points, so many real trees are not labelled at all.

## How it works

```
OAM chip  ->  scale by zoom  ->  ONNX network  ->  confidence filter + NMS  ->  EPSG:4326 box polygons
OAM-x-y-z.tif   (chip name)      boxes, scores, labels     (0.10, 0.05, cap)         GeoJSON features
                                 before NMS
```

- torchvision RetinaNet with a ResNet-50 body and FPN, one class (tree), 32.2 million parameters.
- The ONNX graph takes one chip as `image` (float32, 3 x H x W, RGB in [0, 1], any height and width) and returns `boxes`, `scores` and
  `labels`: every candidate box in model-input pixels, best score first, before non-maximum suppression. ImageNet normalisation, padding to
  a multiple of 32, the network, anchors, top-1000 per level, box decoding and clipping are all inside the graph.
- Everything else lives in the pinned [`td-hot`](https://github.com/AbdelrahmanKatkat/td-hot) helper library: the scale applied to each chip
  before the network, the confidence filter, NMS (0.05) and the per-chip cap, georeferencing to EPSG:4326, the label boxes, training and scoring.
  There is no sliding window: a 256 px chip is one pass. Its parity with torchvision and the DeepForest package is documented in the library's
  `docs/Theory.md`.
- The pack itself is a thin adapter: `pipeline.py` wires the library into the fAIr steps and renames the output properties.

## Inputs and outputs

- **Input:** RGB chips named `OAM-<x>-<y>-<z>.<ext>`. The scale is `chip ground resolution / 0.05 m`, where the ground resolution comes from
  the zoom and row in the name (Web Mercator, `cos(latitude)`); a scale above 16 is rejected, and the scaled chip is capped at 1024 px on its
  longest side so memory does not grow with the zoom. A zoom 20 chip at Freetown (14.8 cm per pixel) is upscaled about 3x. Chips must
  therefore keep their OAM names.
- **Output:** one polygon per crown in EPSG:4326, with the properties `class` (always 1), `label` (`tree`), `confidence` and `source` (the
  chip file name).
- **Training labels:** a GeoJSON of OSM tree points or crown polygons, converted to boxes (see Labels).

## Labels

Training labels are expected to come from OpenStreetMap through fAIr: a GeoJSON of mapped trees, as `natural=tree` points and tree crown
polygons. The network needs one box per crown in chip pixels, so the labels are converted each time a step runs, by `td_hot`'s
`build_chip_labels` (the dataset itself is never changed):

| OSM feature                           | What the model trains on                                                                                                                            |
| ------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| Polygon (crown outline)               | its bounding box; the outline itself is not used                                                                                                    |
| Point (`natural=tree`)                | a square crown centred on the point, with a side of the `diameter_crown` tag in metres (5 m if missing or not a positive number, for example `3 m`) |
| Any other geometry (line, multi-part) | one box around all of its coordinates, so a long line or a far-apart multi-part feature gives one large box                                         |

- The `diameter_crown` tag is read from the feature's `tags` or, if there are none, from its properties. A point is taken as longitude and
  latitude in degrees, and a metre is approximated as 1/111,320 of a degree of latitude (within 1%), with longitude scaled by `cos(latitude)`.
- Labels are reprojected from their own CRS (EPSG:4326 unless the file says otherwise) into each chip's CRS, then clipped to the chip. A box
  outside the chip is dropped, and so is a box thinner than 2 px after clipping.
- Chips need georeferencing and the OAM name, and a chip with no label left is left out of both training and validation.
- OSM is incomplete: trees nobody mapped are missing from the labels, and the model's boxes on them count as false positives.

## Pretrained source

The base checkpoint is `NEON_OAM`, hosted at [`AbdelrahmanU3/NEON_OAM`](https://huggingface.co/AbdelrahmanU3/NEON_OAM) (the STAC item pins its
commit). It starts from the NEON model of the DeepForest project, published under the MIT license, which was trained on NEON airborne RGB at
about 10 cm per pixel: millions of boxes generated by a LiDAR tree detector, then fine-tuned on hand-annotated crowns.

![One NEON test image: the ground truth, the original NEON checkpoint and NEON_OAM](https://huggingface.co/AbdelrahmanU3/NEON_OAM/resolve/367506c984369015337a100528b03d55902ac854/pretrained_vs_finetuned.jpg)

The same NEON test image (61 hand-annotated trees) with the ground truth (green), the original NEON checkpoint (red) and `NEON_OAM` (blue),
each at the confidence that gave its best F1 on this image. The fine-tuned model finds fewer of these NEON trees: it traded some NEON-style
performance for OAM-style performance.

The fine-tune used the `hotosm/fAIr-guru-td-trees` OAM dataset of Freetown (zoom 20): OSM `tree` points became square boxes with the side
given by `diameter_crown` (tiles without a diameter were dropped), tiles dominated by `tree_group` areas were dropped and the rest of such areas
was filled with the mean colour. It trained 23.6 million weights (the last ResNet stage, the FPN and the heads) on 34,005 chips for 19 epochs of
a random 20% of the tiles each, with SGD (learning rate 1e-3, encoder learning rate x0.385), 300 warmup steps and a per-batch cosine schedule.
The best epoch by validation loss (epoch 14) is the published file. The recipe is reproducible with `scripts/finetune.py` in td-hot.

| Measured on                                         | Box precision | Recall | F1    | Setting                                    |
| --------------------------------------------------- | ------------- | ------ | ----- | ------------------------------------------ |
| Freetown validation (3,457 chips)                   | n/a           | n/a    | 0.554 | IoU 0.4, best of a confidence sweep (0.10) |
| Freetown test (5,244 chips, never used for choices) | 0.395         | 0.546  | 0.458 | IoU 0.4, confidence 0.10, NMS 0.05         |
| Freetown test                                       | 0.302         | 0.418  | 0.351 | IoU 0.5, confidence 0.10, NMS 0.05         |
| One NEON test image (61 trees)                      | 0.594         | 0.311  | 0.409 | the original NEON checkpoint scores 0.692  |

The labels are boxes derived from OSM points and a diameter, so they are noisy; the scores are lower at IoU 0.5 than at 0.4.

## Training

Finetuning continues from the base weights with torchvision's focal and L1 box losses, SGD with momentum 0.9, a cosine schedule stepped per batch
(optionally with a linear warmup) and horizontal-flip augmentation, through `td_hot.train`. By default the ResNet body is frozen
and only the FPN and detection heads are trained, which keeps memory low, limits forgetting and suits datasets of a few hundred chips. The
`freeze_encoder` hyperparameter unfreezes everything, and `unfreeze_from` (`layer4` to `layer1`) trains the later ResNet stages only; in a short
test on Freetown, `layer4` matched the fully unfrozen model and kept more of the NEON skill. BatchNorm statistics are never updated. The weights
of the best epoch by validation loss are kept, and training stops early after `early_stop_patience` epochs without improvement.

The labels are converted to chip-pixel boxes as described in Labels above.

The train and validation chips are split in spatial blocks on OAM tile coordinates, because neighbouring chips share canopy and trees cut at
chip borders. A chip whose name is not `OAM-<x>-<y>-<z>` is rejected. Reported metrics are box precision, recall and F1 at IoU 0.5 on the
validation split, measured through the exported ONNX and the same decoding that serves predictions, with the inference hyperparameters.

### Hyperparameters

Declared in the STAC item (`fair:hyperparameters_spec`) with their ranges; the defaults are:

| Training            | Default | Meaning                                                                  |
| ------------------- | ------- | ------------------------------------------------------------------------ |
| `epochs`            | 5       | finetuning epochs                                                        |
| `batch_size`        | 2       | chips per batch (each chip is scaled about 3x at zoom 20 before the net) |
| `learning_rate`     | 0.001   | SGD learning rate, momentum 0.9                                          |
| `scheduler`         | cosine  | `cosine` decays to 0 over all batches, `none` keeps it constant          |
| `warmup_steps`      | 0       | batches of linear learning-rate ramp-up                                  |
| `epoch_fraction`    | 1.0     | train on a fresh random fraction of the chips each epoch                 |
| `freeze_encoder`    | true    | train only the FPN and heads                                             |
| `unfreeze_from`     | none    | with a frozen encoder, still train the ResNet stages from this one on    |
| `encoder_lr_factor` | 1.0     | learning-rate multiplier for the encoder weights that train              |
| `max_grad_norm`     | 1.0     | gradient clipping norm, 0 disables it                                    |
| `val_ratio`         | 0.2     | share of the chips held out as spatial validation                        |
| `split_seed`        | 42      | seed of the block order in the split                                     |
| `block_size`        | 4       | side of a spatial block, in tile units                                   |

| Inference              | Default | Meaning                                                   |
| ---------------------- | ------- | --------------------------------------------------------- |
| `confidence_threshold` | 0.10    | minimum score of a kept box                               |
| `nms_threshold`        | 0.05    | IoU above which overlapping boxes of one chip are removed |
| `model_gsd_m`          | 0.05    | metres per pixel the network sees                         |

## Limitations

- Trained on Freetown imagery at zoom 20. Zoom 19 imagery is upscaled about 6x and zoom 18 about 11x (accepted, but far outside what was
  measured; the 1024 px cap then limits the upscaling of a 256 px chip to 4x, so the model sees coarser pixels than it was trained on).
  Zoom 17 or lower is rejected. Other cities, sensors and seasons are unmeasured.
- Forgetting: on a NEON image the model scores 0.41 F1 against 0.69 for the original checkpoint, so it trades NEON-style performance for
  OAM-style performance.
- Precision is low (0.40 at IoU 0.4 on the Freetown test split): expect many false boxes, especially in dense or shadowed canopy. The
  defaults (confidence 0.10, NMS 0.05) were chosen by sweeping both on the validation split; raising the confidence trades recall for
  precision, and a higher NMS lowered F1 at every value from 0.10 to 0.50.
- Crowns are boxes, not outlines, and touching crowns in dense forest are often merged or missed.
- A crown cut by a chip border is predicted as two boxes, one per chip; the chips are not merged.
- Training at about 3x the chip size is slow on CPU. A GPU is requested by the STAC item; CI sets `FAIR_FORCE_CPU=1`.

## Usage

The model follows the standard fAIr flow described in the contributing guide: register the base model, register a dataset with tree labels,
finetune, promote and predict. The STAC item declares every training and inference hyperparameter with its default and range.

The container images install the library from a release tag (`TD_HOT_REF` in the `Dockerfile`, set once per image stage); update it whenever
the library changes.

## Citation

Weinstein, B.G. et al. (2020). DeepForest: A Python package for RGB deep learning tree crown delineation. Methods in Ecology and
Evolution. https://doi.org/10.1111/2041-210X.13472

Weinstein, B.G. et al. (2019). Individual Tree-Crown Detection in RGB Imagery Using Semi-Supervised Deep Learning Neural Networks.
Remote Sensing 11(11), 1309. https://doi.org/10.3390/rs11111309

## License

MIT
