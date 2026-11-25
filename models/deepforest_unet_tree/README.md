# deepforest-unet-tree

Segments tree canopy in very high resolution RGB imagery and returns one polygon per canopy patch. The network is a U-Net whose
encoder is the ResNet-50 of the DeepForest NEON_OAM detector, so it starts from what that detector learned about trees in aerial
images. The decoder was trained on the Guru trees dataset (OAM chips of Freetown with OSM tree labels), and the last encoder stage
(`layer4`) was fine-tuned with it.

## Status

The base model of this pack is the trained U-Net published at
[AbdelrahmanU3/deepforest_UNet_OAM](https://huggingface.co/AbdelrahmanU3/deepforest_UNet_OAM), as weights and as ONNX. It scores a pixel IoU
of 0.570 (F1 0.726) on the 4,158 validation chips of the Guru dataset and 0.404 (F1 0.576) on its 5,627 test chips, at the default
threshold 0.3, from one training run. The test score is lower mostly because about a quarter of the test chips have no OSM labels although many
show canopy, so the model's correct detections there count as errors; recall is high on both splits (0.76 and 0.83). The model card and
`docs/Theory.md` of `ts-hot` give the experiments and the caveats. The pack's step tests pass offline; the platform run (Docker, ZenML, Kind) has
not been done for this version.

## How it works

`predict` is two steps, both reading the same `inference.*` parameters: inference (`ts_hot.predict_session`, the first two stages below), then `postprocess` (the function the STAC item names; stages 3 and 4). Live serving and the batch `run_inference` step both call `predict`.

1. Inference, `preprocess`: scales the chip to the model's resolution from its OAM file name: the zoom and the tile row give the ground size of a
   pixel, and the chip is resized so that one model pixel covers 0.05 m of ground (`model_gsd_m`). A zoom 20 chip at Freetown (about 15 cm
   per pixel) is upscaled about 3x; a chip is capped at `max_input_px` (1024) on its longest side.
2. Inference, the ONNX graph: takes the scaled chip as `image` (float32, 3 x H x W, RGB in [0, 1]) and returns `probability`, the per-pixel probability
   of tree canopy.
3. `postprocess` turns the probability into polygons: optional smoothing, the threshold, optional opening and closing, tracing the canopy
   outlines, dropping small polygons, optional hole filling, and simplifying.
4. The polygons are mapped back to chip pixels and georeferenced to EPSG:4326.

Thresholding and the polygon steps run outside the ONNX graph, so they stay tunable per request. All of the logic is in the pinned `ts-hot`
library; the pack only calls it in order and names the output properties.

## Inputs and outputs

- **Input:** RGB chips named `OAM-<x>-<y>-<z>.<ext>`. The name is required: it sets the scale. A zoom that would need an upscale above 16x
  is rejected.
- **Output:** one polygon per canopy patch in EPSG:4326, with holes kept. Its properties are `class` (always 1), `label` (`tree`),
  `confidence` (the mean probability inside the polygon) and `source` (the chip file name).
- **Training labels:** a GeoJSON of polygons (tree canopy outlines) and tree points. A point with a `diameter_crown` tag (in metres, as `4`,
  `4 m` or `3,5 m`) becomes a disc of that diameter; a point without a usable diameter is ignored in the loss and the scores, so it is
  neither a positive nor a negative example.

## Training

Training starts from the base checkpoint, which is either a U-Net state dict (to continue from a trained model, the default now) or the
DeepForest NEON_OAM state dict (to start from its encoder). The step tells the two apart from the weights themselves. The loss is binary
cross-entropy plus a Dice term, and the optimiser is AdamW with a cosine schedule.

The defaults are the best run on the Guru trees: the encoder is frozen except `layer4`, which trains at a tenth of the learning rate (the
BatchNorm statistics are always frozen, which limits forgetting); ten epochs that each use a random tenth of the chips, at batch size 4. Each
epoch the best checkpoint by validation loss is kept. No augmentation is on by default: a quarter turn plus mirror, a mirror only, brightness /
contrast / gamma jitter, scale jitter and blur or JPEG are all declared and can be switched on, but none of them changed the Guru score by more
than 0.003 IoU. The ranges in the STAC item are narrow on purpose: they cover what was tried or measured (for example batch 2 used about 1.6 GB
and batch 4 about 3.3 GB of GPU memory on the Guru chips), because GPU resources are limited.

The chips are split in spatial blocks on OAM tile coordinates, so the chips of one block never fall on both sides of the split. Chips without a
label are kept as background examples. The step rebuilds the labels from the chips and the GeoJSON it was given, and nothing is shared
between steps.

Evaluation runs on the validation split through the exported ONNX file, with the inference threshold. It reports pixel precision, recall,
F1 and IoU of the tree class.

## Pretrained source

The base checkpoint and the ONNX are the trained U-Net `NEON_OAM_UNET` at
[AbdelrahmanU3/deepforest_UNet_OAM](https://huggingface.co/AbdelrahmanU3/deepforest_UNet_OAM), pinned by commit in the STAC item. Its encoder
comes from `NEON_OAM` ([AbdelrahmanU3/NEON_OAM](https://huggingface.co/AbdelrahmanU3/NEON_OAM)), the
[`weecology/deepforest-tree`](https://huggingface.co/weecology/deepforest-tree) detector (DeepForest, MIT) fine-tuned on Freetown OAM chips by
the `deepforest-tree` pack; a `NEON_OAM` state dict can still be given as the base to start a new decoder. The DeepForest weights are MIT
licensed, and this model is MIT licensed to match. The licence of the Guru dataset has not been checked here.

## Limitations

- Trained and measured on Freetown imagery at zoom 20 only, from one training run. Other cities, sensors, seasons and zooms are unmeasured.
- The scores are pixel scores against incomplete OSM labels: the test split has many chips with no labels (so its precision is low) and the
  validation split is the most completely mapped (so its scores are the optimistic ones). Neither is a clean accuracy.
- Polygons are not tested for self-intersection, and no object-level (per crown) score has been measured.
- A canopy patch cut by a chip border is returned as two polygons, one per chip; chips are not merged.
- Labels from OSM are incomplete: trees nobody mapped count as false positives when the model finds them.
- Pixel scores measure the canopy area, not the crowns. A polygon that is a good crown can still score poorly at the pixel level.

## Hyperparameters

Training and inference share the scale: `model_gsd_m` and `max_input_px` are declared for both, and they must stay equal, because the
network sees the scale it was trained at. The other keys are declared in the STAC item with their defaults and ranges.
