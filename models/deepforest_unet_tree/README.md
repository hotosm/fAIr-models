# deepforest-unet-tree

Segments tree canopy in very high resolution RGB imagery and returns one polygon per canopy patch. The network is a U-Net whose
encoder is the ResNet-50 of the DeepForest NEON_OAM detector, so it starts from what that detector learned about trees in aerial
images. Only the decoder is new. The model can be used once it has been trained on a dataset of OAM chips with tree labels; the base
checkpoint on its own holds the encoder and no decoder, so it cannot predict.

## Status

The architecture, the training and export steps and the inference path are implemented and tested offline. No decoder has been trained
and measured on a fAIr dataset yet, so no accuracy is claimed here. The first measurement will be the pixel scores of the validation
split that the training step reports. The pack is not ready to be promoted until a trained checkpoint and its scores exist.

## How it works

A chip is scaled to the model's resolution from its OAM file name: the zoom and the tile row give the ground size of a pixel, and the
chip is resized so that one model pixel covers 0.05 m of ground (`model_gsd_m`). A zoom 20 chip at Freetown (about 15 cm per pixel) is
upscaled about 3x; a chip is capped at 1024 px on its longest side. The ONNX graph takes the scaled chip as `image` (float32, 3 x H x W, RGB
in [0, 1]) and returns `probability`, the per-pixel probability of tree canopy. Thresholding, tracing the canopy outlines, dropping small
polygons, simplifying and georeferencing to EPSG:4326 happen in the pinned `ts-hot` library, outside the graph, so they stay tunable.

## Inputs and outputs

- **Input:** RGB chips named `OAM-<x>-<y>-<z>.<ext>`. The name is required: it sets the scale. A zoom that would need an upscale above 16x
  is rejected.
- **Output:** one polygon per canopy patch in EPSG:4326, with holes kept. Its properties are `class` (always 1), `label` (`tree`),
  `confidence` (the mean probability inside the polygon) and `source` (the chip file name).
- **Training labels:** a GeoJSON of polygons (tree canopy outlines) and tree points. A point with a `diameter_crown` tag (in metres, as `4`,
  `4 m` or `3,5 m`) becomes a disc of that diameter; a point without a usable diameter is ignored in the loss and the scores, so it is
  neither a positive nor a negative example.

## Training

Training starts from the base checkpoint, which is either a U-Net state dict (to continue from a trained model) or the DeepForest
NEON_OAM state dict (to start from its encoder). The step tells the two apart from the weights themselves. The loss is binary
cross-entropy plus a Dice term, and the optimiser is AdamW with a cosine schedule. The encoder is frozen by default, and its BatchNorm
statistics are always frozen, which limits the forgetting of what NEON_OAM learned. Each epoch the best checkpoint by validation loss is
kept.

The chips are split in spatial blocks on OAM tile coordinates, so the chips of one block never fall on both sides of the split. Chips without a
label are kept as background examples. The step rebuilds the labels from the chips and the GeoJSON it was given, and nothing is shared
between steps.

Evaluation runs on the validation split through the exported ONNX file, with the inference threshold. It reports pixel precision, recall,
F1 and IoU of the tree class.

## Pretrained source

The base checkpoint is `NEON_OAM`, hosted at [AbdelrahmanU3/NEON_OAM](https://huggingface.co/AbdelrahmanU3/NEON_OAM). It is the
[`weecology/deepforest-tree`](https://huggingface.co/weecology/deepforest-tree) detector (DeepForest, MIT) fine-tuned on Freetown OAM chips
by the `deepforest-tree` pack. The weights are MIT licensed, and this model is MIT licensed to match.

## Limitations

- The decoder is untrained in the base checkpoint, so there are no predictions and no measured accuracy yet.
- Trained on Freetown imagery at zoom 20 only (for the encoder). Other cities, sensors, seasons and zooms are unmeasured.
- A canopy patch cut by a chip border is returned as two polygons, one per chip; chips are not merged.
- Labels from OSM are incomplete: trees nobody mapped count as false positives when the model finds them.
- Pixel scores measure the canopy area, not the crowns. A polygon that is a good crown can still score poorly at the pixel level.

## Hyperparameters

Training and inference share the scale: `model_gsd_m` and `max_input_px` are declared for both, and they must stay equal, because the
network sees the scale it was trained at. The other keys are declared in the STAC item with their defaults and ranges.
