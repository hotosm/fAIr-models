# Solar PV Panels

A U-Net that labels each pixel of an RGB aerial chip as solar panel or background,
then traces the panel pixels into polygons. It serves live and batch predictions,
and it can be finetuned in fAIr on panels labelled in a new area, which is how it
is meant to be adapted outside its training region.

## Task

Binary semantic segmentation of solar photovoltaic (PV) panels, on rooftops and on
the ground. Each 256 x 256 chip goes through the network in one pass, the panel
probability is thresholded, and the mask is vectorised into GeoJSON polygons.

## Inputs and outputs

| Stage      | Shape                          | Notes                                    |
| ---------- | ------------------------------ | ---------------------------------------- |
| Input      | `[batch, 3, 256, 256]` float32 | RGB values divided by 255                |
| Output     | `[batch, 1, 256, 256]` float32 | One solar panel logit per pixel          |
| Prediction | GeoJSON                        | Solar panel polygons in EPSG:4326        |

The ImageNet mean and standard deviation are applied inside the ONNX graph.
`confidence_threshold` (default 0.5) is the minimum panel probability, the sigmoid
of the logit, for a pixel to count as panel.

## Model

U-Net from `segmentation_models_pytorch` 0.5.0 with a ResNet-34 encoder initialised
from ImageNet weights (24.4 M parameters). Trained with binary cross-entropy plus
Dice loss in PyTorch 2.11, on chips randomly rotated, flipped and changed in
brightness, colour, haze, blur and JPEG quality, then exported to ONNX opset 18.

## Training data

[Jiang et al. (2021)](https://doi.org/10.5194/essd-13-5389-2021), a multi-resolution
PV panel dataset from Jiangsu, China
([doi:10.5281/zenodo.5171712](https://doi.org/10.5281/zenodo.5171712), CC-BY-4.0):

- PV01: 645 rooftop chips from UAV orthophotos at 0.1 m.
- PV03: 2,308 chips from aerial photography at 0.3 m, of which 236 are rooftop and
  the rest ground-mounted arrays.

PV01 chips are tiled into mosaics and PV03 chips are cropped, so that training
images span 0.15 to 0.32 m per pixel, the range of OpenAerialMap tiles at zooms 19
and 20. Train and validation sets are split by spatial blocks, so no validation
chip overlaps a training area.

## Evaluation

Validation on held-out blocks of the training region (threshold 0.5), at zoom 19
(about 0.3 m per pixel) and, for rooftops, zoom 20 (about 0.15 m per pixel):

| Source       | Zoom 19 IoU | Zoom 19 F1 | Zoom 20 IoU | Zoom 20 F1 |
| ------------ | ----------- | ---------- | ----------- | ---------- |
| PV01 rooftop | 0.797       | 0.887      | 0.870       | 0.930      |
| PV03 rooftop | 0.841       | 0.913      | 0.747       | 0.855      |
| PV03 ground  | 0.912       | 0.954      |             |            |

PV03 zoom 20 chips are 0.3 m imagery enlarged two times, so they are blurrier than
real zoom 20 tiles.

Zero-shot test in Kaduna, Nigeria, against outlines of 18 rooftop panel arrays
(287 m2 of panels) drawn on [OpenAerialMap imagery](https://map.openaerialmap.org/)
from 2025 and checked at zoom 21, threshold 0.5:

| Zoom | IoU   | Precision | Recall | Arrays found | False alarms |
| ---- | ----- | --------- | ------ | ------------ | ------------ |
| 19   | 0.179 | 0.599     | 0.204  | 7 of 18      | 5            |
| 20   | 0.307 | 0.787     | 0.335  | 10 of 18     | 0            |

An array counts as found when the model marks any of its pixels, and a false alarm
is a predicted shape of at least 1.5 m2 that touches no panel outline. At zoom 20
the model draws no false alarms but outlines only a third of the panel area: it has
seen no African imagery yet. At zoom 19 the false alarms are grey-blue painted
metal roofs. Lowering `confidence_threshold` to 0.3 finds 12 of 18 arrays at zoom
20, with one false alarm.

![Image, label and prediction for two held-out chips from Jiangsu and two Kaduna roofs at zoom 20](https://huggingface.co/dexsholex/unet-solar-pv/resolve/main/unet_solar_pv_examples.png)

Top two rows: held-out validation chips from Jiangsu. Bottom two rows: Kaduna at
zoom 20. On the first roof the model outlines most of the array; on the second it
misses the top array and only partly outlines the bottom one. Label yellow,
prediction blue, threshold 0.5. Imagery: Jiang et al. (2021), CC-BY-4.0, and
OpenAerialMap (shamaki, 2025-05-29), CC-BY 4.0.

## Imagery

RGB aerial or drone imagery at zoom 19 or 20 (about 0.15 to 0.3 m per pixel). The
STAC item recommends zoom 20 and previews Kaduna, Nigeria: in the Kaduna test the
model scores much higher at zoom 20 than at zoom 19, where grey-blue painted metal
roofs can be mistaken for panels.

## Usage

Pick Solar PV Panels in fAIr, draw an area over true-colour imagery at zoom 20 (or
19) and run a prediction: panels come back as polygons labelled `solar_panel`.
Raise `confidence_threshold` to keep only the surest panels, or lower it towards
0.3 to find more arrays at the cost of a few false alarms. Batch predictions run
the same code through `inference_pipeline`. To adapt the model to a new city, label
the panels on some chips there and finetune it.

## Fine-tuning

Fine-tuning starts from the released weights (the `checkpoint` asset, a PyTorch
`state_dict`) and trains the whole network on the labelled chips with the same
binary cross-entropy plus Dice loss, randomly turning and flipping each chip. The
defaults are 15 epochs, batch size 8 and an AdamW learning rate of 0.0001 that
decays to zero on a cosine schedule. With only a handful of labelled chips,
`freeze_encoder` keeps the ImageNet and solar features of the encoder and trains
the decoder alone.

The split is spatial: fAIr names chips after their map tiles, so neighbouring chips
are grouped into 4 x 4 tile blocks and whole blocks go to validation. Neighbouring
chips often share a roof or an array, and a random split would let the model be
scored on panels it trained on. Validation reports solar panel pixel IoU, precision,
recall and F1 at the confidence threshold, and the finetuned model is exported to
ONNX with the same input and output as the base model.

## Limitations

- The base model was trained only on imagery from eastern China. Accuracy drops on
  African cities, as the Kaduna test shows, most of all for small rooftop arrays;
  finetune on local labels before relying on it there.
- Corrugated or painted metal roofs and tents in dense informal settlements can be
  mistaken for panels.
- Expects true-colour RGB. Thermal or false-colour imagery gives meaningless results.
- Polygons are traced per chip, so an array that crosses a chip edge comes back as
  two polygons. Chips must be 256 x 256 pixels.

## Citation

The training data: Jiang, H., Yao, L., Lu, N., Qin, J., Liu, T., Liu, Y. and Zhou,
C. (2021). Multi-resolution dataset for photovoltaic panel segmentation from
satellite and aerial imagery. Earth System Science Data, 13(11), 5389-5401.
https://doi.org/10.5194/essd-13-5389-2021

The model: Oyinloye, O. (2026). unet-solar-pv: solar PV panel segmentation for
fAIr. https://huggingface.co/dexsholex/unet-solar-pv

## Licence

The model weights and code are released under Apache-2.0. The training data is
CC-BY-4.0 by Jiang et al. (2021).
