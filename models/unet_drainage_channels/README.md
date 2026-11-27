# Drainage Channels

A U-Net that finds open drainage channels (drains and ditches) in RGB drone imagery
and returns their centrelines as lines, the way OpenStreetMap maps them
(`waterway=drain`, `waterway=ditch`). Drains decide where rainwater goes in a dense
city, and a drain missing from the map cannot be planned for or cleared before the
rains. The model gives mappers a first draft to check instead of a blank map. It
serves live and batch predictions, and it can be finetuned in fAIr on drains mapped
in a new town.

## Task

Semantic segmentation into background (0) and open drain (1), followed by line
tracing. Each 256 x 256 chip goes through the network in one pass, pixels above the
confidence threshold form a drain mask, and the mask is thinned to centrelines that
come back as GeoJSON lines. Culverts and covered drains cannot be seen from above and
are out of scope.

## Inputs and outputs

| Stage      | Shape                          | Notes                                                     |
| ---------- | ------------------------------ | --------------------------------------------------------- |
| Input      | `[batch, 3, 256, 256]` float32 | RGB values divided by 255                                 |
| Output     | `[batch, 1, 256, 256]` float32 | One drain logit per pixel                                 |
| Prediction | GeoJSON                        | Lines in EPSG:4326 with `class`, `label` and `score`      |

The ImageNet mean and standard deviation are applied inside the ONNX graph. A pixel is
drain when the sigmoid of its logit is above `confidence_threshold` (default 0.3). The
mask then loses its pieces smaller than 1 m2 and its pixels within 2 m of the no-data
edge of the image, is thinned to lines one pixel wide, and the lines are joined
between ends and junctions and simplified to within 0.3 m. A line with a loose end
that is shorter than `min_length_m` (default 2) is dropped. Each line carries `class`
1, `label` `drain` and `score`, the mean drain probability along it.

## Model

U-Net from `segmentation_models_pytorch` 0.5.0 with a ResNet-34 encoder initialised
from ImageNet weights (24.4 M parameters), trained in PyTorch 2.11 on a Kaggle T4 GPU
and exported to ONNX opset 18. The loss is binary cross-entropy plus Dice. A drain
pixel counts 6.15 times a background pixel in the cross-entropy: the square root of
background share over drain share, capped at 10.

Drains in OpenStreetMap are lines, often mapped a metre or more off the channel seen
in the imagery, and two rules deal with that. The first is a margin: each drain is
drawn at its `width` tag (1 m without one), and 1 m on each side of it counts in no
loss. The second is a slide step. After two epochs on the labels as given, every
training step first moves each chip's label to the position where it disagrees least
with the model's own prediction, among 169 positions up to 1.78 m east or west and
north or south, and trains against that. A small cost per pixel of shift keeps a label
in place when nothing is gained. In the last epoch the labels moved 1.65 m on average,
and about 60% of chips chose a position at the edge of the window. So the step
tolerates misplaced labels; it does not measure how far they are off.

Each training chip was randomly turned, flipped and changed in brightness, contrast,
colour, saturation, blur and JPEG quality. Training ran for 16 epochs with AdamW
(learning rate 0.0003, one-cycle schedule, batch size 16). The released weights are
those of the last epoch, which also scored best on validation.

## Training data

Imagery: the 20 drone images of Dar es Salaam, Tanzania, on
[OpenAerialMap](https://map.openaerialmap.org/) whose licence allows it (CC BY 4.0),
read as fAIr's own zoom 20 tiles:

| OpenAerialMap images                    | Provider                                                     | Date           | Pixel size    |
| --------------------------------------- | ------------------------------------------------------------ | -------------- | ------------- |
| 15 Ramani Huria flights                 | Tanzania Open Data Initiative, via the Open Imagery Network  | Apr to May 2015 | 4.1 to 7.2 cm |
| University of Dar es Salaam - CoICT     | WeRobotics                                                   | Jul 2016       | 2.9 cm        |
| Dar selected subwards                   | Tanzania Open Data Initiative                                | Mar 2020       | 6.1 cm        |
| Msimbazi River Delta                    | OMDTZ & HeiGIT gGmbH                                         | Oct 2023       | 5.0 cm        |
| Mabibo Mburahati 2024 and Barafu 2024   | OMDTZ                                                        | Oct, Nov 2024  | 2.1, 5.1 cm   |

Labels: OpenStreetMap `waterway=drain` and `waterway=ditch` lines (© OpenStreetMap
contributors, ODbL), taken on 6 November 2026: 9,689 drains and 807 ditches in the
city, most of them carrying the width and depth measured on the ground by
[Ramani Huria](https://wiki.openstreetmap.org/wiki/Dar_es_Salaam/Ramani_Huria) from
2017 to 2019. Under the imagery lie 2,008 open drains and ditches (158 km); 81% of
that length has a measured width. Culverts and covered drains are in the labels but
count in no loss.

A tile is used only if it holds imagery and lies within 15 m of a mapped open drain,
because a drain missing from OpenStreetMap would otherwise be taught as background.
One tile can be read from two images. In three of the 2015 images the mapped lines sat
several metres off the channels across the whole image, so their lines were moved by
one fixed offset per image before training (0.6 m east and 4.0 m south, 1.6 m west and
2.2 m south, 1.3 m east and 5.9 m south). The offsets came from comparing a first
model's predictions with the lines.

This gives 11,618 chips, split by blocks of 8 x 8 tiles (about 300 m): 9,274 chips for
training and 2,344 from held-out blocks for validation, which hold about 20% of the
drain length in every width class.

## Evaluation

The scores are on the lines the model returns, at the default threshold of 0.3, from
this pipeline's `evaluate_model` code run over the chips as GeoTIFFs. Completeness is
the share of mapped open-drain length that has a predicted line within the tolerance.
Correctness is the share of predicted line length that has a mapped open drain within
the tolerance; predicted line over a covered drain or at the edge of the image counts
as neither right nor wrong.

Accra, Ghana, is a city the model never trained on: 1,565 chips of Open Cities Africa
and OpenStreetMap Ghana drone imagery over drains mapped mostly by the Open Cities
Accra surveys.

| Test                                   | Tolerance | Completeness | Correctness | F1    |
| -------------------------------------- | --------- | ------------ | ----------- | ----- |
| Dar es Salaam, held-out blocks         | 1 m       | 0.245        | 0.228       | 0.236 |
| Dar es Salaam, held-out blocks         | 2 m       | 0.416        | 0.378       | 0.396 |
| Dar es Salaam, held-out blocks         | 4 m       | 0.634        | 0.570       | 0.600 |
| Accra, unseen city                     | 1 m       | 0.234        | 0.274       | 0.253 |
| Accra, unseen city                     | 2 m       | 0.400        | 0.456       | 0.426 |
| Accra, unseen city                     | 4 m       | 0.523        | 0.579       | 0.550 |

Read these scores as agreement with OpenStreetMap, which is not the same as agreement
with the ground. In the first two rows of the picture below, the model's lines follow
the channels in the image while the mapped lines run beside them, 1 to 3 m away. A
predicted line on the channel then fails the 1 m and often the 2 m test although a
mapper would accept it. That is why the scores rise steadily with the tolerance, and
why 4 m is reported as well. The model has no test set traced on the imagery yet, so
how much of the gap is label offset and how much is model error is not measured.

The scores also vary by place. Over the five drone images of Accra, F1 within 2 m
runs from 0.14 to 0.61, and over the 17 images that reach the Dar es Salaam validation
blocks from 0.17 to 0.48. Dar es Salaam is scored against the lines as mapped; with
the three shifted images' lines moved as in training, its F1 within 2 m is 0.423.
Pixel IoU of the drain mask is 0.107 in Dar es Salaam and 0.127 in Accra, low because
a thin feature half a metre off scores nothing per pixel.

What each training step added was measured in the pilot on Accra, with an earlier
score on the predicted mask (F1 of mapped length found within 1 m of the mask and of
mask pixels within 1 m of a mapped drain, threshold 0.3). It is not comparable with
the table above:

| Training                                                  | F1    |
| --------------------------------------------------------- | ----- |
| Labels as mapped, margin only                             | 0.313 |
| With the three images' lines moved                        | 0.360 |
| With the slide step as well (the released model)          | 0.385 |

Each row is one training run, so differences of about 0.02 are within what a rerun
could change. An earlier run with the slide step scored 0.376.

![Imagery, and the same with OpenStreetMap drains in yellow and predicted lines in cyan, for Dar es Salaam, Accra and Kaduna](https://huggingface.co/dexsholex/unet-drainage-channels/resolve/main/unet_drainage_channels_examples.jpg)

Imagery (left) and the same with OpenStreetMap's open drains in yellow and the model's
lines in cyan (right). Top: held-out blocks in Dar es Salaam; the black square is a
tile that is not in the validation set. Middle: Accra, where the model finds the
drains along the road and misses most of those between houses. Bottom: Kaduna,
Nigeria, where there are no labels to score against and the lines were checked by eye
only. Imagery: OMDTZ & HeiGIT gGmbH (top, CC BY 4.0), GFDRR
Open Cities Africa (middle, CC BY-SA 4.0) and shamaki (bottom, CC BY 4.0), via
OpenAerialMap. Drains: © OpenStreetMap contributors, ODbL.

## Related work

Completeness and correctness within a buffer are the usual scores for extracted road
centrelines (Heipke et al., 1997). Mnih and Hinton (2012) trained aerial image
labellers on maps that were misregistered against the imagery by letting the loss
treat the true label as a shifted copy of the mapped one; the slide step here is a
simpler version of that idea, which picks one best shift per chip.

## Imagery

RGB drone imagery at zoom 20 (about 0.15 m per pixel), where a 1 m drain is about 7
pixels wide. The model was trained at zoom 20 only. At zoom 19 on the same Accra
drains it does much worse (F1 within 2 m 0.257 against 0.426, F1 within 4 m 0.409
against 0.550), so the STAC item offers zoom 20 only. It previews an area of the
Msimbazi River Delta image in Dar es Salaam.

## Usage

Pick Drainage Channels in fAIr, draw an area over true-colour drone imagery at zoom 20
and run a prediction: drains come back as lines with `class` 1, `label` `drain` and a
`score` from 0 to 1. Raise `confidence_threshold` for fewer, surer lines, or
`min_length_m` to drop short ones. Treat every line as a draft: check it against the
imagery, join the pieces of one drain, and tag it `waterway=drain` or
`waterway=ditch` before it goes to OpenStreetMap. Batch predictions run the same code
through `inference_pipeline`.

## Fine-tuning

Fine-tuning starts from the released weights (the `checkpoint` asset, a PyTorch
`state_dict`) and trains the whole network on the labelled chips with the same loss,
margin and slide step, randomly turning, flipping and changing the brightness,
contrast, colour and saturation of each chip. The defaults are 15 epochs, batch size 8
and an AdamW learning rate of 0.0001 that decays to zero on a cosine schedule, with
the labels sliding from the second epoch. With only a handful of labelled chips,
`freeze_encoder` trains the decoder alone.

Labels are lines. A label's `width` tag, in metres, sets how wide it is drawn (1 m
without one). A label tagged `tunnel=culvert`, `tunnel=yes`, `covered=yes` or with a
negative `layer` is treated as covered: it is left out of the loss and the scores, so
the model is not taught that a hidden drain is background. A label whose `waterway`
is something other than `drain` or `ditch` is left out. Tags are read from the label's
properties or from `tags`, as fAIr downloads them from OpenStreetMap.

`slide_tolerance_m` (default 2) is how far a label may slide. Set it to 0 when the
labels were traced on the same imagery and already sit on the channels.

The split is spatial: fAIr names chips after their map tiles, so neighbouring chips are
grouped into 8 x 8 tile blocks and whole blocks go to validation. Of 200 seeded
shuffles of the blocks, the one that holds out the share of drain length closest to the
validation ratio is kept. Validation reports completeness, correctness and F1 of the
predicted lines within 1, 2 and 4 m, and pixel IoU, precision and recall of the drain
mask, and the finetuned model is exported to ONNX with the same input and output as
the base model.

## Limitations

- The scores measure agreement with OpenStreetMap lines that are themselves 1 to 3 m
  off the channels. There is no test set traced on the imagery yet.
- Within 2 m, the model finds about 40% of the mapped drain length, and 38% to 46% of
  what it draws is on a mapped drain. In the example images it misses narrow
  drains between houses, often hidden under roof edges, and wide channels covered
  with slabs. It draws false lines along kerbs, central reservations, walls and the
  edges of car parks. Some of its lines off the map may be real drains that are not
  mapped; that was not checked.
- Trained on one city and scored on one other. Results in Accra differ a lot between
  neighbourhoods; finetune on local labels before relying on it elsewhere.
- Zoom 20 only. Chips must be 256 x 256 pixels.
- Lines are traced chip by chip, so a drain that crosses several chips comes back as
  one line per chip, with small gaps or steps at the chip edges. A wide channel comes
  back as a single centreline, without its width.
- The released weights come from a single training run, and the threshold was chosen
  on the validation blocks, which can make the Dar es Salaam scores slightly
  optimistic.
- Expects true-colour RGB. Thermal or false-colour imagery gives meaningless results.

## Citation

The model: Oyinloye, O. (2026). unet-drainage-channels: open drainage channel mapping
for fAIr. https://huggingface.co/dexsholex/unet-drainage-channels

Heipke, C., Mayer, H., Wiedemann, C. and Jamet, O. (1997). Evaluation of automatic
road extraction. International Archives of Photogrammetry and Remote Sensing, 32
(3-4W2).

Mnih, V. and Hinton, G. E. (2012). Learning to label aerial images from noisy data.
Proceedings of the 29th International Conference on Machine Learning (ICML).

## Licence

The model weights are released under CC-BY-4.0. They were trained on imagery published
under CC BY 4.0 by the Tanzania Open Data Initiative, WeRobotics, OMDTZ and HeiGIT
gGmbH through OpenAerialMap, and on drains mapped by OpenStreetMap contributors
(ODbL), most of them surveyed by Ramani Huria. The Accra imagery (CC BY-SA 4.0) was
used for testing only.
