# Roof Material

A U-Net that labels each pixel of an RGB aerial or drone chip as metal sheet roof,
thatch roof, other roof or background, then traces the roof pixels into polygons. Roof
material is a sign of housing quality: Guthula et al. (2024) built a roof dataset of
the same city to support malaria risk assessment, because some roof types are linked
to higher risk. The model serves live and batch predictions, and it can be finetuned
in fAIr on roofs labelled in a new area.

## Task

Semantic segmentation into four classes: background (0), metal sheet (1), thatch (2)
and other roof (3), which in the training data means asbestos cement and concrete.
Each 256 x 256 chip goes through the network in one pass, every pixel takes its most
likely class, and the roof pixels of each class are vectorised into GeoJSON polygons.

## Inputs and outputs

| Stage      | Shape                          | Notes                                              |
| ---------- | ------------------------------ | -------------------------------------------------- |
| Input      | `[batch, 3, 256, 256]` float32 | RGB values divided by 255                          |
| Output     | `[batch, 4, 256, 256]` float32 | One logit per class per pixel                      |
| Prediction | GeoJSON                        | Roof polygons in EPSG:4326 with `class` and `label` |

The ImageNet mean and standard deviation are applied inside the ONNX graph.
`confidence_threshold` (default 0) is the minimum probability of a pixel's most likely
roof class, the softmax of the logits, for the pixel to stay a roof; at 0 every pixel
simply takes its most likely class.

## Model

U-Net from `segmentation_models_pytorch` 0.5.0 with a ResNet-34 encoder initialised
from ImageNet weights (24.4 M parameters), trained in PyTorch 2.11 on a Kaggle T4 GPU
and exported to ONNX opset 18. The loss is cross-entropy plus Dice. Cross-entropy
weighs each class by the square root of background share over class share, capped at
5 (background 1.0, metal sheet 2.52, thatch 3.85, other roof 5.0), and Dice is taken
over the three roof classes only. Chips that contain other roofs were drawn three
times as often. Each training chip was randomly turned, flipped and changed in
brightness, contrast, colour, saturation, blur and JPEG quality. Training ran for 30
epochs with AdamW (learning rate 0.0003, one-cycle schedule, batch size 16), and the
released weights are those of epoch 24, the best mean roof IoU on validation.

## Training data

Imagery: four #MapeandoMeuBairro drone images of Nacala, Mozambique, on
[OpenAerialMap](https://map.openaerialmap.org/) (CC BY-SA 4.0), read as fAIr's own
zoom 19 and zoom 20 tiles:

| OpenAerialMap image          | Date     | Pixel size |
| ---------------------------- | -------- | ---------- |
| `61b0cf79...` Ribáuè         | Oct 2021 | 4.1 cm     |
| `62471237...` Ontupaia       | Mar 2022 | 4.9 cm     |
| `62471330...` Ontupaia       | Mar 2022 | 4.9 cm     |
| `62470f31...` Ribáuè-Mocone  | Mar 2022 | 4.9 cm     |

Labels: OpenStreetMap buildings and their `roof:material` tags from the 2021
participatory mapping of the same neighbourhoods (© OpenStreetMap contributors,
ODbL), taken on 3 November 2026. Of 21,074 buildings in the area, 16,502 carry a tag
the model learns: 10,330 metal sheet, 5,861 thatch and 311 other roof (238 asbestos,
73 concrete). Buildings without a tag, or with a rare one (coconut leaves, roof
tiles), are left out of the loss, so the model is never taught that a house of
unknown material is background. A tile is used only if it lies inside the drone image
and within 30 m of tagged buildings, because a house missing from OpenStreetMap would
otherwise be taught as background.

This gives 4,252 chips. They are split by blocks of 8 x 8 zoom 20 tiles (about 300 m),
with both zooms of a place on the same side, and of 200 shuffles the one that holds
out closest to 20% of every roof class is used: 3,362 chips for training and 890 for
validation.

## Evaluation

Validation on the 890 held-out chips, plain argmax (threshold 0). Pixels of buildings
without a usable tag are left out. Running the released weights over the same chips,
as GeoTIFFs, through this pipeline's `evaluate_model` gives the same pixel scores.

| Zoom | Class       | IoU   | Precision | Recall |
| ---- | ----------- | ----- | --------- | ------ |
| 20   | Metal sheet | 0.643 | 0.726     | 0.849  |
| 20   | Thatch      | 0.640 | 0.707     | 0.870  |
| 20   | Other roof  | 0.395 | 0.717     | 0.468  |
| 19   | Metal sheet | 0.611 | 0.708     | 0.816  |
| 19   | Thatch      | 0.596 | 0.691     | 0.812  |
| 19   | Other roof  | 0.424 | 0.690     | 0.523  |

Pixel IoU is held down partly because roof outlines in OpenStreetMap do not always sit
exactly on the roof edges in the drone image. A per-roof check is closer to what a
mapper needs: each labelled roof takes the class that most of its pixels were given,
and it counts as missed when that class is background (roof pieces under 4 m2, cut off
by a chip edge, are skipped).

| Zoom | Class       | Roofs | Right | Called metal sheet | Called thatch | Called other roof | Missed |
| ---- | ----------- | ----- | ----- | ------------------ | ------------- | ----------------- | ------ |
| 20   | Metal sheet | 2,593 | 87.7% |                    | 2.6%          | 0.5%              | 9.3%   |
| 20   | Thatch      | 1,442 | 90.7% | 1.7%               |               | 0.0%              | 7.6%   |
| 20   | Other roof  | 100   | 51.0% | 35.0%              | 2.0%          |                   | 12.0%  |
| 19   | Metal sheet | 2,129 | 85.7% |                    | 1.9%          | 0.5%              | 11.9%  |
| 19   | Thatch      | 1,132 | 84.5% | 2.6%               |               | 0.0%              | 13.0%  |
| 19   | Other roof  | 74    | 44.6% | 24.3%              | 4.1%          |                   | 27.0%  |

Metal sheet and thatch are told apart well: of the metal sheet and thatch roofs the
model finds, 97% to 98% get the right material. Other roof is the weak class. The
model finds about half of these roofs and calls most of the rest metal sheet. There
are only 100 other roofs in the zoom 20 validation set, and their IoU moved between
0.33 and 0.43 over the last 17 epochs, so treat that score as rough. The epoch was chosen on the same validation set, which can make all scores
slightly optimistic; mean roof IoU stayed between 0.54 and 0.556 over the last 11
epochs.

![Image, OpenStreetMap label and prediction for six held-out chips in Nacala](https://huggingface.co/dexsholex/unet-roof-material/resolve/main/unet_roof_material_examples.png)

Image, label and prediction for six held-out chips. Metal sheet blue, thatch yellow,
other roof red. Imagery: #MapeandoMeuBairro via OpenAerialMap, CC BY-SA 4.0. Labels:
© OpenStreetMap contributors, ODbL.

## Related work

Guthula et al. (2024) released the Nacala-Roof-Material dataset, drone images of the
same city with their own roof outlines and roof types, and benchmarked U-Net, YOLOv8
and DINOv2 models on it. This model is trained on OpenStreetMap labels instead and is
scored on a different split, so its numbers are not comparable with theirs.

## Imagery

RGB drone or aerial imagery at zoom 19 or 20 (about 0.3 and 0.15 m per pixel). The
STAC item recommends zoom 20, where metal sheet and thatch score higher, and previews
the Ribáuè neighbourhood of Nacala.

## Usage

Pick Roof Material in fAIr, draw an area over true-colour imagery at zoom 20 (or 19)
and run a prediction: roofs come back as polygons with `class` 1, 2 or 3 and `label`
`metal_sheet`, `thatch` or `other_roof`. Raise `confidence_threshold` to keep only the
roofs the model is surest about. Batch predictions run the same code through
`inference_pipeline`. To adapt the model to a new city, label some roofs there and
finetune it.

## Fine-tuning

Fine-tuning starts from the released weights (the `checkpoint` asset, a PyTorch
`state_dict`) and trains the whole network on the labelled chips with the same
class-weighted cross-entropy plus Dice loss, randomly turning, flipping and changing
the brightness and contrast of each chip. The defaults are 15 epochs, batch size 8 and
an AdamW learning rate of 0.0001 that decays to zero on a cosine schedule. With only a
handful of labelled chips, `freeze_encoder` trains the decoder alone.

A label's class comes from its OpenStreetMap `roof:material` tag (`metal_sheet`,
`thatch`, `asbestos` or `concrete`), or else from an integer `label` of 1 (metal
sheet), 2 (thatch) or 3 (other roof). Buildings with any other material, or none, can
stay in the labels: their pixels are left out of the loss and the scores, so the model
is not taught that they are background.

The split is spatial: fAIr names chips after their map tiles, so neighbouring chips are
grouped into 4 x 4 tile blocks and whole blocks go to validation. Of 200 seeded
shuffles of the blocks, the one that holds out each roof class closest to the
validation ratio is kept, so a rare material still gets scored. Validation reports
pixel IoU, precision and recall for each roof class and their mean IoU, and the
finetuned model is exported to ONNX with the same input and output as the base model.

## Limitations

- Trained on one city, with drone imagery from 2021 and 2022. Roof styles, building
  layouts and imagery elsewhere will differ; finetune on local labels before relying
  on it there.
- Other roof is weak (about half of these roofs right) and was learned from only
  asbestos and concrete. Painted metal, tiled or other roofs may land in any class.
- About 8% to 13% of roofs are missed. In the example images, roofs partly under trees
  and large, very bright roofs are among them.
- Neighbouring roofs of the same material that touch in the class map come back as one
  polygon, and a roof that crosses a chip edge comes back as two. Chips must be
  256 x 256 pixels.
- Expects true-colour RGB. Thermal or false-colour imagery gives meaningless results.

## Citation

Related dataset: Guthula, V. B., Oehmcke, S., Chilaule, R., Zhang, H., Lang, N.,
Kariryaa, A., Mottelson, J. and Igel, C. (2024). Nacala-Roof-Material: Drone Imagery
for Roof Detection, Classification, and Segmentation to Support Mosquito-borne
Disease Risk Assessment. arXiv:2406.04949. https://doi.org/10.48550/arXiv.2406.04949

The model: Oyinloye, O. (2026). unet-roof-material: roof material segmentation for
fAIr. https://huggingface.co/dexsholex/unet-roof-material

## Licence

The model weights are released under CC-BY-SA-4.0, in line with the share-alike terms
of the training imagery (#MapeandoMeuBairro via OpenAerialMap, CC BY-SA 4.0) and the
labels (© OpenStreetMap contributors, ODbL).
