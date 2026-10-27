# Freetown Trees

A U-Net that labels each pixel of an RGB aerial chip as tree or background, then
traces the tree pixels into crown polygons. It is a prediction-only model: the
repository holds its ONNX export and the prediction code, with no training
pipeline, so it serves live and batch predictions and cannot be finetuned.

## Task

Binary semantic segmentation of tree crowns. Each 256 x 256 chip goes through the
network in one pass, the tree probability is thresholded, and the mask is
vectorised into GeoJSON polygons.

## Inputs and outputs

| Stage      | Shape                      | Notes                                         |
| ---------- | -------------------------- | --------------------------------------------- |
| Input      | `[batch, 3, 256, 256]` float32 | RGB values divided by 255                 |
| Output     | `[batch, 2, 256, 256]` float32 | Background and tree logits per pixel      |
| Prediction | GeoJSON                    | Tree crown polygons in EPSG:4326              |

`confidence_threshold` (default 0.5) is the minimum tree probability for a pixel
to count as tree.

## Imagery

Checked on OpenAerialMap imagery of Freetown, Sierra Leone, at zooms 18 to 20.
The STAC item recommends zoom 19.

## Limitations

Polygons are traced per chip, so a crown that crosses a chip edge comes back as
two polygons. Chips must be 256 x 256 pixels.
