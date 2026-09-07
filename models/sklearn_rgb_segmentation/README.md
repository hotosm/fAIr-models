# RGB Segmentation (scikit-learn)

The smallest reference model in fAIr. A logistic-regression classifier labels each
pixel as building or background from its red, green and blue values. It has no
deep-learning dependencies, trains in seconds on CPU, and exports to a few-kilobyte
ONNX file, so it is a fast way to run the full fAIr flow end to end on a small machine.

## Task

Binary semantic segmentation. The model reads a 3-band RGB chip, classifies every
pixel, and vectorises the building pixels into GeoJSON polygons. It is generic: any
dataset of RGB chips with matching building (or other single-class) labels can train it.

## Inputs and outputs

| Stage      | Shape                 | Notes                                 |
| ---------- | --------------------- | ------------------------------------- |
| Input      | `[pixels, 3]` float32 | RGB values scaled to 0 to 1           |
| Output     | `[pixels, 2]` float32 | Background and building probabilities |
| Prediction | GeoJSON               | Building polygons in EPSG:4326        |

## Training

`train_model` rasterises the label polygons onto each training chip, stacks the RGB
pixels, and fits a `StandardScaler` plus `LogisticRegression` pipeline. `evaluate_model`
reports building-class IoU on the held-out chips with `sklearn.metrics.jaccard_score`.
`export_onnx` converts the fitted pipeline with skl2onnx.

## Limitations

The model uses colour alone, with no spatial context, so it separates classes only when
their colours differ. It is intended as a learning and exploration example, not for
accurate mapping.

## License

Apache-2.0.
