# RAMP EfficientNetB0 + U-Net Building Segmentation

Semantic segmentation model for building footprint extraction from RGB aerial imagery, derived from the RAMP (Replicable AI for Microplanning) project.

## Architecture

- **Model**: EfficientNetB0 encoder + U-Net decoder (`EffUnet`)
- **Framework**: TensorFlow 2.15 / `tf.keras` (via `segmentation_models` with `SM_FRAMEWORK=tf.keras`)
- **Task**: Semantic segmentation (sparse categorical crossentropy)
- **Input**: RGB chips (256x256, float32, channels-last)
- **Classes**: 4 (background=0, building=1, boundary=2, contact=3)

The boundary (class 2) and contact (class 3) channels help the model cleanly separate adjacent buildings at inference time, even when they share a wall. The `predict()` helper collapses the 4-class softmax to a binary building mask before vectorization.

## Pretrained Source

Baseline RAMP weights (TensorFlow SavedModel) hosted by HOTOSM:

- Checkpoint: `https://api-prod.fair.hotosm.org/api/v1/workspace/download/ramp/baseline.zip`
- ONNX model: `https://api-prod.fair.hotosm.org/api/v1/workspace/download/ramp/ramp-v1.onnx`

## Pipeline

Training pipeline steps (ZenML) defined in `pipeline.py`:

- `split_dataset` - preprocesses chips + labels into 4-class multimasks and produces a seeded random train/validation split via `hot_fair_utilities.split_training_2_validation`
- `train_model` - fine-tunes RAMP on the split, returning the best SavedModel serialized as a zipped byte stream
- `evaluate_model` - computes `fair:accuracy`, `fair:mean_iou` (building IoU), `fair:precision`, and `fair:recall` on the validation split
- `export_onnx` - converts the trained SavedModel to an ONNX byte stream via `tf2onnx`

Inference is served through `fair.serve.base`, which calls the module-level `predict(session, input_images, params) -> FeatureCollection`: each chip is preprocessed, run through an `onnxruntime` session, decoded to a binary building mask, and vectorized to georeferenced polygons.

## Base Image

Training, test, and inference Docker stages all build on
`ghcr.io/hotosm/fair-utilities-ramp:cpu-latest` (or `:gpu-latest` via the
`BASE_IMAGE` build arg), which provides TensorFlow, GDAL, `hot_fair_utilities`
RAMP extras, and the RAMP runtime under `/app/.venv`.

```bash
# Build targets
docker build -f models/ramp/Dockerfile --target runtime   -t ramp:runtime   .
docker build -f models/ramp/Dockerfile --target test      -t ramp:test      .
docker build -f models/ramp/Dockerfile --target inference -t ramp:inference .
```

## Limitations and Bias

- Training data and baseline weights are derived from the RAMP corpus (primarily humanitarian-mapping contexts); performance on dense urban scenes with complex roof structures may be lower than on sparser rural settlements.
- The model is sensitive to imagery with strong color casts, motion blur, or significant off-nadir angle; preprocess inputs to approximately nadir RGB at the target zoom before inference.
- Binary building output from `predict()` discards the boundary/contact auxiliary classes after decoding; downstream polygonization may still merge neighbouring buildings that share a footprint edge.

## Citation

RAMP - Replicable AI for Microplanning. Upstream source: https://github.com/radiantearth/ramp-code

## License

- Model weights and code: Apache-2.0
- Training data: ODbL-1.0 (OpenStreetMap-derived labels)
