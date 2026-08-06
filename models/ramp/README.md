# RAMP EfficientNetB0 + U-Net Building Segmentation

## Overview

RAMP is a semantic-segmentation model that extracts **building footprints** from **RGB very-high-resolution aerial/satellite imagery**. In this repository it is packaged as a **global** base model (worldwide STAC footprint) intended for **fine-tuning on a target area** and for **ONNX-based inference** within the fAIr pipeline.

## Architecture

- **Model**: `EffUnet` (EfficientNetB0 encoder + U-Net decoder)
- **Task**: semantic segmentation
- **Input**: \([-1, 256, 256, 3]\) float32 RGB chips (channels-last)
- **Output**: \([-1, 256, 256, 4]\) float32 per-class scores for 4 classes
  - **0**: background
  - **1**: building
  - **2**: boundary (helps separate adjacent buildings)
  - **3**: contact (helps separate close neighbouring buildings)

The boundary (class 2) and contact (class 3) channels help the model cleanly separate adjacent buildings at inference time, even when they share a wall. The `predict()` helper collapses the 4-class softmax to a binary building mask before vectorization.

## Pretrained Source

Pretrained artifacts for this model pack are declared in `models/ramp/stac-item.json`:

- **Checkpoint (zipped TF SavedModel)**: `https://huggingface.co/hotosm/ramp/resolve/74daea54694f2e4924f1222520c614c7f5c029fe/v1-baseline.zip`
- **ONNX model**: `https://huggingface.co/hotosm/ramp/resolve/83c77a7e5feb3af62e3604d7bb96c6c6e9ff1a96/ramp-v1.onnx`

Upstream RAMP resources:

- **Project documentation**: `https://rampml.global/`
- **Upstream codebase**: `https://github.com/devglobalpartners/ramp-code`
- **Training data**: RAMP datasets on Radiant MLHub (see `https://rampml.global/training-data/`). Per upstream docs, labels are **manually annotated** building footprints; the published RAMP training datasets are released under **CC BY-NC 4.0**.

Published paper reference: none is included in this repository; cite the upstream project documentation/repository.

## Limitations

- **Domain shift**: performance may degrade on imagery with different sensors, resolutions, or preprocessing than the training distribution.
- **Challenging imagery**: haze, blur, strong color casts, or off-nadir views can reduce quality.
- **Dense/attached roofs**: polygonization may still merge neighbouring buildings in dense urban areas.

## Usage

### Inference (Docker)

```bash
docker build -f models/ramp/Dockerfile --target inference -t ramp:inference .
docker run --rm -p 8080:8080 -e MODEL_MODULE=models.ramp.pipeline ramp:inference

curl -s http://localhost:8080/health
```

Example request (server downloads tiles into chips, runs ONNX, returns GeoJSON FeatureCollection):

```bash
curl -s http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "model_uri": "https://huggingface.co/hotosm/ramp/resolve/83c77a7e5feb3af62e3604d7bb96c6c6e9ff1a96/ramp-v1.onnx",
    "image_uri": "https://example.com/tiles/{z}/{x}/{y}.png",
    "bbox": [0.0, 0.0, 0.01, 0.01],
    "zoom": 18,
    "params": { "confidence_threshold": 0.5, "min_class_value": 1 }
  }'
```

### Fine-tuning (Python)

Fine-tuning is implemented as a ZenML pipeline in `models/ramp/pipeline.py` and is exercised end-to-end in `models/test_integration.py` via `fair.client.FairClient` (register → finetune → promote → predict).

## Citation

```bibtex
@misc{ramp_docs,
  title        = {Replicable AI for Microplanning (ramp)},
  author       = {{DevGlobal Partners}},
  howpublished = {\\url{https://rampml.global/}},
  note         = {Accessed 2026-05-25}
}
```

## License

Apache-2.0
