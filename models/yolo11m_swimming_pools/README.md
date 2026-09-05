# yolo11m-swimming-pools

## Overview

Single-class swimming pool **object detection** for VHR RGB aerial imagery on the fAIr platform. This model pack implements the fAIr/ZenML contract and delegates most model logic to the reusable [`spd-hot`](https://github.com/AbdelrahmanKatkat/spd-hot) library (preprocess, ONNX decode + NMS, georeferencing, and GeoJSON output).

The model returns a GeoJSON `FeatureCollection` where each detection is represented as an EPSG:4326 `Polygon` (the bbox corners) with a confidence score.

## Architecture

- **Model**: Ultralytics YOLO11m, single class `swimming_pool` (`value: 0`).
- **Input**: 3-band RGB GeoTIFF chips; ONNX input size is **640×640** (declared in `stac-item.json` `mlm:input`).
- **Output**: GeoJSON polygons (bboxes as polygons), with per-detection confidence.
- **Serving**: ONNX Runtime session; decoding + NMS + georeferencing implemented in `spd-hot`.

## Pretrained source

Pretrained weights and the original dataset recipe come from Mozilla AI:

- Weights: [mozilla-ai/swimming-pool-detector](https://huggingface.co/mozilla-ai/swimming-pool-detector)
- Dataset recipe: OSM `leisure=swimming_pool` (excluding indoor) matched to Mapbox z18 tiles via Mozilla’s [osm-ai-helper](https://github.com/mozilla-ai/osm-ai-helper) (see also [mozilla-ai/osm-swimming-pools](https://huggingface.co/datasets/mozilla-ai/osm-swimming-pools)).

## Limitations

- **Domain shift (Mapbox → OAM)**: upstream training imagery is Mapbox; OpenAerialMap scenes can differ in sensor, radiometry, and GSD. An OAM-specific finetune dataset is **In progress**.
- **Example pipeline dataset**: the repo’s generic example runner (`examples/run.py`) currently fine-tunes detection models on the Banepa **buildings** sample dataset. That dataset is not swimming pools; updating examples to support a per-model dataset choice is **In progress**.
- **Smoke-test geography**: GitHub smoke tests are currently wired to the Banepa OAM TMS bbox. This validates packaging/serving, not pool quality.
- **Label definition**: `spd-hot` converts GeoJSON geometries to YOLO labels and does not filter by feature properties. For pool finetuning, the labels GeoJSON must contain pool polygons only (do not mix buildings/other features).

## Usage

- **Live inference**: run the inference container and call `POST /predict` with a `model_uri` (ONNX), a TMS `image_uri`, a bbox, a zoom, and inference params. Defaults and preview fields are defined in `models/yolo11m_swimming_pools/stac-item.json`.
- **Fine-tuning**: run `training_pipeline` with a dataset that provides `chips` (RGB GeoTIFF directory) and `labels` (directory containing exactly one GeoJSON of pool polygons). The pipeline trains on the train split only, evaluates on the val split only, and exports ONNX.
- **Contributor workflow**: follow `docs/getting-started.md` and `docs/contributing/model.md` (validate → build → example run → serve smoke).

## Citation

- Upstream detector and dataset recipe: [mozilla-ai/swimming-pool-detector](https://huggingface.co/mozilla-ai/swimming-pool-detector), [mozilla-ai/osm-swimming-pools](https://huggingface.co/datasets/mozilla-ai/osm-swimming-pools), [osm-ai-helper](https://github.com/mozilla-ai/osm-ai-helper).
- Architecture: [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/).
- fAIr adapter + decode/georef helpers: [`spd-hot`](https://github.com/AbdelrahmanKatkat/spd-hot).

## License

AGPL-3.0-only (this model pack; must match `properties.license` in `stac-item.json`). OpenStreetMap-derived labels are under ODbL. Upstream weights and dataset licensing are described in the linked Hugging Face model and dataset cards.

