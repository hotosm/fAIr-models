# segformer_parking_lot

SegFormer MiT-B5 parking-lot semantic segmentation model pack for fAIr.

## Summary

- Task: semantic segmentation (parking lot vs background)
- Input: RGB GeoTIFF chips
- Output: EPSG:4326 GeoJSON polygons (`class_name=parking_lot`)
- Base checkpoint: `UTEL-UIUC/SegFormer-large-parking`
- Library: delegates model logic to [`pl-hot`](https://github.com/AbdelrahmanKatkat/pl-hot)

## Pipeline shape

- `split_dataset`: prepares train/val image+mask folders from chips + one GeoJSON.
- `train_model`: fine-tunes SegFormer using `pl_hot.train.train_segformer`.
- `evaluate_model`: reports `accuracy`, `mean_iou` (+ optional per-class IoU).
- `export_onnx`: exports single-file ONNX bytes via `pl_hot.export.export_onnx_bytes`.
- `run_inference` uses the shared `predict()` path: preprocess each chip, run
  ONNX, explicitly call `postprocess()` to decode the logits, then polygonize,
  clean, and merge the results into GeoJSON.

## Notes

- `prepare_seg_dataset_from_geojson` raises if train or val would be empty (for example, all chips in one spatial block).
- SegFormer ONNX logits can be lower resolution (H/4 x W/4). Postprocess upsamples logits to `model_input_size` before thresholding.
- fAIr strips `inference.` from STAC keys before passing them to `predict()`.
- Inference image is torch-free: ONNX Runtime + geospatial stack only.
