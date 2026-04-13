"""End-to-end smoke tests for models/yolo_v8_v2 Docker runtime.

Run this script INSIDE the container. It validates:
1) Critical imports (gdal, cv2, ultralytics, hot_fair_utilities)
2) Dataset layout and required inputs
3) Preprocess via pipeline.preprocess + YOLO formatting via pipeline.split_dataset
4) Short training run via pipeline.train_yolo_model (checkpoint creation)
5) resolve_model_href for local .pt files
6) Inference via pipeline.infer_yolo_model (output generation)
7) Polygonization via pipeline.postprocess to GeoJSON
8) Inference intermediate artifacts check

All training, inference, and postprocessing steps delegate to pipeline.py
helpers so the smoke test exercises the same code paths as production.

Data layouts supported:
  - data/sample: --dataset-root /workspace/data/sample
    Uses train/oam/*.tif + train/osm/*.geojson + predict/oam/*.tif.
  - Legacy: dataset/input/ + dataset/prediction/input/

Usage inside container:
  python /workspace/models/yolo_v8_v2/tests/inside_container_smoke_test.py \\
      --dataset-root /workspace/data/sample
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _stage(name: str) -> None:
    print(f"\n=== {name} ===")


def _count_images(path: Path) -> int:
    return len(list(path.glob("*.png"))) + len(list(path.glob("*.tif"))) + len(list(path.glob("*.tiff")))


def _load_yolo_pipeline_module():
    """Import models/yolo_v8_v2/pipeline.py directly so smoke tests reuse production helpers."""
    module_dir = Path("/workspace/models/yolo_v8_v2")
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))
    import pipeline as yolo_pipeline

    return yolo_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO v8 v2 container smoke tests.")
    parser.add_argument(
        "--dataset-root",
        default="/workspace/data/sample",
        help="Dataset root: data/sample or legacy layout.",
    )
    parser.add_argument(
        "--weights",
        default="yolov8s_v2-seg.pt",
        help="Initial YOLO weights path. Downloads automatically if missing.",
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--pc", type=float, default=2.0)
    parser.add_argument("--confidence", type=float, default=0.5)
    return parser.parse_args()


def _prepare_data_sample_layout(dataset_root: Path) -> Path:
    """Convert data/sample to YOLO input layout.

    hot_fair_utilities.preprocess expects input_path with *.png and labels.geojson.
    data/sample has train/oam/*.tif and train/osm/*.geojson.
    Creates dataset_root/yolo_work/input/ with PNG chips and labels.geojson,
    plus dataset_root/yolo_work/prediction/input/ with prediction TIFs.
    Returns the working root (yolo_work) for the test.
    """
    oam_dir = dataset_root / "train" / "oam"
    osm_dir = dataset_root / "train" / "osm"
    predict_oam = dataset_root / "predict" / "oam"
    _assert(oam_dir.is_dir(), f"train/oam not found under {dataset_root}")
    _assert(osm_dir.is_dir(), f"train/osm not found under {dataset_root}")
    _assert(predict_oam.is_dir(), f"predict/oam not found under {dataset_root}")

    work_root = dataset_root / "yolo_work"
    input_dir = work_root / "input"
    pred_input_dir = work_root / "prediction" / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    pred_input_dir.mkdir(parents=True, exist_ok=True)

    import numpy as np
    import rasterio
    from PIL import Image

    tif_files = sorted(oam_dir.glob("OAM-*.tif"))
    _assert(len(tif_files) > 0, f"No OAM-*.tif files in {oam_dir}")
    for tif_path in tif_files:
        png_path = input_dir / (tif_path.stem + ".png")
        with rasterio.open(tif_path) as src:
            data = src.read()
            if data.shape[0] >= 3:
                rgb = np.transpose(data[:3], (1, 2, 0))
                if rgb.max() <= 1.0:
                    rgb = (rgb * 255).astype(np.uint8)
                Image.fromarray(rgb).save(png_path)

    geojson_files = sorted(osm_dir.glob("*.geojson"))
    _assert(len(geojson_files) > 0, f"No .geojson in {osm_dir}")
    import geopandas as gpd
    import pandas as pd

    gdfs = [gpd.read_file(p) for p in geojson_files]
    crs = gdfs[0].crs or "EPSG:4326"
    merged = (
        gdfs[0]
        if len(gdfs) == 1
        else gpd.GeoDataFrame(
            pd.concat([g.to_crs(crs) for g in gdfs], ignore_index=True),
            crs=crs,
        )
    )
    # Normalize extension dtypes for Fiona GeoJSON compatibility
    for col in merged.columns:
        if col == "geometry":
            continue
        if pd.api.types.is_extension_array_dtype(merged[col].dtype):
            merged[col] = merged[col].astype(object).where(merged[col].notna(), None)
    merged.to_file(input_dir / "labels.geojson", driver="GeoJSON")

    pred_tifs = list(predict_oam.glob("OAM-*.tif"))
    _assert(len(pred_tifs) > 0, f"No OAM-*.tif in {predict_oam}")
    for tif in pred_tifs[:5]:
        shutil.copy(tif, pred_input_dir / tif.name)

    print("PASS: prepared from data/sample")
    return work_root


def main() -> None:
    args = parse_args()

    os.environ.setdefault("RAMP_HOME", "/workspace")

    # Load pipeline module early so helpers are available
    yolo_pipeline = _load_yolo_pipeline_module()

    dataset_root = Path(args.dataset_root).resolve()
    if (dataset_root / "train" / "oam").is_dir() and (dataset_root / "train" / "osm").is_dir():
        dataset_root = _prepare_data_sample_layout(dataset_root)

    input_dir = dataset_root / "input"
    pred_input_dir = dataset_root / "prediction" / "input"
    output_dir = dataset_root / "output_test"
    pred_output_dir = dataset_root / "prediction" / "output_test"
    geojson_output = pred_output_dir / "prediction.geojson"

    # -------------------------------------------------------------------------
    _stage("Test 1: Critical imports")
    import cv2  # noqa: F401
    import hot_fair_utilities  # noqa: F401
    import rasterio  # noqa: F401  # YOLO image has rasterio, not osgeo Python bindings
    import ultralytics  # noqa: F401

    print("PASS: rasterio/cv2/ultralytics/hot_fair_utilities imported successfully")

    # -------------------------------------------------------------------------
    _stage("Test 2: Dataset layout checks")
    _assert(dataset_root.is_dir(), f"Dataset root not found: {dataset_root}")
    _assert(input_dir.is_dir(), f"Training input folder not found: {input_dir}")
    _assert(
        (input_dir / "labels.geojson").is_file(),
        f"labels.geojson not found in: {input_dir}",
    )
    _assert(
        _count_images(input_dir) > 0,
        f"No training chips found in {input_dir} (*.png/*.tif/*.tiff)",
    )
    _assert(
        pred_input_dir.is_dir(),
        f"Inference input folder not found: {pred_input_dir}",
    )
    _assert(
        _count_images(pred_input_dir) > 0,
        f"No inference chips found in {pred_input_dir} (*.png/*.tif/*.tiff)",
    )
    print("PASS: required dataset structure and files exist")

    # -------------------------------------------------------------------------
    _stage("Test 3: Preprocess + YOLO format via pipeline.preprocess + split_dataset")
    shutil.rmtree(output_dir, ignore_errors=True)

    preprocessed_dir = yolo_pipeline.preprocess(
        input_path=str(input_dir),
        output_path=str(output_dir),
        p_val=0.2,
    )

    split_info = yolo_pipeline.split_dataset(
        preprocessed_path=str(preprocessed_dir),
        output_path=str(output_dir),
        hyperparameters={"p_val": 0.2, "split_seed": 42},
    )
    yolo_dir = Path(split_info["yolo_data_dir"])
    dataset_yaml = Path(split_info["dataset_yaml"])
    _assert(dataset_yaml.is_file(), f"YOLO dataset yaml not found: {dataset_yaml}")
    print(f"PASS: yolo_dataset.yaml created at {dataset_yaml}")

    # -------------------------------------------------------------------------
    _stage("Test 4: Training smoke test via pipeline.train_yolo_model")

    model_path, iou = yolo_pipeline.train_yolo_model(
        data_base_path=str(dataset_root),
        yolo_data_dir=str(yolo_dir),
        weights_path=args.weights,
        epochs=args.epochs,
        batch_size=args.batch_size,
        pc=args.pc,
    )
    model_path = Path(model_path)
    _assert(model_path.is_file(), f"Checkpoint not found after training: {model_path}")
    print(f"PASS: checkpoint created: {model_path}")
    print(f"INFO: reported IoU%: {iou}")

    # -------------------------------------------------------------------------
    _stage("Test 5: resolve_model_href local .pt resolution")

    resolved = yolo_pipeline.resolve_model_href(str(model_path))
    _assert(
        Path(resolved).is_file(),
        f"resolve_model_href did not return a valid .pt file: {resolved}",
    )
    print("PASS: resolve_model_href resolved a valid local .pt checkpoint")

    # -------------------------------------------------------------------------
    _stage("Test 6: Inference smoke test via pipeline.infer_yolo_model")
    shutil.rmtree(pred_output_dir, ignore_errors=True)
    pred_output_dir.mkdir(parents=True, exist_ok=True)

    result_dict = yolo_pipeline.infer_yolo_model(
        model_uri=str(model_path),
        input_path=str(pred_input_dir),
        prediction_path=str(pred_output_dir),
        output_dir=str(pred_output_dir),
        confidence=args.confidence,
    )

    _assert(isinstance(result_dict, dict), "Inference did not return GeoJSON dict content.")
    _assert(result_dict.get("type") == "FeatureCollection", "GeoJSON response missing FeatureCollection type.")
    print(f"PASS: inference returned GeoJSON dict with {len(result_dict.get('features', []))} feature(s)")

    pred_tifs = list(pred_output_dir.glob("*.tif"))
    _assert(
        len(pred_tifs) > 0,
        f"Inference did not create georeferenced prediction tif files in {pred_output_dir}",
    )
    print(f"PASS: inference generated {len(pred_tifs)} tif predictions")

    # -------------------------------------------------------------------------
    _stage("Test 7: Polygonization via pipeline.postprocess")

    geojson_dict = yolo_pipeline.postprocess(
        prediction_path=str(pred_output_dir),
        output_geojson=str(geojson_output),
    )
    _assert(isinstance(geojson_dict, dict), "postprocess did not return GeoJSON dict content.")
    _assert(Path(geojson_output).is_file(), f"GeoJSON output not found: {geojson_output}")
    print(f"PASS: polygonization output created: {geojson_output}")

    # -------------------------------------------------------------------------
    _stage("Test 8: Inference intermediate artifacts")

    _assert(
        len(pred_tifs) > 0,
        f"No georeferenced prediction .tif files found in {pred_output_dir}",
    )
    print(f"PASS: {len(pred_tifs)} georeferenced prediction .tif file(s) verified")

    # -------------------------------------------------------------------------
    _stage("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
