"""End-to-end smoke tests for models/yolo_v8_v2 Docker runtime.

Run this script inside the container. It validates:
1) critical imports (gdal/cv2/ultralytics/hot_fair_utilities),
2) dataset layout and required inputs,
3) preprocess + YOLO formatting,
4) short training run (checkpoint creation),
5) inference output generation,
6) polygonization to GeoJSON.

Data layouts supported:
  - data/sample: --dataset-root /workspace/data/sample
  - Legacy: dataset/input/ + dataset/prediction/input/
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _stage(name: str) -> None:
    print(f"\n=== {name} ===")


def _count_images(path: Path) -> int:
    return len(list(path.glob("*.png"))) + len(list(path.glob("*.tif"))) + len(
        list(path.glob("*.tiff"))
    )


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
    """Convert data/sample to YOLO input layout."""
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
    merged = gdfs[0] if len(gdfs) == 1 else gpd.GeoDataFrame(
        pd.concat([g.to_crs(crs) for g in gdfs], ignore_index=True),
        crs=crs,
    )
    merged.to_file(input_dir / "labels.geojson", driver="GeoJSON")

    pred_tifs = list(predict_oam.glob("OAM-*.tif"))
    _assert(len(pred_tifs) > 0, f"No OAM-*.tif in {predict_oam}")
    for tif in pred_tifs[:5]:
        shutil.copy(tif, pred_input_dir / tif.name)

    print(f"PASS: prepared from data/sample")
    return work_root


def main() -> None:
    args = parse_args()

    os.environ.setdefault("RAMP_HOME", "/workspace")

    dataset_root = Path(args.dataset_root).resolve()
    if (dataset_root / "train" / "oam").is_dir() and (dataset_root / "train" / "osm").is_dir():
        dataset_root = _prepare_data_sample_layout(dataset_root)

    input_dir = dataset_root / "input"
    pred_input_dir = dataset_root / "prediction" / "input"
    preprocessed_dir = dataset_root / "preprocessed_test"
    yolo_dir = dataset_root / "yolo_test"
    pred_output_dir = dataset_root / "prediction" / "output_test"
    geojson_output = pred_output_dir / "prediction.geojson"

    _stage("Test 1: Critical imports")
    import cv2  # noqa: F401
    import hot_fair_utilities  # noqa: F401
    import ultralytics  # noqa: F401
    from osgeo import gdal  # noqa: F401

    print("PASS: gdal/cv2/ultralytics/hot_fair_utilities imported successfully")

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

    _stage("Test 3: Preprocess + YOLO format")
    shutil.rmtree(preprocessed_dir, ignore_errors=True)
    shutil.rmtree(yolo_dir, ignore_errors=True)

    from hot_fair_utilities.preprocessing.preprocess import preprocess
    from hot_fair_utilities.preprocessing.yolo_v8_v2.yolo_format import yolo_format

    preprocess(
        input_path=str(input_dir),
        output_path=str(preprocessed_dir),
        rasterize=True,
        rasterize_options=["binary"],
        georeference_images=True,
        multimasks=True,
    )
    yolo_format(
        preprocessed_dirs=str(preprocessed_dir),
        yolo_dir=str(yolo_dir),
        multimask=True,
        p_val=0.05,
    )
    dataset_yaml = yolo_dir / "yolo_dataset.yaml"
    _assert(dataset_yaml.is_file(), f"YOLO dataset yaml not found: {dataset_yaml}")
    print(f"PASS: yolo_dataset.yaml created at {dataset_yaml}")

    _stage("Test 4: Training smoke test")
    from hot_fair_utilities.training.yolo_v8_v2.train import train

    model_path, iou = train(
        data=str(dataset_root),
        weights=args.weights,
        epochs=args.epochs,
        batch_size=args.batch_size,
        pc=args.pc,
        output_path=str(yolo_dir),
        dataset_yaml_path=str(dataset_yaml),
    )
    model_path = Path(model_path)
    _assert(model_path.is_file(), f"Checkpoint not found after training: {model_path}")
    print(f"PASS: checkpoint created: {model_path}")
    print(f"INFO: reported IoU%: {iou}")

    _stage("Test 5: Inference smoke test")
    shutil.rmtree(pred_output_dir, ignore_errors=True)
    pred_output_dir.mkdir(parents=True, exist_ok=True)

    from hot_fair_utilities.inference.predict import predict

    predict(
        checkpoint_path=str(model_path),
        input_path=str(pred_input_dir),
        prediction_path=str(pred_output_dir),
        confidence=args.confidence,
        remove_images=False,
    )
    pred_tifs = list(pred_output_dir.glob("*.tif"))
    _assert(
        len(pred_tifs) > 0,
        f"Inference did not create georeferenced prediction tif files in {pred_output_dir}",
    )
    print(f"PASS: inference generated {len(pred_tifs)} tif predictions")

    _stage("Test 6: Polygonization")
    from hot_fair_utilities.postprocessing.polygonize import polygonize

    polygonize(
        input_path=str(pred_output_dir),
        output_path=str(geojson_output),
        remove_inputs=False,
    )
    _assert(geojson_output.is_file(), f"GeoJSON output not found: {geojson_output}")
    print(f"PASS: polygonization output created: {geojson_output}")

    _stage("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
