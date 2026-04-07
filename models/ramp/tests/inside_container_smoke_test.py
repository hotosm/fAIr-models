"""End-to-end smoke tests for models/ramp Docker runtime.

Run this script INSIDE the container. It validates:
1) Critical imports (tensorflow, ramp, segmentation_models, osgeo.gdal, solaris)
2) ``pipeline.preprocess`` path (georeference + multimask generation)
3) ``pipeline.run_preprocessing`` contract: chip/mask arrays + on-disk counts
4) Training wrappers: ``pipeline.train_model`` and ``pipeline.train_ramp_model``
   both return loaded ``tf.keras.Model``
5) ``pipeline.resolve_model_href`` for local SavedModel directories
6) ``pipeline.run_inference`` returns merged GeoJSON dict content
7) ``pipeline.run_postprocessing`` wrapper returns dict and writes output
8) Inference intermediate georeferenced rasters for debug visibility

All training, inference, and postprocessing steps delegate to pipeline.py
helpers so the smoke test exercises the same code paths as production.

Data layouts supported:
  - data/sample layout: --dataset-root /workspace/data/sample
    Uses train/oam/*.tif + train/osm/*.geojson. Converts TIF→PNG and merges
    OSM labels into a temporary input directory (hot_fair_utilities expects PNG).
  - Legacy RAMP layout: --dataset-root /path/to/dataset
    Expects dataset/input/*.png and dataset/input/labels.geojson.

Usage inside container:
  python /workspace/models/ramp/tests/inside_container_smoke_test.py \\
      --dataset-root /workspace/data/sample
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Any


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _stage(name: str) -> None:
    print(f"\n=== {name} ===")


def _count_files(path: Path, pattern: str) -> int:
    return len(list(path.glob(pattern)))


def _load_ramp_pipeline_module():
    """Import models/ramp/pipeline.py directly so smoke tests reuse production helpers."""
    import sys

    module_dir = Path("/workspace/models/ramp")
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))
    import pipeline as ramp_pipeline

    return ramp_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RAMP container smoke tests.")
    parser.add_argument(
        "--dataset-root",
        default="/workspace/data/sample",
        help="Dataset root: data/sample (train/oam + train/osm) or legacy (input/ with PNG + labels.geojson).",
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--backbone", default="efficientnetb0")
    return parser.parse_args()


def _prepare_data_sample_layout(dataset_root: Path) -> Path:
    """Convert data/sample (train/oam + train/osm) to RAMP input layout.

    hot_fair_utilities.preprocess expects input_path with *.png and labels.geojson.
    data/sample has train/oam/*.tif and train/osm/*.geojson.
    Creates dataset_root/ramp_work/input/ with PNG chips and labels.geojson.
    Returns the working root (ramp_work) for the test.
    """
    oam_dir = dataset_root / "train" / "oam"
    osm_dir = dataset_root / "train" / "osm"
    _assert(oam_dir.is_dir(), f"train/oam not found under {dataset_root}")
    _assert(osm_dir.is_dir(), f"train/osm not found under {dataset_root}")

    work_root = dataset_root / "ramp_work"
    input_dir = work_root / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

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
    _assert(len(geojson_files) > 0, f"No .geojson files in {osm_dir}")

    import geopandas as gpd
    import pandas as pd

    gdfs = [gpd.read_file(p) for p in geojson_files]
    if len(gdfs) == 1:
        merged = gdfs[0]
    else:
        crs = gdfs[0].crs or "EPSG:4326"
        merged = gpd.GeoDataFrame(
            pd.concat([g.to_crs(crs) for g in gdfs], ignore_index=True),
            crs=crs,
        )
    # Fiona/GeoJSON writers cannot reliably handle pandas extension dtypes
    # (for example string[python], Int64, boolean). Normalize to plain Python
    # objects with None for missing values before writing.
    for col in merged.columns:
        if col == "geometry":
            continue
        if pd.api.types.is_extension_array_dtype(merged[col].dtype):
            merged[col] = merged[col].astype(object).where(merged[col].notna(), None)
    labels_path = input_dir / "labels.geojson"
    merged.to_file(labels_path, driver="GeoJSON")
    print(f"PASS: prepared {len(tif_files)} PNG chip(s) + labels.geojson from data/sample")

    return work_root


def main() -> None:
    args = parse_args()

    os.environ.setdefault("RAMP_HOME", "/workspace")
    # segmentation_models picks the backend when the package is first imported.
    # Must be tf.keras for TF 2.15+ (Keras 3); calling set_framework() after import is too late.
    os.environ.setdefault("SM_FRAMEWORK", "tf.keras")

    dataset_root = Path(args.dataset_root).resolve()

    # Support data/sample layout (train/oam + train/osm) or legacy (input/ with PNG + labels)
    if (dataset_root / "train" / "oam").is_dir() and (dataset_root / "train" / "osm").is_dir():
        dataset_root = _prepare_data_sample_layout(dataset_root)

    input_dir = dataset_root / "input"
    preprocessed_dir = dataset_root / "preprocessed_test"
    chips_dir = preprocessed_dir / "chips"
    masks_dir = preprocessed_dir / "multimasks"
    pred_input_dir = preprocessed_dir / "chips"
    pred_output_dir = dataset_root / "prediction_test" / "output"
    vectors_dir = dataset_root / "prediction_test" / "vectors"

    # -------------------------------------------------------------------------
    _stage("Test 1: Critical imports")

    import hot_fair_utilities  # noqa: F401
    import ramp  # noqa: F401
    import segmentation_models as sm
    import solaris
    import tensorflow as tf
    from osgeo import gdal

    sm.set_framework("tf.keras")  # redundant if SM_FRAMEWORK already set; keeps intent explicit
    print(f"PASS: tensorflow {tf.__version__} / segmentation_models / ramp / gdal imported")
    print(f"PASS: GDAL runtime version {gdal.VersionInfo('--version')}")
    print(f"PASS: solaris {solaris.__version__} imported")

    # -------------------------------------------------------------------------
    _stage("Test 2: Input dataset layout checks")

    _assert(dataset_root.is_dir(), f"Dataset root not found: {dataset_root}")
    _assert(input_dir.is_dir(), f"Input folder not found: {input_dir}")
    _assert(
        (input_dir / "labels.geojson").is_file(),
        f"labels.geojson not found in {input_dir}",
    )
    n_png = _count_files(input_dir, "*.png")
    _assert(n_png > 0, f"No PNG chips found in {input_dir}. Add OAM PNG chips to run this test.")
    print(f"PASS: found {n_png} PNG chip(s) + labels.geojson")

    ramp_pipeline = _load_ramp_pipeline_module()

    # -------------------------------------------------------------------------
    _stage("Test 3: Preprocessing via pipeline.preprocess + run_preprocessing contract")

    shutil.rmtree(preprocessed_dir, ignore_errors=True)
    out_path = ramp_pipeline.preprocess(
        input_path=str(input_dir),
        output_path=str(preprocessed_dir),
        boundary_width=3,
        contact_spacing=8,
    )
    _assert(Path(out_path).resolve() == preprocessed_dir.resolve(), f"Unexpected preprocess output: {out_path}")

    n_chips = _count_files(chips_dir, "*.tif")
    n_masks = _count_files(masks_dir, "*.mask.tif")
    _assert(n_chips > 0, f"No chips produced in {chips_dir}")
    _assert(n_masks > 0, f"No multimasks produced in {masks_dir}")
    _assert(n_chips == n_masks, f"Chip count ({n_chips}) != mask count ({n_masks})")
    print(f"PASS: preprocessing produced {n_chips} chip(s) + {n_masks} multimask(s)")

    # Use pipeline.run_preprocessing directly so this smoke test follows pipeline.py behavior.
    import numpy as np

    data_loader_contract: list[tuple[Any, Any]] = ramp_pipeline.run_preprocessing(
        input_path=str(input_dir),
        output_path=str(preprocessed_dir),
        boundary_width=3,
        contact_spacing=8,
    )
    _assert(len(data_loader_contract) > 0, "Expected at least one chip/mask array pair.")
    _assert(
        isinstance(data_loader_contract[0][0], np.ndarray),
        "Chip data must be a numpy ndarray (pipeline.run_preprocessing contract).",
    )
    _assert(
        isinstance(data_loader_contract[0][1], np.ndarray),
        "Mask data must be a numpy ndarray (pipeline.run_preprocessing contract).",
    )
    print(
        f"PASS: dataloader contract — {len(data_loader_contract)} (chip, mask) ndarray pair(s), "
        "matching pipeline.run_preprocessing"
    )

    # -------------------------------------------------------------------------
    _stage("Test 4: Training smoke run via train_model/train_ramp_model wrappers")
    # train_model wraps train_ramp_model and validates preprocessing output.
    # train_ramp_model handles val split internally and returns loaded tf.keras.Model.

    import tensorflow as tf

    trained_model_step = ramp_pipeline.train_model(
        data_base_path=str(dataset_root),
        data_loader=data_loader_contract,
        preprocessed_path=str(preprocessed_dir),
        stac_item_path="/workspace/models/ramp/stac-item.json",
        val_fraction=0.15,
    )
    _assert(isinstance(trained_model_step, tf.keras.Model), "train_model did not return a Keras model checkpoint.")

    trained_model = ramp_pipeline.train_ramp_model(
        data_base_path=str(dataset_root),
        preprocessed_path=str(preprocessed_dir),
        stac_item_path="/workspace/models/ramp/stac-item.json",
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        backbone=args.backbone,
        early_stopping_patience=2,
        log_zenml_step_metadata=False,
    )
    _assert(isinstance(trained_model, tf.keras.Model), "train_ramp_model did not return a Keras model checkpoint.")
    print(f"PASS: {args.epochs}-epoch training completed; train wrappers returned tf.keras.Model")

    # -------------------------------------------------------------------------
    _stage("Test 5: resolve_model_href local SavedModel resolution")

    local_saved_model_dir = dataset_root / "local_smoke_savedmodel"
    shutil.rmtree(local_saved_model_dir, ignore_errors=True)
    trained_model.save(str(local_saved_model_dir))
    resolved_model_dir = ramp_pipeline.resolve_model_href(str(local_saved_model_dir))
    _assert(
        (Path(resolved_model_dir) / "saved_model.pb").is_file(),
        f"resolve_model_href did not return a SavedModel dir: {resolved_model_dir}",
    )
    print("PASS: resolve_model_href resolved a valid local SavedModel directory")

    # -------------------------------------------------------------------------
    _stage("Test 6: Inference smoke run via run_inference")

    shutil.rmtree(pred_output_dir, ignore_errors=True)
    pred_output_dir.mkdir(parents=True, exist_ok=True)
    final_geojson = ramp_pipeline.run_inference(
        model_uri=resolved_model_dir,
        input_path=str(pred_input_dir),
        prediction_path=str(pred_output_dir),
        output_dir=str(vectors_dir),
    )
    _assert(isinstance(final_geojson, dict), "Inference did not return GeoJSON dict content.")
    _assert(final_geojson.get("type") == "FeatureCollection", "GeoJSON response missing FeatureCollection type.")
    print(f"PASS: run_inference returned GeoJSON content with {len(final_geojson.get('features', []))} feature(s)")

    # -------------------------------------------------------------------------
    _stage("Test 7: Postprocessing via run_postprocessing wrapper")

    georef_dir = pred_output_dir / "georeference"
    postprocess_out_dir = dataset_root / "prediction_test" / "vectors_postprocess_wrapper"
    shutil.rmtree(postprocess_out_dir, ignore_errors=True)
    postprocess_out_dir.mkdir(parents=True, exist_ok=True)
    wrapped_geojson = ramp_pipeline.run_postprocessing(
        prediction_path=str(georef_dir),
        output_dir=str(postprocess_out_dir),
    )
    _assert(isinstance(wrapped_geojson, dict), "run_postprocessing did not return GeoJSON dict content.")
    _assert(
        (postprocess_out_dir / "predictions.geojson").is_file(),
        "run_postprocessing did not write predictions.geojson",
    )
    print("PASS: run_postprocessing returned dict and wrote predictions.geojson")

    # -------------------------------------------------------------------------
    _stage("Test 8: Inference intermediate artifacts")

    _assert(georef_dir.is_dir(), f"Georeferenced output directory not found: {georef_dir}")
    n_pred = _count_files(georef_dir, "*.tif")
    _assert(n_pred > 0, f"No georeferenced prediction .tif files produced in {georef_dir}")
    print(f"PASS: fairpredictor produced {n_pred} georeferenced prediction .tif file(s)")

    # -------------------------------------------------------------------------
    _stage("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
