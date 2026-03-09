"""End-to-end smoke tests for models/ramp Docker runtime.

Run this script INSIDE the container. It validates:
1) Critical imports (tensorflow, ramp, segmentation_models, osgeo.gdal, solaris)
2) hot-fair-utilities preprocessing (georeference + multimask generation)
3) Train/val split
4) Short training run (2 epochs, checkpoint creation)
5) Inference output generation (.pred.tif per chip)
6) Polygonization (per-chip GeoJSON from predicted masks)

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


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _stage(name: str) -> None:
    print(f"\n=== {name} ===")


def _count_files(path: Path, pattern: str) -> int:
    return len(list(path.glob(pattern)))


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

    gdfs = [gpd.read_file(p) for p in geojson_files]
    if len(gdfs) == 1:
        merged = gdfs[0]
    else:
        import pandas as pd

        crs = gdfs[0].crs or "EPSG:4326"
        merged = gpd.GeoDataFrame(
            pd.concat([g.to_crs(crs) for g in gdfs], ignore_index=True),
            crs=crs,
        )
    labels_path = input_dir / "labels.geojson"
    merged.to_file(labels_path, driver="GeoJSON")
    print(f"PASS: prepared {len(tif_files)} PNG chip(s) + labels.geojson from data/sample")

    return work_root


def main() -> None:
    args = parse_args()

    os.environ.setdefault("RAMP_HOME", "/workspace")

    dataset_root = Path(args.dataset_root).resolve()

    # Support data/sample layout (train/oam + train/osm) or legacy (input/ with PNG + labels)
    if (dataset_root / "train" / "oam").is_dir() and (dataset_root / "train" / "osm").is_dir():
        dataset_root = _prepare_data_sample_layout(dataset_root)

    input_dir = dataset_root / "input"
    preprocessed_dir = dataset_root / "preprocessed_test"
    chips_dir = preprocessed_dir / "chips"
    masks_dir = preprocessed_dir / "multimasks"
    val_chips_dir = preprocessed_dir / "val_chips"
    val_masks_dir = preprocessed_dir / "val_multimasks"
    checkpts_dir = preprocessed_dir / "checkpoints"
    pred_input_dir = preprocessed_dir / "chips"
    pred_output_dir = dataset_root / "prediction_test" / "output"
    vectors_dir = dataset_root / "prediction_test" / "vectors"

    # -------------------------------------------------------------------------
    _stage("Test 1: Critical imports")

    import segmentation_models as sm  # noqa: F401
    import tensorflow as tf  # noqa: F401
    from osgeo import gdal  # noqa: F401
    import ramp  # noqa: F401
    import hot_fair_utilities  # noqa: F401

    sm.set_framework("tf.keras")
    print(f"PASS: tensorflow {tf.__version__} / segmentation_models / ramp / gdal imported")

    import solaris  # noqa: F401
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

    # -------------------------------------------------------------------------
    _stage("Test 3: Preprocessing (georeference + multimask generation)")

    shutil.rmtree(preprocessed_dir, ignore_errors=True)

    from hot_fair_utilities import preprocess as _preprocess

    _preprocess(
        input_path=str(input_dir),
        output_path=str(preprocessed_dir),
        rasterize=True,
        rasterize_options=["binary"],
        georeference_images=True,
        multimasks=True,
        input_boundary_width=3,
        input_contact_spacing=8,
    )

    n_chips = _count_files(chips_dir, "*.tif")
    n_masks = _count_files(masks_dir, "*.mask.tif")
    _assert(n_chips > 0, f"No chips produced in {chips_dir}")
    _assert(n_masks > 0, f"No multimasks produced in {masks_dir}")
    _assert(n_chips == n_masks, f"Chip count ({n_chips}) != mask count ({n_masks})")
    print(f"PASS: preprocessing produced {n_chips} chip(s) + {n_masks} multimask(s)")

    # -------------------------------------------------------------------------
    _stage("Test 4: Train / val split")

    import random

    chip_files = sorted(chips_dir.glob("*.tif"))
    n_val = max(1, int(len(chip_files) * 0.2))
    random.shuffle(chip_files)
    val_chip_files = chip_files[:n_val]

    val_chips_dir.mkdir(parents=True, exist_ok=True)
    val_masks_dir.mkdir(parents=True, exist_ok=True)

    moved = 0
    for chip_path in val_chip_files:
        mask_path = masks_dir / (chip_path.stem + ".mask.tif")
        if mask_path.is_file():
            shutil.copy(str(chip_path), val_chips_dir / chip_path.name)
            shutil.copy(str(mask_path), val_masks_dir / mask_path.name)
            moved += 1

    _assert(moved > 0, "Val split produced 0 pairs — check chip/mask naming")
    print(f"PASS: val split created {moved} chip+mask pair(s)")

    # -------------------------------------------------------------------------
    _stage("Test 5: Training smoke run (2 epochs)")

    checkpts_dir.mkdir(parents=True, exist_ok=True)

    import datetime

    import segmentation_models as sm

    sm.set_framework("tf.keras")

    from ramp.data_mgmt.data_generator import (
        test_batches_from_gtiff_dirs,
        training_batches_from_gtiff_dirs,
    )
    from ramp.training import (
        callback_constructors,
        loss_constructors,
        metric_constructors,
        optimizer_constructors,
    )

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    cfg = {
        "experiment_name": "smoke_test",
        "num_classes": 4,
        "num_epochs": args.epochs,
        "batch_size": args.batch_size,
        "input_img_shape": [256, 256],
        "output_img_shape": [256, 256],
        "loss": {"get_loss_fn_name": "get_sparse_categorical_crossentropy_fn", "loss_fn_parms": {}},
        "metrics": {
            "use_metrics": True,
            "get_metrics_fn_names": ["get_sparse_categorical_accuracy_fn"],
            "metrics_fn_parms": [{}],
        },
        "optimizer": {
            "get_optimizer_fn_name": "get_adam_optimizer",
            "optimizer_fn_parms": {"learning_rate": 3e-4},
        },
        "model": {
            "get_model_fn_name": "get_effunet_model",
            "model_fn_parms": {
                "backbone": args.backbone,
                "classes": ["background", "building", "boundary", "contact"],
            },
        },
        "saved_model": {"use_saved_model": False},
        "augmentation": {"use_aug": False},
        "early_stopping": {
            "use_early_stopping": True,
            "early_stopping_parms": {
                "monitor": "val_loss",
                "min_delta": 0.005,
                "patience": 2,
                "verbose": 0,
                "mode": "auto",
                "restore_best_weights": True,
            },
        },
        "cyclic_learning_scheduler": {"use_clr": False},
        "tensorboard": {"use_tb": False},
        "prediction_logging": {"use_prediction_logging": False},
        "model_checkpts": {
            "use_model_checkpts": True,
            "model_checkpts_dir": str(checkpts_dir),
            "get_model_checkpt_callback_fn_name": "get_model_checkpt_callback_fn",
            "model_checkpt_callback_parms": {"mode": "max", "save_best_only": True},
        },
        "random_seed": 42,
        "timestamp": timestamp,
    }

    loss_fn = loss_constructors.get_sparse_categorical_crossentropy_fn(cfg)
    optimizer = optimizer_constructors.get_adam_optimizer(cfg)
    acc_metric = metric_constructors.get_sparse_categorical_accuracy_fn({})
    # NOTE: For the container smoke test we intentionally disable any pretrained
    # EfficientNet weight downloads (the old Keras Applications URL can 404).
    # This keeps the test self-contained while still validating the end-to-end
    # training + checkpoint + inference + polygonization flow.
    the_model = sm.Unet(
        backbone_name=args.backbone,
        encoder_weights=None,
        classes=4,
        activation="softmax",
    )
    the_model.compile(optimizer=optimizer, loss=loss_fn, metrics=[acc_metric])

    n_train = _count_files(chips_dir, "*.tif")
    n_val_c = _count_files(val_chips_dir, "*.tif")
    steps_per_epoch = max(1, n_train // args.batch_size)
    validation_steps = max(1, n_val_c // args.batch_size)
    cfg["runtime"] = {
        "n_training": n_train,
        "n_val": n_val_c,
        "steps_per_epoch": steps_per_epoch,
        "validation_steps": validation_steps,
    }

    train_batches = training_batches_from_gtiff_dirs(
        chips_dir, masks_dir, args.batch_size, [256, 256], [256, 256]
    )
    val_batches = test_batches_from_gtiff_dirs(
        val_chips_dir, val_masks_dir, args.batch_size, [256, 256], [256, 256]
    )

    callbacks = [callback_constructors.get_early_stopping_callback_fn(cfg)]

    the_model.fit(
        train_batches,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_batches,
        validation_steps=validation_steps,
        callbacks=callbacks,
    )

    model_save_path = str(checkpts_dir / f"smoke_{timestamp}.tf")
    the_model.save(model_save_path)
    _assert(Path(model_save_path).is_dir(), f"Saved model not found at {model_save_path}")
    print(f"PASS: 2-epoch training completed; model saved to {model_save_path}")

    # -------------------------------------------------------------------------
    _stage("Test 6: Inference smoke run")

    import numpy as np
    import rasterio as rio

    from ramp.data_mgmt.display_data import get_mask_from_prediction
    from ramp.utils.file_utils import get_basename
    from ramp.utils.img_utils import to_channels_first, to_channels_last

    pred_output_dir.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(pred_output_dir, ignore_errors=True)
    pred_output_dir.mkdir(parents=True, exist_ok=True)

    inference_model = tf.keras.models.load_model(model_save_path, compile=False)

    chip_files = sorted(pred_input_dir.glob("*.tif"))
    for chip_file in chip_files[:3]:
        bname = get_basename(str(chip_file))
        mask_name = bname + ".pred.tif"
        with rio.open(chip_file) as src:
            dst_profile = src.profile.copy()
            dst_profile["count"] = 1
            dst_profile["dtype"] = "uint8"
            img = to_channels_last(src.read()).astype("float32")
            max_val = float(img.max())
            if max_val > 0:
                img = img / max_val
            predicted = get_mask_from_prediction(inference_model.predict(np.expand_dims(img, 0)))
            predicted = np.squeeze(predicted, axis=0)
            with rio.open(pred_output_dir / mask_name, "w", **dst_profile) as dst:
                dst.write(to_channels_first(predicted))

    n_pred = _count_files(pred_output_dir, "*.pred.tif")
    _assert(n_pred > 0, f"No .pred.tif files produced in {pred_output_dir}")
    print(f"PASS: inference produced {n_pred} .pred.tif file(s)")

    # -------------------------------------------------------------------------
    _stage("Test 7: Polygonization")

    from osgeo import gdal

    from ramp.utils.img_utils import gdal_get_mask_tensor
    from ramp.utils.mask_to_vec_utils import (
        binary_mask_from_multichannel_mask,
        binary_mask_to_geojson,
    )

    vectors_dir.mkdir(parents=True, exist_ok=True)

    for pred_tif in sorted(pred_output_dir.glob("*.pred.tif")):
        json_name = pred_tif.stem.replace(".pred", "") + ".geojson"
        json_path = str(vectors_dir / json_name)
        ref_ds = gdal.Open(str(pred_tif))
        _assert(ref_ds is not None, f"GDAL could not open {pred_tif}")
        multimask = gdal_get_mask_tensor(str(pred_tif))
        bin_mask = binary_mask_from_multichannel_mask(multimask)
        binary_mask_to_geojson(bin_mask, ref_ds, json_path)

    n_geojson = _count_files(vectors_dir, "*.geojson")
    _assert(n_geojson > 0, f"No GeoJSON files produced in {vectors_dir}")
    print(f"PASS: polygonization produced {n_geojson} GeoJSON file(s)")

    # -------------------------------------------------------------------------
    _stage("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
