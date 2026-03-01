"""ZenML pipeline for RAMP (EfficientNetB0 + U-Net) building semantic segmentation.

Entrypoints referenced by models/ramp/stac-item.json.
Runtime: ramp-fair (TensorFlow/Keras), hot-fair-utilities (preprocessing only).

Implements the fAIr 3.0 contract (FAIr_3.0_Optimized_Pipeline.md):
  - pre_processing_function  → preprocess()
  - post_processing_function → postprocess()
  - mlm:entrypoint (training) → training_pipeline()
  - inference (model from STAC mlm:model asset href) → inference_pipeline()

Key architecture difference from YOLO packs:
  - Preprocessing  → hot_fair_utilities.preprocess (shared)
  - Training       → ramp.training.* (TF/Keras, NOT ultralytics)
  - Inference      → tf.keras.models.load_model (TF SavedModel)
  - Postprocessing → ramp.utils.mask_to_vec_utils (GDAL polygonize)

Model weights: Backend passes model_uri from STAC Item (assets.model.href).
Supports Google Drive, direct HTTP URLs, S3 (future), and local paths.
Weights are downloaded on first use and cached locally.

All heavy imports are lazy: this module is importable in the fAIr-models
host environment where tensorflow, ramp, and solaris are not installed.
"""

from __future__ import annotations

import datetime
import random
import re
import shutil
import zipfile
from pathlib import Path
from typing import Annotated
from urllib.request import urlretrieve

from annotated_types import Ge, Le
from zenml import log_metadata, pipeline, step

# Cache directory for downloaded models (inside container, typically /workspace)
_DEFAULT_MODEL_CACHE = Path("/workspace/.ramp_model_cache")

# Google Drive folder URL pattern
_GDRIVE_FOLDER_RE = re.compile(
    r"https?://drive\.google\.com/drive/folders/([a-zA-Z0-9_-]+)",
    re.IGNORECASE,
)
_GDRIVE_FILE_RE = re.compile(
    r"https?://drive\.google\.com/file/d/([a-zA-Z0-9_-]+)",
    re.IGNORECASE,
)


def resolve_model_href(
    model_uri: str,
    cache_dir: Path | None = None,
) -> str:
    """Resolve model_uri to a local SavedModel directory path.

    Supports:
      - Local path: returned as-is if it exists and contains saved_model.pb
      - Google Drive folder URL: downloaded via gdown, cached
      - Direct HTTP(S) URL to .zip: downloaded, extracted, cached
      - S3 URLs (s3://...): placeholder for future; raise if not implemented

    Returns the absolute path to the SavedModel directory (contains saved_model.pb).
    """
    cache_dir = cache_dir or _DEFAULT_MODEL_CACHE
    path = Path(model_uri)

    # Local path: must exist and look like a SavedModel dir
    if not (
        model_uri.startswith("http://")
        or model_uri.startswith("https://")
        or model_uri.startswith("s3://")
    ):
        resolved = path.resolve()
        if resolved.is_dir() and (resolved / "saved_model.pb").is_file():
            return str(resolved)
        if resolved.exists():
            return str(resolved)
        raise FileNotFoundError(f"Model path not found or invalid: {resolved}")

    # S3: future support (backend could download before calling)
    if model_uri.startswith("s3://"):
        raise NotImplementedError(
            "S3 model URIs require backend pre-download or boto3 in container. "
            "Use Google Drive or HTTP URL for now."
        )

    # Google Drive folder
    folder_match = _GDRIVE_FOLDER_RE.search(model_uri)
    if folder_match:
        folder_id = folder_match.group(1)
        dest_dir = cache_dir / f"gdrive_{folder_id}"
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Check if already downloaded
        if (dest_dir / "saved_model.pb").is_file():
            return str(dest_dir)

        try:
            import gdown

            gdown.download_folder(
                id=folder_id,
                output=str(dest_dir),
                quiet=True,
            )
        except ImportError as e:
            raise ImportError(
                "gdown is required to download models from Google Drive. "
                "Add 'gdown' to the RAMP Dockerfile."
            ) from e

        if not (dest_dir / "saved_model.pb").is_file():
            # gdown may create a subfolder with the Drive folder name
            subdirs = [d for d in dest_dir.iterdir() if d.is_dir()]
            if len(subdirs) == 1 and (subdirs[0] / "saved_model.pb").is_file():
                return str(subdirs[0])
            raise RuntimeError(
                f"Downloaded folder {dest_dir} does not contain saved_model.pb. "
                "Ensure the Drive folder contains the full Keras SavedModel (saved_model.pb + variables/)."
            )
        return str(dest_dir)

    # Google Drive single file (less common for SavedModel)
    file_match = _GDRIVE_FILE_RE.search(model_uri)
    if file_match:
        file_id = file_match.group(1)
        dest_dir = cache_dir / f"gdrive_file_{file_id}"
        dest_dir.mkdir(parents=True, exist_ok=True)
        out_file = dest_dir / "downloaded"
        try:
            import gdown

            gdown.download(id=file_id, output=str(out_file), quiet=True)
        except ImportError as e:
            raise ImportError("gdown required for Google Drive downloads.") from e
        if out_file.suffix == ".zip":
            with zipfile.ZipFile(out_file, "r") as zf:
                zf.extractall(dest_dir)
            out_file.unlink()
        saved_pb = dest_dir / "saved_model.pb"
        if saved_pb.is_file():
            return str(dest_dir)
        for sub in dest_dir.rglob("saved_model.pb"):
            return str(sub.parent)
        raise RuntimeError(f"Downloaded file did not yield a valid SavedModel in {dest_dir}")

    # Direct HTTP(S) URL to .zip
    if model_uri.lower().endswith(".zip"):
        base_name = Path(model_uri.split("/")[-1]).stem
        dest_dir = cache_dir / base_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        zip_path = cache_dir / (base_name + ".zip")
        if not any(dest_dir.rglob("saved_model.pb")):
            urlretrieve(model_uri, zip_path)
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(dest_dir)
            zip_path.unlink(missing_ok=True)
        for sub in dest_dir.rglob("saved_model.pb"):
            return str(sub.parent)
        raise RuntimeError(f"Zip from {model_uri} did not contain a valid SavedModel")

    raise ValueError(
        f"Unsupported model_uri format: {model_uri}. "
        "Use: local path, Google Drive folder URL, or HTTP(S) URL to a .zip file."
    )


# ---------------------------------------------------------------------------
# STAC MLM processing-expression callables
# ---------------------------------------------------------------------------


def preprocess(
    input_path: str,
    output_path: str,
    boundary_width: int = 3,
    contact_spacing: int = 8,
) -> str:
    """Preprocess OAM chips + labels for RAMP training.

    Step 1 — Georeference PNGs → chips/*.tif (EPSG:3857)
    Step 2 — Reproject + clip labels → labels/*.geojson (per chip)
    Step 3 — Generate 4-channel sparse multimasks → multimasks/*.mask.tif
              Classes: 0=background, 1=building, 2=boundary, 3=contact-point

    Returns the preprocessed output directory path.
    """
    from hot_fair_utilities import preprocess as _preprocess

    _preprocess(
        input_path=input_path,
        output_path=output_path,
        rasterize=True,
        rasterize_options=["binary"],
        georeference_images=True,
        multimasks=True,
        input_boundary_width=boundary_width,
        input_contact_spacing=contact_spacing,
    )
    return output_path


def postprocess(prediction_masks_dir: str, output_dir: str) -> str:
    """Convert RAMP multichannel predicted masks to per-chip GeoJSON polygons.

    For each .pred.tif in prediction_masks_dir:
      1. Reads the 4-class sparse multimask (uint8, channels-first, 1 band).
      2. Extracts a binary building footprint mask (class == 1, ignores boundary/contact).
      3. Polygonizes via GDAL and writes a matching .geojson file.

    Returns the output_dir path containing per-chip GeoJSONs.
    """
    from osgeo import gdal

    from ramp.utils.img_utils import gdal_get_mask_tensor
    from ramp.utils.mask_to_vec_utils import (
        binary_mask_from_multichannel_mask,
        binary_mask_to_geojson,
    )

    pred_dir = Path(prediction_masks_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_tifs = sorted(pred_dir.glob("*.pred.tif"))
    if not pred_tifs:
        raise RuntimeError(f"No *.pred.tif files found in {pred_dir}")

    for mask_path in pred_tifs:
        json_name = mask_path.stem.replace(".pred", "") + ".geojson"
        json_path = str(out_dir / json_name)

        ref_ds = gdal.Open(str(mask_path))
        if ref_ds is None:
            raise RuntimeError(f"GDAL could not open {mask_path}")

        multimask = gdal_get_mask_tensor(str(mask_path))
        bin_mask = binary_mask_from_multichannel_mask(multimask)
        binary_mask_to_geojson(bin_mask, ref_ds, json_path)

    return output_dir


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _make_val_split(
    chips_dir: Path,
    masks_dir: Path,
    val_chips_dir: Path,
    val_masks_dir: Path,
    val_fraction: float = 0.15,
) -> None:
    """Move val_fraction of chip+mask pairs to validation directories.

    RAMP mask filenames follow the convention <chip_stem>.mask.tif so we
    match chips to their masks by stem before moving.
    """
    chip_files = sorted(chips_dir.glob("*.tif"))
    if not chip_files:
        raise RuntimeError(f"No .tif chips found in {chips_dir}")

    n_val = max(1, int(len(chip_files) * val_fraction))
    random.shuffle(chip_files)
    val_chips = chip_files[:n_val]

    val_chips_dir.mkdir(parents=True, exist_ok=True)
    val_masks_dir.mkdir(parents=True, exist_ok=True)

    moved = 0
    for chip_path in val_chips:
        mask_path = masks_dir / (chip_path.stem + ".mask.tif")
        if mask_path.is_file():
            shutil.move(str(chip_path), val_chips_dir / chip_path.name)
            shutil.move(str(mask_path), val_masks_dir / mask_path.name)
            moved += 1

    if moved == 0:
        raise RuntimeError(
            f"Val split produced 0 pairs. "
            f"Check that chips in {chips_dir} match masks in {masks_dir}."
        )


def _build_train_config(
    chips_subdir: str,
    masks_subdir: str,
    val_chips_subdir: str,
    val_masks_subdir: str,
    checkpts_subdir: str,
    backbone: str,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    early_stopping_patience: int,
    timestamp: str,
) -> dict:
    """Build a RAMP JSON training config dict from pipeline parameters."""
    return {
        "experiment_name": "RAMP EffUnet multimask training",
        "discard_experiment": False,
        "logging": {"log_experiment": False},
        "datasets": {
            "train_img_dir": chips_subdir,
            "train_mask_dir": masks_subdir,
            "val_img_dir": val_chips_subdir,
            "val_mask_dir": val_masks_subdir,
        },
        "num_classes": 4,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "input_img_shape": [256, 256],
        "output_img_shape": [256, 256],
        "loss": {
            "get_loss_fn_name": "get_sparse_categorical_crossentropy_fn",
            "loss_fn_parms": {},
        },
        "metrics": {
            "use_metrics": True,
            "get_metrics_fn_names": ["get_sparse_categorical_accuracy_fn"],
            "metrics_fn_parms": [{}],
        },
        "optimizer": {
            "get_optimizer_fn_name": "get_adam_optimizer",
            "optimizer_fn_parms": {"learning_rate": learning_rate},
        },
        "model": {
            "get_model_fn_name": "get_effunet_model",
            "model_fn_parms": {
                "backbone": backbone,
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
                "patience": early_stopping_patience,
                "verbose": 1,
                "mode": "auto",
                "restore_best_weights": True,
            },
        },
        "cyclic_learning_scheduler": {"use_clr": False},
        "tensorboard": {"use_tb": False},
        "prediction_logging": {"use_prediction_logging": False},
        "model_checkpts": {
            "use_model_checkpts": True,
            "model_checkpts_dir": checkpts_subdir,
            "get_model_checkpt_callback_fn_name": "get_model_checkpt_callback_fn",
            "model_checkpt_callback_parms": {"mode": "max", "save_best_only": True},
        },
        "random_seed": 20220523,
        "timestamp": timestamp,
    }


# ---------------------------------------------------------------------------
# ZenML steps
# ---------------------------------------------------------------------------


@step
def run_preprocessing(
    input_path: str,
    output_path: str,
    boundary_width: int = 3,
    contact_spacing: int = 8,
) -> str:
    """Georeference OAM chips and generate 4-class multimasks. Returns preprocessed dir."""
    return preprocess(input_path, output_path, boundary_width, contact_spacing)


@step
def train_model(
    data_base_path: str,
    preprocessed_path: str,
    backbone: str = "efficientnetb0",
    num_epochs: int = 100,
    batch_size: int = 16,
    learning_rate: float = 3e-4,
    early_stopping_patience: int = 35,
    val_fraction: float = 0.15,
) -> str:
    """Fine-tune EfficientNetB0 + U-Net on 4-class multimask chips.

    1. Splits preprocessed chips/masks into train and val sets.
    2. Builds the EfficientNet-B0 U-Net from segmentation_models.
    3. Trains with sparse categorical crossentropy loss.
    4. Returns the path to the best Keras SavedModel directory.

    val_sparse_categorical_accuracy is logged as ZenML step metadata.
    """
    import os

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
        model_constructors,
        optimizer_constructors,
    )
    from ramp.utils.model_utils import get_best_model_value_and_epoch

    os.environ["RAMP_HOME"] = data_base_path

    pre_path = Path(preprocessed_path)
    chips_dir = pre_path / "chips"
    masks_dir = pre_path / "multimasks"
    val_chips_dir = pre_path / "val_chips"
    val_masks_dir = pre_path / "val_multimasks"
    checkpts_dir = pre_path / "checkpoints"

    if not val_chips_dir.is_dir():
        _make_val_split(chips_dir, masks_dir, val_chips_dir, val_masks_dir, val_fraction)

    def _rel(d: Path) -> str:
        return str(d.relative_to(data_base_path))

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    cfg = _build_train_config(
        chips_subdir=_rel(chips_dir),
        masks_subdir=_rel(masks_dir),
        val_chips_subdir=_rel(val_chips_dir),
        val_masks_subdir=_rel(val_masks_dir),
        checkpts_subdir=_rel(checkpts_dir),
        backbone=backbone,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        early_stopping_patience=early_stopping_patience,
        timestamp=timestamp,
    )

    loss_fn = loss_constructors.get_sparse_categorical_crossentropy_fn(cfg)
    optimizer = optimizer_constructors.get_adam_optimizer(cfg)
    accuracy_metric = metric_constructors.get_sparse_categorical_accuracy_fn({})
    the_model = model_constructors.get_effunet_model(cfg)
    the_model.compile(optimizer=optimizer, loss=loss_fn, metrics=[accuracy_metric])

    n_train = len(list(chips_dir.glob("*.tif")))
    n_val = len(list(val_chips_dir.glob("*.tif")))
    steps_per_epoch = max(1, n_train // batch_size)
    validation_steps = max(1, n_val // batch_size)
    cfg["runtime"] = {
        "n_training": n_train,
        "n_val": n_val,
        "steps_per_epoch": steps_per_epoch,
        "validation_steps": validation_steps,
    }

    img_shape = cfg["input_img_shape"]
    mask_shape = cfg["output_img_shape"]

    train_batches = training_batches_from_gtiff_dirs(
        chips_dir, masks_dir, batch_size, img_shape, mask_shape
    )
    val_batches = test_batches_from_gtiff_dirs(
        val_chips_dir, val_masks_dir, batch_size, img_shape, mask_shape
    )

    callbacks = [
        callback_constructors.get_early_stopping_callback_fn(cfg),
        callback_constructors.get_model_checkpt_callback_fn(cfg),
    ]

    history = the_model.fit(
        train_batches,
        epochs=num_epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_batches,
        validation_steps=validation_steps,
        callbacks=callbacks,
    )

    best_epoch, best_val_acc = get_best_model_value_and_epoch(history)
    log_metadata(
        metadata={
            "best_val_sparse_categorical_accuracy": float(best_val_acc),
            "best_epoch": int(best_epoch),
        }
    )

    checkpts_ts_dir = checkpts_dir / timestamp
    checkpoints = sorted(checkpts_ts_dir.glob("*.tf"))
    if not checkpoints:
        raise RuntimeError(f"No .tf checkpoint found in {checkpts_ts_dir}")
    return str(checkpoints[-1])


@step
def run_inference(
    model_uri: str,
    input_path: str,
    prediction_path: str,
    model_cache_dir: str | None = None,
) -> str:
    """Run RAMP EfficientNetB0 U-Net inference on georeferenced chips.

    model_uri: From STAC Item (assets.model.href). Can be:
      - Local path to SavedModel directory
      - Google Drive folder URL
      - HTTP(S) URL to .zip containing SavedModel
    Resolves URLs to local path (downloads and caches if needed).

    Loads Keras SavedModel, runs prediction chip-by-chip, and writes
    one .pred.tif (single-band uint8 sparse mask) per input chip.

    Returns prediction_path containing the .pred.tif files.
    """
    import numpy as np
    import rasterio as rio
    import tensorflow as tf
    from tqdm import tqdm

    from ramp.data_mgmt.display_data import get_mask_from_prediction
    from ramp.utils.file_utils import get_basename
    from ramp.utils.img_utils import to_channels_first, to_channels_last

    cache = Path(model_cache_dir) if model_cache_dir else None
    model_dir = resolve_model_href(model_uri, cache_dir=cache)
    model = tf.keras.models.load_model(model_dir, compile=False)
    out_dir = Path(prediction_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    chip_files = sorted(Path(input_path).glob("**/*.tif"))
    if not chip_files:
        png_files = sorted(Path(input_path).glob("**/*.png"))
        if png_files:
            raise RuntimeError(
                "No GeoTIFF chips (*.tif) found for inference. "
                f"Found {len(png_files)} PNG(s) in {input_path}. "
                "RAMP inference expects georeferenced RGB GeoTIFF chips (typically the output of "
                "`preprocess(...)` under `<preprocessed>/chips/`). "
                "Run preprocessing and point `input_path` to the resulting `chips/` directory."
            )
        raise RuntimeError(
            f"No GeoTIFF chips (*.tif) found in {input_path}. "
            "RAMP inference expects georeferenced RGB GeoTIFF chips (typically under `<preprocessed>/chips/`)."
        )

    for chip_file in tqdm(chip_files, desc="RAMP inference"):
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
            predicted = get_mask_from_prediction(model.predict(np.expand_dims(img, 0)))
            predicted = np.squeeze(predicted, axis=0)
            with rio.open(out_dir / mask_name, "w", **dst_profile) as dst:
                dst.write(to_channels_first(predicted))

    return prediction_path


@step
def run_postprocessing(
    prediction_path: str,
    output_dir: str,
) -> str:
    """Polygonize RAMP predicted masks into per-chip building GeoJSONs."""
    return postprocess(prediction_path, output_dir)


# ---------------------------------------------------------------------------
# ZenML pipelines
# ---------------------------------------------------------------------------


@pipeline
def training_pipeline(
    input_path: str,
    output_path: str,
    backbone: str = "efficientnetb0",
    num_epochs: Annotated[int, Ge(1), Le(2000)] = 100,
    batch_size: Annotated[int, Ge(1), Le(64)] = 16,
    learning_rate: Annotated[float, Ge(1e-6), Le(1e-2)] = 3e-4,
    early_stopping_patience: Annotated[int, Ge(5), Le(100)] = 35,
    val_fraction: Annotated[float, Ge(0.05), Le(0.4)] = 0.15,
    boundary_width: int = 3,
    contact_spacing: int = 8,
) -> None:
    """Full RAMP training: georeference + multimask → val split → EfficientNetB0 U-Net."""
    preprocessed_path = run_preprocessing(
        input_path=input_path,
        output_path=f"{output_path}/preprocessed",
        boundary_width=boundary_width,
        contact_spacing=contact_spacing,
    )
    train_model(
        data_base_path=output_path,
        preprocessed_path=preprocessed_path,
        backbone=backbone,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        early_stopping_patience=early_stopping_patience,
        val_fraction=val_fraction,
    )


@pipeline
def inference_pipeline(
    model_uri: str,
    input_path: str,
    prediction_path: str,
    output_dir: str,
    model_cache_dir: str | None = None,
) -> None:
    """RAMP inference: load model (from STAC) → predict → polygonize to GeoJSONs.

    model_uri: From STAC Item assets.model.href. Supports:
      - Local path (e.g. /workspace/checkpoints/model.tf)
      - Google Drive folder URL
      - HTTP URL to .zip containing SavedModel
    """
    pred_path = run_inference(
        model_uri=model_uri,
        input_path=input_path,
        prediction_path=prediction_path,
        model_cache_dir=model_cache_dir,
    )
    run_postprocessing(prediction_path=pred_path, output_dir=output_dir)
