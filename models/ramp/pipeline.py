"""ZenML pipeline for RAMP (EfficientNetB0 + U-Net) building semantic segmentation.

Entrypoints referenced by stac-item.json.
Runtime: ramp-fair (TensorFlow/Keras), hot-fair-utilities (preprocessing + training).

Implements the fAIr entrypoints:
  - pre_processing_function  → preprocess()
  - post_processing_function → postprocess()
  - mlm:entrypoint (training) → training_pipeline()
  - inference (model from STAC mlm:model asset href) → inference_pipeline()

Model weights: Backend passes model_uri from STAC Item (assets.model.href).
Supports direct HTTP(S) URLs to .zip archives and local paths.
Google Drive is not supported; weights should be published to HTTP or staged locally.

Inference/postprocess use the **fairpredictor** PyPI distribution; its importable top-level
package is ``predictor`` (``pip install fairpredictor`` → ``import predictor``), same as
``hot_fair_utilities.inference.predict``.

All heavy imports are lazy: this module is importable in the fAIr-models
host environment where tensorflow, ramp, and solaris are not installed.
"""

import re
import zipfile
from pathlib import Path
from shutil import copy2, rmtree
from typing import Annotated, Any, Dict, List, Optional, Tuple, Union
from urllib.request import urlretrieve

from zenml import log_metadata, pipeline, step

_DEFAULT_MODEL_CACHE = Path("/workspace/.ramp_model_cache")
_DEFAULT_RAMP_BASELINE_URL = (
    "https://api-prod.fair.hotosm.org/api/v1/workspace/download/ramp/baseline.zip"
)


def _download_and_extract_zip(zip_url: str, dest_dir: Path) -> None:
    """Download a ZIP URL and extract in dest_dir."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_name = Path(zip_url.split("/")[-1]).name or "archive.zip"
    zip_path = dest_dir / zip_name
    urlretrieve(zip_url, zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)
    zip_path.unlink(missing_ok=True)


def _to_local_path(path_value: str, purpose: str) -> Path:
    """Resolve a path with UPath and ensure local filesystem semantics."""
    from upath import UPath

    upath_obj = UPath(path_value)
    protocol = getattr(upath_obj, "protocol", "") or ""
    if protocol not in ("", "file"):
        raise NotImplementedError(
            f"{purpose} requires a local filesystem path. "
            f"Received protocol={protocol!r} for {path_value!r}."
        )
    return Path(str(upath_obj))


def _resolve_input_directory(path_value: str, purpose: str) -> Path:
    """Resolve local or remote (S3/HTTP/etc.) directories to a local path."""
    if "://" in str(path_value):
        # Available in the runtime (same helper used by YOLO model packs).
        from fair.utils.data import resolve_directory

        return Path(str(resolve_directory(path_value, pattern="*")))
    return _to_local_path(path_value, purpose)


def _resolve_input_file(path_value: str, purpose: str) -> Path:
    """Resolve local or remote (S3/HTTP/etc.) files to a local path."""
    if "://" in str(path_value):
        from fair.utils.data import resolve_path

        return Path(str(resolve_path(path_value)))
    return _to_local_path(path_value, purpose)


def _extract_zip(zip_path: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)


def _ensure_ramp_baseline(data_base_path: str, baseline_rel_path: str) -> Path:
    """Return the directory that contains baseline weights; download under data_base_path if missing.

    ``baseline_rel_path`` is e.g. ``ramp-data/baseline/checkpoint.tf`` (file relative to RAMP_HOME).
    """
    rel = Path(baseline_rel_path)
    local_dir = Path(data_base_path) / rel.parent
    local_ck = local_dir / rel.name
    if local_ck.is_file() or (local_dir / "saved_model.pb").exists():
        return local_dir
    image_ck = Path("/app") / baseline_rel_path
    if image_ck.is_file():
        return image_ck.parent
    _download_and_extract_zip(_DEFAULT_RAMP_BASELINE_URL, local_dir)
    return local_dir


_QUBVEL_EFFICIENTNET_RELEASE = (
    "https://github.com/qubvel/efficientnet/releases/download/v0.0.1/"
)


def _patch_keras_get_file_for_efficientnet_weights() -> None:
    """Redirect broken Callidior EfficientNet weight URLs to qubvel's GitHub release assets.

    The ``efficientnet`` package (via ``segmentation_models``) downloads encoder weights from
    ``github.com/Callidior/keras-applications/releases/...``; those assets now return **404**.
    qubvel hosts compatible ``*_imagenet_1000_notop.h5`` files on the same model's releases.
    """
    import tensorflow as tf

    ku = tf.keras.utils
    if getattr(ku.get_file, "_ramp_efficientnet_mirror", False):
        return

    _orig = ku.get_file

    def _get_file(fname, origin, *args, **kwargs):
        if isinstance(origin, str) and "Callidior" in origin and isinstance(fname, str):
            m = re.match(
                r"^(efficientnet-b\d+)_weights_tf_dim_ordering_tf_kernels_autoaugment_notop\.h5$",
                fname,
            )
            if m:
                alt = f"{m.group(1)}_imagenet_1000_notop.h5"
                origin = f"{_QUBVEL_EFFICIENTNET_RELEASE}{alt}"
                kwargs = dict(kwargs)
                kwargs["file_hash"] = None
        return _orig(fname, origin, *args, **kwargs)

    _get_file._ramp_efficientnet_mirror = True  # type: ignore[attr-defined]
    ku.get_file = _get_file


def _patch_predictor_savedmodel_loader_for_tf215() -> None:
    """Patch fairpredictor's SavedModel directory loader for TF 2.15 compatibility.

    Why this patch exists:
    Some fairpredictor versions load SavedModel directories using `keras.layers.TFSMLayer(...)`.
    `TFSMLayer` is not available in TensorFlow 2.15's `tf.keras.layers`, so inference fails with:
      AttributeError: module 'keras.api._v2.keras.layers' has no attribute 'TFSMLayer'

    We monkey-patch `predictor.prediction._load_keras_model` so directory-based SavedModels use
    `tf.saved_model.load(...)` and a small `.predict(...)` wrapper.

    We patch the module object (via importlib) to ensure downstream `from predictor.prediction import ...`
    uses the patched function.
    """
    import importlib
    import os
    from pathlib import Path

    import tensorflow as tf

    pred = importlib.import_module("predictor.prediction")
    if getattr(pred, "_fair_models_tf215_savedmodel_patch_applied", False):
        return

    original_loader = getattr(pred, "_load_keras_model", None)
    if original_loader is None:
        raise RuntimeError("predictor.prediction._load_keras_model not found; fairpredictor API changed.")

    def _safe_load_keras_model(keras_backend, path: str):
        if os.path.isdir(path) and (Path(path) / "saved_model.pb").exists():
            # TF 2.15 (Keras 2.x) can load SavedModel directories directly; avoid TFSMLayer entirely.
            return tf.keras.models.load_model(path, compile=False)

        return original_loader(keras_backend, path)

    _safe_load_keras_model._fair_models_tf215_savedmodel_loader = True  # type: ignore[attr-defined]
    pred._load_keras_model = _safe_load_keras_model
    pred._fair_models_tf215_savedmodel_patch_applied = True


def resolve_model_href(
    model_uri: str,
    cache_dir: Optional[Path] = None,
) -> str:
    """Resolve model_uri to a local SavedModel directory path.

    Supports:
      - Local path: returned as-is if it exists
      - Direct HTTP(S) URL to .zip: downloaded, extracted, cached

    Returns the absolute path to the SavedModel directory.
    """
    if not isinstance(model_uri, str):
        raise TypeError("model_uri must be a string")
    if cache_dir is not None and not isinstance(cache_dir, Path):
        raise TypeError("cache_dir must be a pathlib.Path or None")

    cache_dir = cache_dir or _DEFAULT_MODEL_CACHE

    is_http = model_uri.startswith("http://") or model_uri.startswith("https://")

    # SavedModel directory (local or remote)
    if not Path(model_uri.split("?", 1)[0]).suffix and not model_uri.lower().endswith(".zip"):
        resolved_dir = (_resolve_input_directory(model_uri, "model_uri") if "://" in model_uri else _to_local_path(model_uri, "model_uri")).resolve()
        if (resolved_dir / "saved_model.pb").exists():
            return str(resolved_dir)
        if resolved_dir.exists():
            # Let downstream error messages be explicit.
            raise FileNotFoundError(f"SavedModel directory missing saved_model.pb: {resolved_dir}")
        raise FileNotFoundError(f"Model path not found: {resolved_dir}")

    # Remote/local ZIP containing SavedModel
    if model_uri.lower().endswith(".zip"):
        base_name = Path(model_uri.split("?", 1)[0]).name
        stem = Path(base_name).stem or "ramp_model"
        dest_dir = cache_dir / stem
        if any(dest_dir.rglob("saved_model.pb")):
            for sub in dest_dir.rglob("saved_model.pb"):
                return str(sub.parent)

        if is_http:
            _download_and_extract_zip(model_uri, dest_dir)
        else:
            local_zip = _resolve_input_file(model_uri, "model_uri")
            _extract_zip(local_zip, dest_dir)

        for sub in dest_dir.rglob("saved_model.pb"):
            return str(sub.parent)
        raise RuntimeError(f"Zip from {model_uri} did not contain a valid SavedModel")

    # Plain local path to file/dir (fallback)
    resolved = _to_local_path(model_uri, "model_uri").resolve()
    if resolved.exists():
        return str(resolved)
    raise FileNotFoundError(f"Model path not found: {resolved}")

    raise ValueError(
        f"Unsupported model_uri: {model_uri}. "
        "Use a local path or an HTTP(S) URL to a .zip file containing the SavedModel."
    )


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

    local_input = _resolve_input_directory(input_path, "input_path")
    _preprocess(
        input_path=str(local_input),
        output_path=output_path,
        rasterize=True,
        rasterize_options=["binary"],
        georeference_images=True,
        multimasks=True,
        input_boundary_width=boundary_width,
        input_contact_spacing=contact_spacing,
    )
    return output_path


def postprocess(prediction_masks_dir: str, output_dir: str) -> dict:
    """Run fairpredictor-style postprocessing and return merged prediction GeoJSON content."""
    import json

    # geomltoolkits>=2 moved modules (no geomltoolkits.regularizer/utils)
    from geomltoolkits.geometry.validate import validate_polygon_geometries
    from geomltoolkits.raster.merge import merge_rasters
    from geomltoolkits.raster.morphology import morphological_cleaning
    from geomltoolkits.raster.vectorize import vectorize_mask

    pred_dir = _to_local_path(prediction_masks_dir, "prediction_masks_dir")
    out_dir = _to_local_path(output_dir, "output_dir")
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_tifs = sorted(pred_dir.glob("*.tif"))
    if not pred_tifs:
        raise RuntimeError(f"No *.tif files found in {pred_dir}")

    merged_mask_path = out_dir / "merged_prediction_mask.tif"
    merged_geojson_path = out_dir / "predictions.geojson"

    merge_rasters(str(pred_dir), str(merged_mask_path))
    morphological_cleaning(str(merged_mask_path))
    gdf = vectorize_mask(
        input_tiff=str(merged_mask_path),
        output_geojson=str(merged_geojson_path),
        simplify_tolerance=0.5,
        min_area=3,
        orthogonalize=True,
        ortho_skew_tolerance_deg=15,
        ortho_max_angle_change_deg=15,
    )

    geojson_dict = json.loads(gdf.to_json())
    if not geojson_dict.get("features"):
        # geomltoolkits validator raises on empty FeatureCollections; treat this as a valid
        # "no buildings found" prediction and keep the already-written GeoJSON file.
        if not merged_geojson_path.is_file():
            with merged_geojson_path.open("w", encoding="utf-8") as fh:
                json.dump(geojson_dict, fh)
        return geojson_dict

    if gdf.crs and gdf.crs != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")
    elif not gdf.crs:
        gdf.set_crs("EPSG:3857", inplace=True)
        gdf = gdf.to_crs("EPSG:4326")
    # Pandas extension dtypes break some Fiona GeoJSON writes; coerce to object.
    import pandas as pd

    for col in gdf.columns:
        if col == "geometry":
            continue
        if pd.api.types.is_extension_array_dtype(gdf[col].dtype):
            gdf[col] = gdf[col].astype(object).where(gdf[col].notna(), None)
    gdf.to_file(merged_geojson_path, driver="GeoJSON")

    validated_geojson = validate_polygon_geometries(
        geojson_dict,
        output_path=str(merged_geojson_path),
    )
    if isinstance(validated_geojson, str) and Path(validated_geojson).is_file():
        if Path(validated_geojson) != merged_geojson_path:
            copy2(validated_geojson, merged_geojson_path)
        with merged_geojson_path.open("r", encoding="utf-8") as file_handle:
            return json.load(file_handle)

    if isinstance(validated_geojson, dict):
        return validated_geojson
    return json.loads(gdf.to_json())


def _load_hyperparams_from_stac(stac_item_path: str) -> dict:
    """Load mlm:hyperparameters from a STAC Item JSON file.

    Relative paths are resolved against /workspace (container) or cwd (local).
    """
    import json

    path = _to_local_path(stac_item_path, "stac_item_path")
    if not path.is_absolute():
        for base in (Path("/workspace"), Path.cwd()):
            candidate = base / path
            if candidate.is_file():
                path = candidate
                break
        else:
            path = Path("/workspace") / path
    with open(path, encoding="utf-8") as f:
        item = json.load(f)
    return dict(item.get("properties", {}).get("mlm:hyperparameters", {}))


@step
def run_preprocessing(
    input_path: str,
    output_path: str,
    boundary_width: int = 3,
    contact_spacing: int = 8,
) -> List[Tuple[Any, Any]]:
    """Georeference OAM chips and return chip/mask data arrays."""
    import rasterio

    preprocessed = Path(preprocess(input_path, output_path, boundary_width, contact_spacing))
    chips_dir = preprocessed / "chips"
    masks_dir = preprocessed / "multimasks"
    data_loader: list[tuple[Any, Any]] = []
    for chip in sorted(chips_dir.glob("*.tif")):
        mask = masks_dir / f"{chip.stem}.mask.tif"
        if mask.is_file():
            with rasterio.open(chip) as chip_src:
                chip_data = chip_src.read()
            with rasterio.open(mask) as mask_src:
                mask_data = mask_src.read()
            data_loader.append((chip_data, mask_data))
    return data_loader


@step
def train_model(
    data_base_path: str,
    preprocessed_path: str,
    data_loader: Optional[List[Tuple[Any, Any]]] = None,
    stac_item_path: str = "models/ramp/stac-item.json",
    val_fraction: Optional[float] = None,
    split_info: Optional[Dict[str, Any]] = None,
) -> Any:
    """ZenML step wrapper for RAMP training; returns the best Keras SavedModel checkpoint."""
    if data_loader is not None and not data_loader:
        raise RuntimeError("Preprocessing returned an empty dataloader; no chip/mask pairs found.")
    return train_ramp_model(
        data_base_path=data_base_path,
        preprocessed_path=preprocessed_path,
        stac_item_path=stac_item_path,
        val_fraction=val_fraction,
        split_info=split_info,
        log_zenml_step_metadata=True,
    )


@step
def split_dataset(
    preprocessed_path: str,
    output_path: str,
    hyperparameters: Optional[Dict[str, Any]] = None,
) -> Annotated[Dict[str, Any], "split_info"]:
    """Create train/validation directories for RAMP and return split metadata.

    Uses the existing utilities implementation:
    `hot_fair_utilities.training.ramp.prepare_data.split_training_2_validation`.
    """
    from hot_fair_utilities.training.ramp.prepare_data import split_training_2_validation

    hyperparameters = hyperparameters or {}
    preprocessed_dir = _to_local_path(preprocessed_path, "preprocessed_path")
    if not preprocessed_dir.is_dir():
        raise FileNotFoundError(f"Preprocessed directory not found: {preprocessed_dir}")

    out_dir = _to_local_path(output_path, "output_path")
    ramp_train_dir = out_dir / "ramp_training_work"

    split_training_2_validation(
        str(preprocessed_dir),
        str(ramp_train_dir),
        multimasks=True,
    )

    train_count = len(list((ramp_train_dir / "chips").glob("*.tif")))
    val_count = len(list((ramp_train_dir / "val-chips").glob("*.tif")))
    total = train_count + val_count
    val_ratio = (float(val_count) / float(total)) if total else 0.0

    split_info: Dict[str, Any] = {
        "strategy": "random",
        "val_ratio": val_ratio,
        "train_count": train_count,
        "val_count": val_count,
        "seed": int(hyperparameters.get("split_seed", 42)),
        "ramp_train_dir": str(ramp_train_dir),
        "source": "hot_fair_utilities.training.ramp.prepare_data.split_training_2_validation",
    }
    log_metadata(metadata={"fair/split": split_info})
    return split_info


def train_ramp_model(
    data_base_path: str,
    preprocessed_path: str,
    stac_item_path: str = "models/ramp/stac-item.json",
    val_fraction: Annotated[Optional[float], "0.0 <= val_fraction <= 0.5"] = None,
    num_epochs: Annotated[Optional[int], "1 <= num_epochs <= 20"] = None,
    batch_size: Annotated[Optional[int], "1 <= batch_size <= 8"] = None,
    backbone: Optional[str] = None,
    early_stopping_patience: Annotated[Optional[int], "1 <= early_stopping_patience <= 20"] = None,
    split_info: Optional[Dict[str, Any]] = None,
    log_zenml_step_metadata: bool = False,
) -> Any:
    """Fine-tune EfficientNetB0 + U-Net on 4-class multimask chips.

    Uses hot_fair_utilities.training.ramp for training orchestration.
    RAMP_CONFIG is used as the base configuration; hyperparameters from the
    STAC Item and keyword arguments selectively override the base.

    Val split is handled internally by split_training_2_validation:
    preprocessed_path → ramp_training_work/ (train + val-chips + val-multimasks).

    Sets ``RAMP_HOME`` before importing training helpers: ``run_training`` caches
    ``working_ramp_home`` at import time, so ``/app`` is used when the GHCR baseline exists there.

    If the RAMP baseline exists under the hot-fair-utilities image (``/app/ramp-data/baseline/``)
    it is used for fine-tuning. Otherwise weights are resolved under ``data_base_path`` or downloaded.

    Returns the best checkpoint as a loaded ``tf.keras.Model`` (SavedModel on disk is loaded with compile=False).
    """
    import os

    # run_training.run_training sets ``working_ramp_home = os.environ["RAMP_HOME"]`` at *import* time.
    # That value is used for ``Path(working_ramp_home) / saved_model_path`` when loading the baseline.
    # Dataset paths in cfg are absolute (from manage_fine_tuning_config), so they still resolve correctly.
    # Docker + GHCR base: baseline lives under /app; do not rely on /workspace/ramp-data (bind mount hides it).
    resolved_base = str(Path(data_base_path).resolve())
    image_baseline_ck = Path("/app/ramp-data/baseline/checkpoint.tf")
    if image_baseline_ck.is_file():
        os.environ["RAMP_HOME"] = "/app"
    else:
        os.environ["RAMP_HOME"] = resolved_base

    # segmentation_models configures efficientnet at import time via SM_FRAMEWORK.
    os.environ.setdefault("SM_FRAMEWORK", "tf.keras")
    import tensorflow as tf

    _patch_keras_get_file_for_efficientnet_weights()
    import segmentation_models as sm
    from hot_fair_utilities.training.ramp.cleanup import extract_highest_accuracy_model
    from hot_fair_utilities.training.ramp.config import RAMP_CONFIG
    from hot_fair_utilities.training.ramp.run_training import (
        manage_fine_tuning_config,
        run_main_train_code,
    )

    sm.set_framework("tf.keras")

    # Load STAC hyperparams and apply call-site overrides
    hyperparams = _load_hyperparams_from_stac(stac_item_path)
    if val_fraction is not None:
        if not 0.0 <= val_fraction <= 0.5:
            raise ValueError("val_fraction must be in [0.0, 0.5]")
        hyperparams["val_fraction"] = val_fraction
    if num_epochs is not None:
        if not 1 <= num_epochs <= 20:
            raise ValueError("num_epochs must be in [1, 20] for RAMP runtime limits")
        hyperparams["num_epochs"] = num_epochs
    if batch_size is not None:
        if not 1 <= batch_size <= 8:
            raise ValueError("batch_size must be in [1, 8] for RAMP runtime limits")
        hyperparams["batch_size"] = batch_size
    if backbone is not None:
        hyperparams["backbone"] = backbone
    if early_stopping_patience is not None:
        if not 1 <= early_stopping_patience <= 20:
            raise ValueError("early_stopping_patience must be in [1, 20]")
        hyperparams["early_stopping_patience"] = early_stopping_patience

    # Resolve effective values from STAC overrides or RAMP_CONFIG defaults
    eff_epochs = hyperparams.get("num_epochs", hyperparams.get("epochs", RAMP_CONFIG["num_epochs"]))
    eff_batch = hyperparams.get("batch_size", RAMP_CONFIG["batch_size"])
    eff_backbone = hyperparams.get("backbone", RAMP_CONFIG["model"]["model_fn_parms"]["backbone"])
    eff_lr = hyperparams.get("learning_rate", RAMP_CONFIG["optimizer"]["optimizer_fn_parms"]["learning_rate"])
    eff_patience = hyperparams.get(
        "early_stopping_patience",
        RAMP_CONFIG["early_stopping"]["early_stopping_parms"]["patience"],
    )
    if not 1 <= int(eff_epochs) <= 20:
        raise ValueError(f"Resolved num_epochs={eff_epochs} is outside [1, 20]")
    if not 1 <= int(eff_batch) <= 8:
        raise ValueError(f"Resolved batch_size={eff_batch} is outside [1, 8]")
    if not 1 <= int(eff_patience) <= 20:
        raise ValueError(f"Resolved early_stopping_patience={eff_patience} is outside [1, 20]")

    # Prefer a precomputed split (CI-visible split_dataset step), fallback to internal split for compatibility.
    if split_info and split_info.get("ramp_train_dir"):
        ramp_train_dir = str(_to_local_path(str(split_info["ramp_train_dir"]), "ramp_train_dir"))
    else:
        from hot_fair_utilities.training.ramp.prepare_data import split_training_2_validation

        ramp_train_dir_path = _to_local_path(
            str(Path(data_base_path) / "ramp_training_work"),
            "ramp_training_work",
        )
        if ramp_train_dir_path.exists():
            rmtree(ramp_train_dir_path)
        ramp_train_dir = str(ramp_train_dir_path)
        split_training_2_validation(
            str(_to_local_path(preprocessed_path, "preprocessed_path")),
            ramp_train_dir,
            multimasks=True,
        )

    # Build config from RAMP_CONFIG via the utilities helper, then apply overrides
    cfg = manage_fine_tuning_config(ramp_train_dir, eff_epochs, eff_batch, freeze_layers=False, multimasks=True)
    cfg["model"]["model_fn_parms"]["backbone"] = eff_backbone
    cfg["optimizer"]["optimizer_fn_parms"]["learning_rate"] = eff_lr
    cfg["early_stopping"]["early_stopping_parms"]["patience"] = eff_patience

    # Use baseline checkpoint for fine-tuning if present, else train from scratch
    saved_rel = cfg["saved_model"]["saved_model_path"]
    baseline_dir: Path
    if cfg["saved_model"]["use_saved_model"]:
        baseline_dir = _ensure_ramp_baseline(
            data_base_path=data_base_path,
            baseline_rel_path=saved_rel,
        )
        ck = baseline_dir / Path(saved_rel).name
        cfg["saved_model"]["use_saved_model"] = ck.is_file() or (baseline_dir / "saved_model.pb").exists()

    run_main_train_code(cfg)
    final_accuracy, final_model_path = extract_highest_accuracy_model(ramp_train_dir)

    if log_zenml_step_metadata:
        log_metadata(
            metadata={
                "best_val_accuracy": float(final_accuracy),
                "best_model_path": str(final_model_path),
            }
        )

    return tf.keras.models.load_model(str(final_model_path), compile=False)


@step
def run_inference(
    model_uri: Union[str, Path, Any],
    input_path: str,
    prediction_path: str,
    output_dir: str,
    model_cache_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """ZenML step wrapper for RAMP inference returning final GeoJSON content.

    model_uri may be a STAC/local path, HTTP(S) .zip URL, or a ``tf.keras.Model`` from training.
    """
    return infer_ramp_model(
        model_uri=model_uri,
        input_path=input_path,
        prediction_path=prediction_path,
        output_dir=output_dir,
        model_cache_dir=model_cache_dir,
    )


def infer_ramp_model(
    model_uri: Union[str, Path, Any],
    input_path: str,
    prediction_path: str,
    output_dir: str,
    model_cache_dir: Optional[str] = None,
    max_chips: Optional[int] = None,
) -> Dict[str, Any]:
    """Run fairpredictor inference and return final merged GeoJSON content.

    model_uri: local path, HTTP(S) URL to a .zip SavedModel, or a ``tf.keras.Model`` from ``train_ramp_model``.
    """
    import tempfile

    import tensorflow as tf

    # Ensure fairpredictor does not use TFSMLayer on TF 2.15.
    _patch_predictor_savedmodel_loader_for_tf215()
    from predictor.prediction import run_prediction  # noqa: E402

    cache = Path(model_cache_dir) if model_cache_dir else None
    if isinstance(model_uri, (str, Path)):
        model_dir = resolve_model_href(str(model_uri), cache_dir=cache)
    elif isinstance(model_uri, tf.keras.Model):
        tmp = Path(tempfile.mkdtemp(prefix="ramp_infer_savedmodel_"))
        model_uri.save(str(tmp))
        model_dir = str(tmp)
    else:
        raise TypeError(
            "model_uri must be a str, pathlib.Path, or a tf.keras.Model from training (compile=False load)."
        )

    # Fail fast if our patch didn't apply (helps diagnose environment mismatches early).
    import importlib

    pred = importlib.import_module("predictor.prediction")
    if not getattr(getattr(pred, "_load_keras_model", None), "_fair_models_tf215_savedmodel_loader", False):
        raise RuntimeError(
            "fairpredictor SavedModel loader patch not applied; "
            "expected predictor.prediction._load_keras_model to be patched for TF 2.15."
        )

    input_dir = _resolve_input_directory(input_path, "input_path")
    out_dir = _to_local_path(prediction_path, "prediction_path")
    out_dir.mkdir(parents=True, exist_ok=True)

    chip_files = sorted(input_dir.glob("**/*.tif"))
    if not chip_files:
        raise RuntimeError(
            f"No GeoTIFF chips (*.tif) found in {input_dir}. "
            "RAMP inference expects georeferenced chips."
        )

    run_input_dir = input_dir
    if max_chips is not None and max_chips > 0:
        subset_dir = out_dir / "subset_input"
        if subset_dir.exists():
            rmtree(subset_dir)
        subset_dir.mkdir(parents=True, exist_ok=True)
        for chip_file in chip_files[:max_chips]:
            copy2(chip_file, subset_dir / chip_file.name)
        run_input_dir = subset_dir

    georef_dir = run_prediction(
        checkpoint_path=model_dir,
        input_path=str(run_input_dir),
        prediction_path=str(out_dir),
        confidence=0.5,
        crs="3857",
    )
    return postprocess(str(georef_dir), output_dir)


@step
def run_postprocessing(
    prediction_path: Union[Path, str],
    output_dir: str,
) -> Dict[str, Any]:
    """Run fairpredictor-style postprocessing and return merged GeoJSON content."""
    return postprocess(str(prediction_path), output_dir)


@pipeline
def training_pipeline(
    input_path: str,
    output_path: str,
    stac_item_path: str = "models/ramp/stac-item.json",
) -> None:
    """Full RAMP training: georeference + multimask → val split → EfficientNetB0 U-Net.

    Hyperparameters (backbone, epochs, batch_size, boundary_width, contact_spacing, etc.)
    are loaded from the STAC Item at stac_item_path.
    """
    hyperparams = _load_hyperparams_from_stac(stac_item_path)
    boundary_width = hyperparams.get("boundary_width", 3)
    contact_spacing = hyperparams.get("contact_spacing", 8)

    data_loader = run_preprocessing(
        input_path=input_path,
        output_path=f"{output_path}/preprocessed",
        boundary_width=boundary_width,
        contact_spacing=contact_spacing,
    )
    split_info = split_dataset(
        preprocessed_path=f"{output_path}/preprocessed",
        output_path=output_path,
        hyperparameters=hyperparams,
    )
    train_model(
        data_base_path=output_path,
        data_loader=data_loader,
        preprocessed_path=f"{output_path}/preprocessed",
        stac_item_path=stac_item_path,
        split_info=split_info,
    )


@pipeline
def inference_pipeline(
    model_uri: Union[str, Path, Any],
    input_path: str,
    prediction_path: str,
    output_dir: str,
    model_cache_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """RAMP inference: load model → predict → postprocess → final GeoJSON.

    model_uri: local path, HTTP(S) URL to a .zip SavedModel, or a ``tf.keras.Model`` from training.
    """
    final_geojson = run_inference(
        model_uri=model_uri,
        input_path=input_path,
        prediction_path=prediction_path,
        output_dir=output_dir,
        model_cache_dir=model_cache_dir,
    )
    return final_geojson
