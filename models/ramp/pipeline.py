"""ZenML pipeline for RAMP (EfficientNetB0 + U-Net) building semantic segmentation."""

import json
import os
import re
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Annotated, Any
from urllib.request import urlretrieve

from zenml import log_metadata, pipeline, step

from fair.zenml.materializers import CheckpointBytesMaterializer, ONNXMaterializer

_DEFAULT_MODEL_CACHE = Path("/workspace/.ramp_model_cache")
_QUBVEL_EFFICIENTNET_RELEASE = "https://github.com/qubvel/efficientnet/releases/download/v0.0.1/"


def _to_local_path(path_value: str, purpose: str) -> Path:
    """Resolve a path with UPath and ensure local filesystem semantics."""
    from upath import UPath

    upath_obj = UPath(path_value)
    protocol = getattr(upath_obj, "protocol", "") or ""
    if protocol not in ("", "file"):
        raise NotImplementedError(
            f"{purpose} requires a local filesystem path. Received protocol={protocol!r} for {path_value!r}."
        )
    return Path(str(upath_obj))


def _resolve_input_directory(path_value: str, purpose: str) -> Path:
    """Resolve local/remote dataset directories to a local path."""
    from fair.utils.data import resolve_directory

    if "://" in str(path_value):
        return resolve_directory(path_value, pattern="*")
    return _to_local_path(path_value, purpose)


def _resolve_input_file(path_value: str, purpose: str) -> Path:
    """Resolve local/remote file paths to a local path."""
    from fair.utils.data import resolve_path

    if "://" in str(path_value):
        return resolve_path(path_value)
    return _to_local_path(path_value, purpose)


def _download_and_extract_zip(zip_url: str, dest_dir: Path) -> None:
    """Download a ZIP URL and extract in dest_dir."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_name = Path(zip_url.split("/")[-1]).name or "ramp_v1.zip"
    zip_path = dest_dir / zip_name
    urlretrieve(zip_url, zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)
    zip_path.unlink(missing_ok=True)


def _extract_zip(zip_path: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)


def resolve_model_href(model_uri: str, cache_dir: Path | None = None) -> str:
    """Resolve .onnx, .keras, or .zip model URIs to local paths.
    - .onnx -> local .onnx file path
    - .keras -> local .keras file path
    - .zip  -> extracted SavedModel directory path (folder containing saved_model.pb)
    """
    if not isinstance(model_uri, str):
        raise TypeError("model_uri must be a string")
    if cache_dir is not None and not isinstance(cache_dir, Path):
        raise TypeError("cache_dir must be a pathlib.Path or None")
    cache_dir = cache_dir or _DEFAULT_MODEL_CACHE
    cache_dir.mkdir(parents=True, exist_ok=True)
    is_http = model_uri.startswith(("http://", "https://"))
    clean_uri = model_uri.split("?", 1)[0]
    suffix = Path(clean_uri).suffix.lower()
    # ONNX path
    if suffix == ".onnx":
        if is_http:
            dest = cache_dir / (Path(clean_uri).name or "model.onnx")
            if not dest.is_file():
                urlretrieve(model_uri, dest)
            return str(dest)
        # local onnx path
        p = Path(model_uri).resolve()
        if p.is_file():
            return str(p)
        raise FileNotFoundError(f"ONNX model not found: {p}")
    # Keras single-file checkpoint
    if suffix == ".keras":
        if is_http:
            dest = cache_dir / (Path(clean_uri).name or "model.keras")
            if not dest.is_file():
                urlretrieve(model_uri, dest)
            return str(dest)
        p = Path(model_uri).resolve()
        if p.is_file():
            return str(p)
        raise FileNotFoundError(f"Keras model not found: {p}")

    # ZIP path -> must contain saved_model.pb somewhere inside
    if suffix == ".zip":
        stem = Path(clean_uri).stem or "ramp_model"
        dest_dir = cache_dir / stem
        # cache hit
        for existing in dest_dir.rglob("saved_model.pb"):
            return str(existing.parent)
        if is_http:
            zip_path = cache_dir / (Path(clean_uri).name or "model.zip")
            if not zip_path.is_file():
                urlretrieve(model_uri, zip_path)
        else:
            zip_path = Path(model_uri).resolve()
            if not zip_path.is_file():
                raise FileNotFoundError(f"ZIP model not found: {zip_path}")
        dest_dir.mkdir(parents=True, exist_ok=True)
        import zipfile

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dest_dir)
        for existing in dest_dir.rglob("saved_model.pb"):
            return str(existing.parent)
        raise RuntimeError(f"Zip from {model_uri} did not contain a valid SavedModel")
    raise ValueError(f"Unsupported model format for {model_uri!r}. Only .onnx, .keras and .zip are accepted.")


def _ensure_ramp_baseline(base_model_weights: str, data_base_path: str | Path) -> Path:
    """Return a local SavedModel directory for fine-tuning, downloading if necessary.

    ``base_model_weights`` may be an HTTP(S) .zip URL (preferred), a local .zip,
    or a SavedModel directory. A pre-provisioned baseline under
    ``/app/ramp-data/baseline`` (from the RAMP utilities Docker image) is used
    when present and no explicit URL is provided. If the pre-provisioned baseline
    is missing, callers must provide `base_model_weights` (typically from STAC).
    """
    image_ck = Path("/app/ramp-data/baseline")
    if not base_model_weights and (image_ck / "saved_model.pb").exists():
        return image_ck

    if not base_model_weights:
        raise ValueError("RAMP baseline weights are required but were not provided. ")

    return Path(resolve_model_href(base_model_weights, cache_dir=Path(data_base_path) / ".baseline_cache"))


def _patch_keras_get_file_for_efficientnet_weights() -> None:
    """Redirect broken Callidior EfficientNet weight URLs to qubvel's GitHub release assets.

    The ``efficientnet`` package (via ``segmentation_models``) downloads encoder weights from
    ``github.com/Callidior/keras-applications/releases/...``; those assets now return **404**.
    qubvel hosts compatible ``*_imagenet_1000_notop.h5`` files on the same model's releases.
    """
    import tensorflow as tf

    ku = tf.keras.utils
    if hasattr(ku.get_file, "_ramp_efficientnet_mirror"):
        return

    original = ku.get_file

    def _get_file(fname, origin, *args, **kwargs):
        if isinstance(origin, str) and "Callidior" in origin:
            m = re.match(
                r"^(efficientnet-b\d+)_weights_tf_dim_ordering_tf_kernels_autoaugment_notop\.h5$",
                fname or "",
            )
            if m:
                alt = f"{m.group(1)}_imagenet_1000_notop.h5"
                origin = f"{_QUBVEL_EFFICIENTNET_RELEASE}{alt}"
                kwargs = dict(kwargs)
                kwargs["file_hash"] = None
        return original(fname, origin, *args, **kwargs)

    _get_file._ramp_efficientnet_mirror = True  # type: ignore[attr-defined]
    ku.get_file = _get_file


def _materialize_training_input(dataset_chips: str, dataset_labels: str, work_dir: Path) -> Path:
    """Create the preprocess input folder with PNG chips and a single labels.geojson.

    Accepts .tif/.tiff/.png chip files on disk; TIFFs are converted to 3-band PNGs while
    preserving filename stem (e.g. OAM-{x}-{y}-{z}.png) so hot_fair_utilities label clipping
    can parse tile ids.
    """
    chips_dir = _resolve_input_directory(dataset_chips, "dataset_chips")
    labels_path = _resolve_input_file(dataset_labels, "dataset_labels")

    input_dir = work_dir / "input"
    if input_dir.exists():
        shutil.rmtree(input_dir)
    input_dir.mkdir(parents=True, exist_ok=True)

    tif_paths = sorted(list(chips_dir.glob("*.tif")) + list(chips_dir.glob("*.tiff")))
    png_paths = sorted(chips_dir.glob("*.png"))

    if tif_paths:
        import numpy as np
        import rasterio
        from PIL import Image

        for tif_path in tif_paths:
            png_path = input_dir / (tif_path.stem + ".png")
            with rasterio.open(tif_path) as src:
                data = src.read()
                if data.shape[0] < 3:
                    continue
                rgb = np.transpose(data[:3], (1, 2, 0))
                rgb = (rgb * 255).astype(np.uint8) if rgb.max() <= 1.0 else np.clip(rgb, 0, 255).astype(np.uint8)
                Image.fromarray(rgb).save(png_path)

    for png_path in png_paths:
        shutil.copy2(png_path, input_dir / png_path.name)

    if not list(input_dir.glob("*.png")):
        raise FileNotFoundError(f"No train chips (.tif/.tiff/.png) found in {chips_dir}")

    if not labels_path.is_file():
        raise FileNotFoundError(f"dataset_labels file not found: {labels_path}")
    shutil.copy2(labels_path, input_dir / "labels.geojson")
    return input_dir


def preprocess(
    input_path: str,
    output_path: str,
    boundary_width: int = 3,
    contact_spacing: int = 8,
) -> str:
    """Preprocess OAM PNG chips + labels into RAMP 4-class multimasks.

    Emits:
      - preprocessed/chips/*.tif             (georeferenced RGB chips, EPSG:3857)
      - preprocessed/labels/*.geojson        (per-chip labels, reprojected + clipped)
      - preprocessed/multimasks/*.mask.tif   (4-class sparse categorical masks)
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


def postprocess(prediction_masks_dir: str, output_dir: str) -> dict[str, Any]:
    """Merge prediction TIFF tiles into a building-footprint GeoJSON (EPSG:4326)."""
    from geomltoolkits.geometry.validate import validate_polygon_geometries
    from geomltoolkits.raster.merge import merge_rasters
    from geomltoolkits.raster.morphology import morphological_cleaning
    from geomltoolkits.raster.vectorize import vectorize_mask

    pred_dir = _to_local_path(prediction_masks_dir, "prediction_masks_dir")
    out_dir = _to_local_path(output_dir, "output_dir")
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_tifs = sorted(pred_dir.glob("*.tif"))
    if not pred_tifs:
        return {"type": "FeatureCollection", "features": []}

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
        if not merged_geojson_path.is_file():
            merged_geojson_path.write_text(json.dumps(geojson_dict), encoding="utf-8")
        return geojson_dict

    if gdf.crs and gdf.crs != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")
    elif not gdf.crs:
        gdf.set_crs("EPSG:3857", inplace=True)
        gdf = gdf.to_crs("EPSG:4326")

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
            shutil.copy2(validated_geojson, merged_geojson_path)
        return json.loads(merged_geojson_path.read_text(encoding="utf-8"))

    if isinstance(validated_geojson, dict):
        return validated_geojson
    return json.loads(gdf.to_json())


def _prepare_training_split(
    dataset_chips: str,
    dataset_labels: str,
    hyperparameters: dict[str, Any],
    force_rebuild: bool = False,
) -> dict[str, Any]:
    """Preprocess + split chips/labels into RAMP train/val layout; return split_info."""
    from hot_fair_utilities.training.ramp.prepare_data import split_training_2_validation

    val_fraction = float(
        hyperparameters.get(
            "training.val_ratio",
            hyperparameters.get("val_fraction", hyperparameters.get("val_ratio", 0.15)),
        )
    )
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0.0, 1.0)")
    boundary_width = int(hyperparameters.get("training.boundary_width", hyperparameters.get("boundary_width", 3)))
    contact_spacing = int(hyperparameters.get("training.contact_spacing", hyperparameters.get("contact_spacing", 8)))
    seed = int(hyperparameters.get("training.split_seed", hyperparameters.get("split_seed", 42)))

    work_dir = Path(tempfile.mkdtemp(prefix="ramp_training_"))
    preprocessed_dir = work_dir / "preprocessed"
    ramp_train_dir = work_dir / "ramp_training_work"

    work_dir.mkdir(parents=True, exist_ok=True)
    input_dir = _materialize_training_input(dataset_chips, dataset_labels, work_dir)
    preprocess(str(input_dir), str(preprocessed_dir), boundary_width, contact_spacing)
    split_training_2_validation(str(preprocessed_dir), str(ramp_train_dir), multimasks=True)

    train_count = len(list((ramp_train_dir / "chips").glob("*.tif")))
    val_count = len(list((ramp_train_dir / "val-chips").glob("*.tif")))

    return {
        "strategy": "random",
        "val_ratio": val_fraction,
        "seed": seed,
        "train_count": train_count,
        "val_count": val_count,
        "description": "Preprocess chips+labels into 4-class multimasks, then random train/val split.",
        "_work_dir": str(work_dir),
        "_preprocessed_dir": str(preprocessed_dir),
        "_ramp_train_dir": str(ramp_train_dir),
    }


def train_ramp_model(
    ramp_train_dir: str,
    base_model_weights: str,
    hyperparameters: dict[str, Any],
    data_base_path: str | None = None,
) -> Path:
    """Fine-tune EfficientNetB0 + U-Net and return the best SavedModel directory path."""
    # run_training reads RAMP_HOME at import time; set it first so saved_model lookups resolve.
    data_base_path = str(Path(data_base_path).resolve()) if data_base_path else str(Path(ramp_train_dir).resolve())
    image_baseline_ck = Path("/app/ramp-data/baseline/saved_model.pb")
    if image_baseline_ck.exists():
        os.environ["RAMP_HOME"] = "/app"
    else:
        os.environ["RAMP_HOME"] = data_base_path

    os.environ.setdefault("SM_FRAMEWORK", "tf.keras")

    _patch_keras_get_file_for_efficientnet_weights()
    import segmentation_models as sm
    from hot_fair_utilities.training.ramp.cleanup import extract_highest_accuracy_model
    from hot_fair_utilities.training.ramp.config import RAMP_CONFIG
    from hot_fair_utilities.training.ramp.run_training import (
        manage_fine_tuning_config,
        run_main_train_code,
    )

    sm.set_framework("tf.keras")

    epochs = int(hyperparameters.get("training.epochs", hyperparameters.get("epochs", RAMP_CONFIG["num_epochs"])))
    batch_size = int(
        hyperparameters.get("training.batch_size", hyperparameters.get("batch_size", RAMP_CONFIG["batch_size"]))
    )
    backbone = str(
        hyperparameters.get(
            "training.backbone",
            hyperparameters.get("backbone", RAMP_CONFIG["model"]["model_fn_parms"]["backbone"]),
        )
    )
    learning_rate = float(
        hyperparameters.get(
            "training.learning_rate",
            hyperparameters.get("learning_rate", RAMP_CONFIG["optimizer"]["optimizer_fn_parms"]["learning_rate"]),
        )
    )
    patience = int(
        hyperparameters.get(
            "training.early_stopping_patience",
            hyperparameters.get(
                "early_stopping_patience", RAMP_CONFIG["early_stopping"]["early_stopping_parms"]["patience"]
            ),
        )
    )

    cfg = manage_fine_tuning_config(ramp_train_dir, epochs, batch_size, freeze_layers=False, multimasks=True)
    cfg["model"]["model_fn_parms"]["backbone"] = backbone
    cfg["optimizer"]["optimizer_fn_parms"]["learning_rate"] = learning_rate
    cfg["early_stopping"]["early_stopping_parms"]["patience"] = patience

    if cfg["saved_model"]["use_saved_model"]:
        baseline_dir = _ensure_ramp_baseline(base_model_weights, data_base_path)
        cfg["saved_model"]["use_saved_model"] = (baseline_dir / "saved_model.pb").exists()

    run_main_train_code(cfg)
    _final_accuracy, final_model_path = extract_highest_accuracy_model(ramp_train_dir)
    return Path(final_model_path)


def _materialize_keras_bytes(blob: bytes) -> Path:
    """Write Keras `.keras` bytes to a temp file and return its path."""
    dest = Path(tempfile.mkdtemp(prefix="ramp_keras_")) / "model.keras"
    dest.write_bytes(blob)
    return dest


def _restore_checkpoint(trained_model: Any) -> Path:
    """Restore a trained RAMP Keras checkpoint as a local `.keras` file path."""
    if isinstance(trained_model, bytes):
        return _materialize_keras_bytes(trained_model)
    if isinstance(trained_model, (str, Path)):
        p = Path(str(trained_model))
        if p.is_file() and p.suffix.lower() == ".keras":
            return p
        return Path(resolve_model_href(str(p)))
    raise TypeError(f"Cannot restore RAMP checkpoint from {type(trained_model).__name__}")


def _convert_savedmodel_to_onnx_bytes(saved_model_dir: Path, opset: int = 13) -> bytes:
    """Convert a TF SavedModel directory to ONNX bytes via tf2onnx."""
    import tensorflow as tf
    import tf2onnx

    with tempfile.TemporaryDirectory() as tmp:
        onnx_path = Path(tmp) / "model.onnx"
        from_saved_model = getattr(tf2onnx.convert, "from_saved_model", None)
        if callable(from_saved_model):
            from_saved_model(
                str(saved_model_dir),
                output_path=str(onnx_path),
                opset=opset,
            )
        else:
            model = tf.keras.models.load_model(str(saved_model_dir), compile=False)
            tf2onnx.convert.from_keras(
                model,
                opset=opset,
                output_path=str(onnx_path),
            )
        return onnx_path.read_bytes()


def _build_feature_collection(features: list[dict[str, Any]]) -> dict[str, Any]:
    return {"type": "FeatureCollection", "features": features}


def _extract_ramp_shapes(session: Any) -> tuple[str, int, int]:
    """Return (input_name, input_height, input_width) for a RAMP ONNX session (NHWC)."""
    input_meta = session.get_inputs()[0]
    shape = input_meta.shape
    if len(shape) != 4:
        raise RuntimeError(f"Unexpected ONNX input shape: {shape}")
    # RAMP ONNX is [batch, H, W, bands] (channels last) — keep this order.
    height = int(shape[1]) if isinstance(shape[1], int) and shape[1] > 0 else 256
    width = int(shape[2]) if isinstance(shape[2], int) and shape[2] > 0 else 256
    return input_meta.name, height, width


def _prepare_onnx_image(img_path: Path, input_height: int, input_width: int) -> tuple[Any, Any, Any]:
    """Load an RGB chip and return (batch, transform, meta) for ONNX inference."""
    import numpy as np
    import rasterio
    from PIL import Image

    with rasterio.open(img_path) as src:
        arr = src.read([1, 2, 3]).astype(np.float32) / 255.0
        transform = src.transform
        crs = src.crs
        src_height = src.height
        src_width = src.width

    resized = [
        np.asarray(Image.fromarray(arr[c]).resize((input_width, input_height), Image.Resampling.BILINEAR))
        for c in range(arr.shape[0])
    ]
    hwc = np.stack(resized, axis=-1).astype(np.float32)  # NHWC
    batch = hwc[np.newaxis, ...]
    return batch, transform, (src_width, src_height, input_width, input_height, crs)


def _decode_ramp_building_mask(
    output: Any,
    input_height: int,
    input_width: int,
    src_height: int,
    src_width: int,
    min_class_value: int = 1,
) -> Any:
    """Decode a RAMP ONNX output tensor into a (src_height, src_width) uint8 building mask.

    Accepts either a 4-class softmax/logits tensor [B,H,W,C] or a [B,H,W,1] class-index tensor.
    Pixels with class >= ``min_class_value`` (default 1=building) become 1; others 0.
    """
    import numpy as np
    from PIL import Image

    arr = np.asarray(output)
    if arr.ndim == 5:
        arr = arr[0]
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[-1] > 1:
        class_idx = arr.argmax(axis=-1)
    elif arr.ndim == 3 and arr.shape[-1] == 1:
        class_idx = arr[..., 0]
    elif arr.ndim == 2:
        class_idx = arr
    else:
        raise RuntimeError(f"Unexpected RAMP ONNX output shape: {arr.shape}")

    class_idx = np.asarray(class_idx).astype(np.int32)
    # Collapse multiclass (background=0, building=1, boundary=2, contact=3) to binary building.
    binary = (class_idx == min_class_value).astype(np.uint8)
    if binary.shape != (src_height, src_width):
        resized = Image.fromarray(binary * 255).resize((src_width, src_height), Image.Resampling.NEAREST)
        binary = (np.asarray(resized) > 127).astype(np.uint8)
    return binary


def _vectorize_binary_mask(mask: Any, transform: Any, crs: Any, confidence: float) -> list[dict[str, Any]]:
    import numpy as np
    import rasterio.features
    from pyproj import Transformer

    mask_uint8 = np.asarray(mask).astype(np.uint8)
    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True) if crs and str(crs) != "EPSG:4326" else None

    features: list[dict[str, Any]] = []
    for geom, value in rasterio.features.shapes(mask_uint8, transform=transform):
        if int(value) < 1:
            continue
        if transformer is not None:
            coords = geom["coordinates"]
            geom["coordinates"] = [[list(transformer.transform(x, y)) for x, y in ring] for ring in coords]
        features.append(
            {
                "type": "Feature",
                "properties": {"class": 1, "confidence": round(confidence, 4)},
                "geometry": geom,
            }
        )
    return features


def predict(session: Any, input_images: str, params: dict[str, Any]) -> dict[str, Any]:
    """Run RAMP ONNX inference and return a FeatureCollection of building polygons.

    Required by ``fair.serve.base``. ``session`` is an ``onnxruntime.InferenceSession`` built by
    the serving layer from the STAC ``assets.model`` ONNX artifact.
    """
    from fair.utils.data import resolve_directory

    confidence_threshold = float(params.get("confidence_threshold", 0.5))
    min_class_value = int(params.get("min_class_value", 1))

    input_name, input_height, input_width = _extract_ramp_shapes(session)
    input_dir = resolve_directory(input_images)
    patterns = ("*.png", "*.tif", "*.tiff", "*.jpg")
    img_paths = sorted(p for pat in patterns for p in input_dir.glob(pat))
    if not img_paths:
        raise FileNotFoundError(f"No input images found in {input_dir}")

    features: list[dict[str, Any]] = []
    for img_path in img_paths:
        batch, transform, meta = _prepare_onnx_image(img_path, input_height, input_width)
        src_width, src_height, _iw, _ih, crs = meta
        outputs = session.run(None, {input_name: batch})
        if not outputs:
            continue
        mask = _decode_ramp_building_mask(
            outputs[0],
            input_height=input_height,
            input_width=input_width,
            src_height=src_height,
            src_width=src_width,
            min_class_value=min_class_value,
        )
        features.extend(_vectorize_binary_mask(mask, transform, crs, confidence_threshold))
    return _build_feature_collection(features)


@step
def split_dataset(
    dataset_chips: str,
    dataset_labels: str,
    hyperparameters: dict[str, Any],
) -> Annotated[dict[str, Any], "split_info"]:
    """Preprocess chips+labels and create the RAMP train/val layout."""
    split_info = _prepare_training_split(dataset_chips, dataset_labels, hyperparameters)
    log_metadata(metadata={"fair/split": {k: v for k, v in split_info.items() if not k.startswith("_")}})
    return split_info


@step(output_materializers={"trained_model": CheckpointBytesMaterializer})
def train_model(
    dataset_chips: str,
    dataset_labels: str,
    base_model_weights: str,
    hyperparameters: dict[str, Any],
    split_info: dict[str, Any],
    num_classes: int = 4,
    model_name: str | None = None,
    base_model_id: str | None = None,
    dataset_id: str | None = None,
) -> Annotated[bytes, "trained_model"]:
    """Fine-tune RAMP EfficientNetB0 U-Net; return the best model as `.keras` bytes."""
    _ = (num_classes, model_name, base_model_id, dataset_id)

    ramp_train_dir = Path(split_info["_ramp_train_dir"])
    if not ramp_train_dir.exists():
        split_info = _prepare_training_split(dataset_chips, dataset_labels, hyperparameters, force_rebuild=True)
        ramp_train_dir = Path(split_info["_ramp_train_dir"])

    work_dir = split_info.get("_work_dir") or str(ramp_train_dir.parent)
    final_model_path = train_ramp_model(
        ramp_train_dir=str(ramp_train_dir),
        base_model_weights=base_model_weights,
        hyperparameters=hyperparameters,
        data_base_path=work_dir,
    )
    import tensorflow as tf

    model = tf.keras.models.load_model(str(final_model_path), compile=False)
    exported_path = Path(tempfile.mkdtemp(prefix="ramp_keras_export_")) / "model.keras"
    model.save(str(exported_path))
    blob = exported_path.read_bytes()
    log_metadata(metadata={"keras_path": str(exported_path), "checkpoint_bytes": len(blob)})
    return blob


@step
def evaluate_model(
    trained_model: Any,
    dataset_chips: str,
    dataset_labels: str,
    hyperparameters: dict[str, Any],
    split_info: dict[str, Any],
    class_names: list[str] | None = None,
) -> Annotated[dict[str, Any], "metrics"]:
    """Compute per-pixel building-class metrics on the validation split."""
    _ = class_names

    import numpy as np
    import rasterio

    ramp_train_dir = Path(split_info.get("_ramp_train_dir", ""))
    if not ramp_train_dir.exists():
        split_info = _prepare_training_split(dataset_chips, dataset_labels, hyperparameters, force_rebuild=True)
        ramp_train_dir = Path(split_info["_ramp_train_dir"])

    val_chips_dir = ramp_train_dir / "val-chips"
    val_masks_dir = ramp_train_dir / "val-multimasks"
    pairs: list[tuple[Path, Path]] = []
    for chip in sorted(val_chips_dir.glob("*.tif")):
        mask = val_masks_dir / f"{chip.stem}.mask.tif"
        if mask.is_file():
            pairs.append((chip, mask))

    restored = _restore_checkpoint(trained_model)

    if not pairs:
        # No val data to evaluate against (e.g. CI mocks); return zeroed metrics with the
        # required fair:* keys so downstream validation still sees the expected schema.
        zero_metrics: dict[str, Any] = {
            "fair:accuracy": 0.0,
            "fair:mean_iou": 0.0,
            "fair:precision": 0.0,
            "fair:recall": 0.0,
        }
        log_metadata(metadata=zero_metrics)
        return zero_metrics

    import tensorflow as tf

    model = tf.keras.models.load_model(str(restored), compile=False)

    tp = fp = fn = 0
    correct = total = 0
    for chip_path, mask_path in pairs:
        with rasterio.open(chip_path) as src:
            chip = src.read([1, 2, 3]).astype(np.float32) / 255.0
        with rasterio.open(mask_path) as src:
            gt = src.read(1).astype(np.int32)
        batch = np.transpose(chip, (1, 2, 0))[np.newaxis, ...]
        pred = model.predict(batch, verbose=0)
        pred_arr = np.asarray(pred)
        if pred_arr.ndim == 4 and pred_arr.shape[-1] > 1:
            pred_idx = pred_arr[0].argmax(axis=-1)
        elif pred_arr.ndim == 4 and pred_arr.shape[-1] == 1:
            pred_idx = pred_arr[0, ..., 0].astype(np.int32)
        else:
            pred_idx = pred_arr[0].astype(np.int32)

        gt_bin = (gt == 1).astype(np.uint8)
        pr_bin = (pred_idx == 1).astype(np.uint8)
        tp += int(((gt_bin == 1) & (pr_bin == 1)).sum())
        fp += int(((gt_bin == 0) & (pr_bin == 1)).sum())
        fn += int(((gt_bin == 1) & (pr_bin == 0)).sum())
        correct += int((pred_idx == gt).sum())
        total += int(gt.size)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0

    metrics_dict: dict[str, Any] = {
        "fair:accuracy": float(accuracy),
        "fair:mean_iou": float(iou),
        "fair:precision": float(precision),
        "fair:recall": float(recall),
    }
    log_metadata(metadata=metrics_dict)
    return metrics_dict


@step(output_materializers={"onnx_model": ONNXMaterializer})
def export_onnx(trained_model: Any) -> Annotated[bytes, "onnx_model"]:
    """Convert the trained RAMP Keras checkpoint to ONNX bytes and validate."""
    import onnx
    import tensorflow as tf
    import tf2onnx

    restored = _restore_checkpoint(trained_model)
    model = tf.keras.models.load_model(str(restored), compile=False)
    with tempfile.TemporaryDirectory() as tmp:
        onnx_path = Path(tmp) / "model.onnx"
        tf2onnx.convert.from_keras(model, opset=13, output_path=str(onnx_path))
        onnx_bytes = onnx_path.read_bytes()
    onnx.checker.check_model(onnx.load_from_string(onnx_bytes))
    log_metadata(metadata={"onnx_bytes": len(onnx_bytes)})
    return onnx_bytes


@step
def run_inference(
    model_uri: str,
    input_images: str,
    inference_params: dict[str, Any],
) -> Annotated[dict[str, Any], "predictions"]:
    """RAMP batch inference using the promoted ONNX `assets.model`."""
    from fair.serve.base import load_session

    session = load_session(model_uri)
    return predict(session, input_images, inference_params)


@step
def run_preprocessing(
    input_path: str,
    output_path: str,
    boundary_width: int = 3,
    contact_spacing: int = 8,
) -> str:
    """STAC entrypoint wrapper for RAMP preprocessing."""
    return preprocess(input_path, output_path, boundary_width, contact_spacing)


@step
def run_postprocessing(prediction_path: str, output_dir: str) -> dict[str, Any]:
    """STAC entrypoint wrapper for RAMP postprocessing."""
    return postprocess(prediction_path, output_dir)


@pipeline
def training_pipeline(
    base_model_weights: str,
    dataset_chips: str,
    dataset_labels: str,
    num_classes: int,
    hyperparameters: dict[str, Any],
) -> None:
    """RAMP training pipeline: split → train → evaluate → export ONNX."""
    split_info = split_dataset(
        dataset_chips=dataset_chips,
        dataset_labels=dataset_labels,
        hyperparameters=hyperparameters,
    )
    trained_model = train_model(
        dataset_chips=dataset_chips,
        dataset_labels=dataset_labels,
        base_model_weights=base_model_weights,
        hyperparameters=hyperparameters,
        split_info=split_info,
        num_classes=num_classes,
    )
    evaluate_model(
        trained_model=trained_model,
        dataset_chips=dataset_chips,
        dataset_labels=dataset_labels,
        hyperparameters=hyperparameters,
        split_info=split_info,
    )
    export_onnx(trained_model=trained_model)


@pipeline
def inference_pipeline(
    model_uri: str,
    input_images: str,
    inference_params: dict[str, Any],
) -> None:
    """RAMP inference pipeline: load ONNX session → predict → FeatureCollection."""
    run_inference(model_uri=model_uri, input_images=input_images, inference_params=inference_params)
