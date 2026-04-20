"""ZenML pipeline for YOLOv8 building instance segmentation."""

import gc
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Annotated, Any
from urllib.request import urlretrieve

from zenml import log_metadata, pipeline, step

from fair.zenml.steps import load_model

_DEFAULT_WEIGHTS_CACHE = Path("/workspace/.yolo_weights_cache")


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


def resolve_model_href(
    model_uri: str,
    cache_dir: Path | None = None,
) -> str:
    """Resolve model_uri to a local .pt checkpoint path."""
    if not isinstance(model_uri, str):
        raise TypeError("model_uri must be a string")
    if cache_dir is not None and not isinstance(cache_dir, Path):
        raise TypeError("cache_dir must be a pathlib.Path or None")

    cache_dir = cache_dir or _DEFAULT_WEIGHTS_CACHE

    if not (model_uri.startswith("http://") or model_uri.startswith("https://")):
        resolved = _to_local_path(model_uri, "model_uri").resolve()
        if resolved.exists():
            return str(resolved)
        raise FileNotFoundError(f"Model path not found: {resolved}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    url_path = model_uri.split("?")[0]
    base_name = Path(url_path).name or "weights.pt"
    if not base_name.endswith(".pt"):
        base_name = "yolo_weights.pt"
    dest = cache_dir / base_name
    if dest.is_file():
        return str(dest)

    urlretrieve(model_uri, dest)
    if not dest.is_file():
        raise RuntimeError(f"Download failed for {model_uri}")
    return str(dest)


def _patch_yolo_write_yolo_file_with_affine() -> None:
    """Patch hot_fair_utilities label conversion with affine world->pixel mapping.

    Why this patch exists:
    The upstream implementation can mix CRS-dependent bounds math and then clamp values,
    which may collapse polygon labels to corners (0/1) and produce zero usable instances.
    This replacement uses rasterio's inverse affine transform directly, which is CRS-safe.
    """
    import importlib

    import hot_fair_utilities.preprocessing.yolo_v8.utils as yolo_utils
    import rasterio
    from pyproj import Transformer

    if getattr(yolo_utils, "_fair_models_affine_patch_applied", False):
        return

    def _infer_label_crs_from_coords(x: float, y: float) -> str:
        """Best-effort CRS inference for label GeoJSON coordinates.

        hot_fair_utilities.preprocess defaults to EPSG:3857 and clips labels in meters.
        The smoke tests start from EPSG:4326 OSM labels, but the per-chip GeoJSONs are
        typically EPSG:3857 after preprocessing. GeoJSON itself usually omits CRS.
        """
        if -180.0 <= x <= 180.0 and -90.0 <= y <= 90.0:
            return "EPSG:4326"
        return "EPSG:3857"

    def _patched_write_yolo_file(iwp, folder, output_path, class_index=0):
        """Write YOLO polygon labels using affine pixel normalization."""
        lwp = iwp.replace(".tif", ".geojson").replace("chips", "labels")
        ywp = str(Path(output_path) / "labels" / folder / Path(iwp).name.replace(".tif", ".txt"))
        Path(ywp).parent.mkdir(parents=True, exist_ok=True)
        if Path(ywp).exists():
            Path(ywp).unlink()

        with open(lwp, encoding="utf-8") as file:
            data = json.load(file)

        with rasterio.open(iwp) as src:
            if src.crs is None:
                raise ValueError(f"No CRS found in chip: {iwp}")

            inv_transform = ~src.transform
            width = float(src.width)
            height = float(src.height)

            # Determine label CRS from coordinate magnitudes (GeoJSON usually omits CRS).
            label_crs = None
            for feature in data.get("features", []):
                geometry = feature.get("geometry") or {}
                coords = geometry.get("coordinates") or []
                if (
                    geometry.get("type") == "Polygon"
                    and coords
                    and coords[0]
                    and coords[0][0]
                    and len(coords[0][0]) >= 2
                ):
                    x0, y0 = float(coords[0][0][0]), float(coords[0][0][1])
                    label_crs = _infer_label_crs_from_coords(x0, y0)
                    break
            label_crs = label_crs or "EPSG:3857"

            transformer = None
            if str(src.crs).upper() != label_crs.upper():
                transformer = Transformer.from_crs(label_crs, src.crs, always_xy=True)

            lines = []
            for feature in data.get("features", []):
                geometry = feature.get("geometry") or {}
                geometry_type = geometry.get("type")
                if geometry_type == "MultiPolygon":
                    polygons = geometry.get("coordinates", [])
                elif geometry_type == "Polygon":
                    polygons = [geometry.get("coordinates", [])]
                else:
                    continue

                for poly in polygons:
                    if not poly:
                        continue

                    # Use outer ring only (holes are not represented in YOLO polygon labels).
                    ring = poly[0] if poly and isinstance(poly[0], list) else []
                    if len(ring) < 4:
                        continue

                    # Drop closing coordinate if it repeats the first point.
                    if (
                        ring
                        and ring[0]
                        and ring[-1]
                        and len(ring[0]) >= 2
                        and len(ring[-1]) >= 2
                        and ring[0][0] == ring[-1][0]
                        and ring[0][1] == ring[-1][1]
                    ):
                        ring = ring[:-1]

                    points = []
                    for coord in ring:
                        if not coord or len(coord) < 2:
                            continue
                        x_world, y_world = float(coord[0]), float(coord[1])
                        if transformer is not None:
                            x_world, y_world = transformer.transform(x_world, y_world)

                        col, row = inv_transform * (x_world, y_world)
                        x_norm = max(0.0, min(1.0, col / width))
                        y_norm = max(0.0, min(1.0, row / height))
                        points.append((round(x_norm, 6), round(y_norm, 6)))

                    # Require at least 3 distinct vertices.
                    if len({p for p in points}) < 3:
                        continue

                    flattened = [str(v) for p in points for v in p]
                    if len(flattened) >= 6:
                        lines.append(f"{class_index} " + " ".join(flattened))

        Path(ywp).write_text("\n".join(lines), encoding="utf-8")

    # yolo_format binds write_yolo_file at module import time, so patch:
    # - the source symbol in utils
    # - the already-imported global in the yolo_format module
    yolo_utils.write_yolo_file = _patched_write_yolo_file
    yolo_format_module = importlib.import_module("hot_fair_utilities.preprocessing.yolo_v8.yolo_format")
    from typing import cast

    cast(Any, yolo_format_module).write_yolo_file = _patched_write_yolo_file
    yolo_utils._fair_models_affine_patch_applied = True


def preprocess(input_path: str, output_path: str, p_val: float = 0.05) -> str:
    """Preprocess OAM chips + labels into a georeferenced, clipped dataset."""
    from hot_fair_utilities import preprocess as _preprocess

    local_input = _resolve_input_directory(input_path, "input_path")
    preprocessed_path = str(Path(output_path) / "preprocessed")
    _preprocess(
        input_path=str(local_input),
        output_path=preprocessed_path,
        rasterize=True,
        rasterize_options=["binary"],
        georeference_images=True,
        multimasks=False,
    )
    return preprocessed_path


def postprocess(prediction_path: str, output_geojson: str) -> dict[str, Any]:
    """Merge predicted-mask GeoTIFF tiles into a building-footprint GeoJSON."""
    from hot_fair_utilities import polygonize

    try:
        polygonize(
            input_path=prediction_path,
            output_path=output_geojson,
            remove_inputs=False,
        )
    except Exception as exc:
        # Zero detections can bubble up from geomltoolkits/rtree as an empty STR bulk-load stream.
        # In that case, treat inference output as a valid empty FeatureCollection.
        if "Empty data stream given" not in str(exc):
            raise
        empty = {"type": "FeatureCollection", "features": []}
        Path(output_geojson).write_text(json.dumps(empty), encoding="utf-8")
        return empty

    with open(output_geojson, encoding="utf-8") as f:
        return json.load(f)


def _select_or_merge_labels(labels_path: Path, destination: Path) -> None:
    """Materialize a single labels.geojson for hot_fair_utilities preprocess."""
    if labels_path.is_file():
        shutil.copy2(labels_path, destination)
        return

    if not labels_path.is_dir():
        raise FileNotFoundError(f"dataset_labels path not found: {labels_path}")

    geojson_files = sorted(labels_path.glob("*.geojson"))
    if not geojson_files:
        raise FileNotFoundError(f"No .geojson files found in labels directory: {labels_path}")
    if len(geojson_files) == 1:
        shutil.copy2(geojson_files[0], destination)
        return

    import geopandas as gpd
    import pandas as pd

    gdfs = [gpd.read_file(p) for p in geojson_files]
    crs = gdfs[0].crs or "EPSG:4326"
    merged = gpd.GeoDataFrame(pd.concat([g.to_crs(crs) for g in gdfs], ignore_index=True), crs=crs)
    for col in merged.columns:
        if col == "geometry":
            continue
        if pd.api.types.is_extension_array_dtype(merged[col].dtype):
            merged[col] = merged[col].astype(object).where(merged[col].notna(), None)
    merged.to_file(destination, driver="GeoJSON")


def _materialize_training_input(dataset_chips: str, dataset_labels: str, work_dir: Path) -> Path:
    """Create preprocess input folder with chip PNGs and labels.geojson."""
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

    _select_or_merge_labels(labels_path, input_dir / "labels.geojson")
    return input_dir


def _training_cache_dir(dataset_chips: str, dataset_labels: str) -> Path:
    cache_key = hashlib.sha256(f"{dataset_chips}|{dataset_labels}".encode()).hexdigest()[:16]
    return Path(tempfile.gettempdir()) / f"yolo_v8_segmentation_{cache_key}"


def _ensure_non_empty_validation_split(yolo_dir: Path) -> None:
    """Guarantee at least one validation sample for YOLO training."""
    val_images_dir = yolo_dir / "images" / "val"
    val_labels_dir = yolo_dir / "labels" / "val"
    if any(val_images_dir.glob("*")):
        return

    for source_split in ("train", "test"):
        source_images_dir = yolo_dir / "images" / source_split
        source_labels_dir = yolo_dir / "labels" / source_split
        image_candidates = sorted(p for p in source_images_dir.glob("*") if p.is_file())
        if not image_candidates:
            continue

        image_path = image_candidates[0]
        val_images_dir.mkdir(parents=True, exist_ok=True)
        val_labels_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(image_path), str(val_images_dir / image_path.name))

        label_name = image_path.with_suffix(".txt").name
        label_path = source_labels_dir / label_name
        if label_path.exists():
            shutil.move(str(label_path), str(val_labels_dir / label_name))
        return

    raise RuntimeError(f"YOLO split produced no validation candidates under {yolo_dir}")


def _prepare_training_split(
    dataset_chips: str,
    dataset_labels: str,
    hyperparameters: dict[str, Any],
    force_rebuild: bool = False,
) -> dict[str, Any]:
    from hot_fair_utilities.preprocessing.yolo_v8 import yolo_format

    p_val = float(
        hyperparameters.get(
            "training.val_ratio",
            hyperparameters.get("p_val", hyperparameters.get("val_ratio", 0.2)),
        )
    )
    seed = int(hyperparameters.get("training.split_seed", hyperparameters.get("split_seed", 42)))
    if not 0.0 < p_val < 1.0:
        raise ValueError("p_val/val_ratio must be in (0.0, 1.0)")

    work_dir = _training_cache_dir(dataset_chips, dataset_labels)
    yolo_dir = work_dir / "yolo"
    dataset_yaml = yolo_dir / "yolo_dataset.yaml"

    if force_rebuild and work_dir.exists():
        shutil.rmtree(work_dir)

    if not dataset_yaml.exists():
        work_dir.mkdir(parents=True, exist_ok=True)
        input_dir = _materialize_training_input(dataset_chips, dataset_labels, work_dir)
        preprocessed_dir = Path(preprocess(str(input_dir), str(work_dir), p_val=p_val))
        _patch_yolo_write_yolo_file_with_affine()
        yolo_format(
            input_path=str(preprocessed_dir),
            output_path=str(yolo_dir),
            seed=seed,
            train_split=1.0 - p_val,
            val_split=p_val,
            test_split=0.0,
        )
        _ensure_non_empty_validation_split(yolo_dir)

    train_count = len(list((yolo_dir / "images" / "train").glob("*")))
    val_count = len(list((yolo_dir / "images" / "val").glob("*")))

    return {
        "strategy": "random",
        "val_ratio": p_val,
        "seed": seed,
        "train_count": train_count,
        "val_count": val_count,
        "description": "Preprocess chips + labels, then seeded random split via YOLO formatting.",
        "_work_dir": str(work_dir),
        "_yolo_dir": str(yolo_dir),
        "_dataset_yaml": str(dataset_yaml),
    }


def train_yolo_model(
    data_base_path: str,
    yolo_data_dir: str,
    weights_path: str,
    epochs: int = 20,
    batch_size: int = 16,
    pc: float = 2.0,
) -> tuple[str, float]:
    """Fine-tune YOLOv8 segmentation and return (checkpoint_path, iou_accuracy_pct)."""
    import hot_fair_utilities.utils
    import ultralytics

    def _safe_get_iou(model_path):
        model_val = ultralytics.YOLO(model_path)
        model_val_metrics = model_val.val().results_dict
        precision = model_val_metrics.get("metrics/precision(M)", 0.0)
        recall = model_val_metrics.get("metrics/recall(M)", 0.0)
        iou_accuracy = 0.0 if precision <= 0.0 or recall <= 0.0 else 1.0 / (1.0 / precision + 1.0 / recall - 1.0)
        final_accuracy = iou_accuracy * 100
        del model_val
        gc.collect()
        return final_accuracy

    hot_fair_utilities.utils.get_yolo_iou_metrics = _safe_get_iou

    from hot_fair_utilities.training.yolo_v8 import train as _train

    dataset_yaml = str(Path(yolo_data_dir) / "yolo_dataset.yaml")
    model_path, iou_accuracy = _train(
        data=data_base_path,
        weights=weights_path,
        epochs=epochs,
        batch_size=batch_size,
        pc=pc,
        output_path=yolo_data_dir,
        dataset_yaml_path=dataset_yaml,
    )
    return model_path, float(iou_accuracy)


def _restore_checkpoint(trained_model: Any):
    from ultralytics import YOLO

    if isinstance(trained_model, bytes):
        checkpoint = Path(tempfile.mkdtemp()) / "best.pt"
        checkpoint.write_bytes(trained_model)
        return YOLO(str(checkpoint))
    if isinstance(trained_model, (str, Path)):
        return YOLO(str(trained_model))
    return trained_model


def _build_feature_collection(features: list[dict[str, Any]]) -> dict[str, Any]:
    return {"type": "FeatureCollection", "features": features}


def _softmax(logits: Any, axis: int) -> Any:
    import numpy as np

    shifted = logits - logits.max(axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=axis, keepdims=True)


def _preprocess_onnx_image(img_path: Path, model_size: int = 640) -> tuple[Any, Any, Any]:
    import numpy as np
    import rasterio
    from PIL import Image
    from rasterio import Affine

    with rasterio.open(img_path) as src:
        arr = src.read([1, 2, 3]).astype(np.float32) / 255.0
        transform = src.transform
        crs = src.crs
        src_height = src.height
        src_width = src.width

    resized = [
        np.asarray(Image.fromarray(arr[c]).resize((model_size, model_size), Image.Resampling.BILINEAR))
        for c in range(arr.shape[0])
    ]
    batch = np.stack(resized, axis=0)[np.newaxis, ...].astype(np.float32)

    # Model output mask is in model_size x model_size coordinates; scale affine accordingly.
    scaled_transform = transform * Affine.scale(src_width / float(model_size), src_height / float(model_size))
    return batch, scaled_transform, crs


def _vectorize_binary_mask(mask: Any, transform: Any, crs: Any, confidence: float) -> list[dict[str, Any]]:
    import numpy as np
    import rasterio.features
    from pyproj import Transformer

    mask_uint8 = mask.astype(np.uint8)
    needs_reproject = crs is not None and str(crs) != "EPSG:4326"
    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True) if needs_reproject else None

    features: list[dict[str, Any]] = []
    for geom, value in rasterio.features.shapes(mask_uint8, transform=transform):
        if int(value) < 1:
            continue
        if transformer:
            coords = geom["coordinates"]
            geom["coordinates"] = [[list(transformer.transform(x, y)) for x, y in ring] for ring in coords]
        features.append(
            {
                "type": "Feature",
                "properties": {"class": 0, "confidence": confidence},
                "geometry": geom,
            }
        )
    return features


def _decode_segmentation_mask(outputs: list[Any], confidence_threshold: float) -> Any | None:
    import numpy as np

    if not outputs:
        return None

    for output in outputs:
        arr = np.asarray(output)
        if arr.ndim != 4:
            continue
        if arr.shape[0] != 1:
            continue

        # Common segmentation ONNX output layouts:
        # - (1, classes, H, W)
        # - (1, 1, H, W)
        if arr.shape[1] > 1:
            probs = _softmax(arr[0], axis=0)
            top_prob = probs.max(axis=0)
            return np.where(top_prob >= confidence_threshold, 1, 0).astype(np.uint8)
        if arr.shape[1] == 1:
            logits = arr[0, 0]
            return (logits >= confidence_threshold).astype(np.uint8)
    return None


def predict(session: Any, input_images: str, params: dict[str, Any]) -> dict[str, Any]:
    from fair.utils.data import resolve_directory

    confidence_threshold = float(params.get("confidence_threshold", 0.5))
    model_size = int(params.get("model_input_size", 640))
    input_name = session.get_inputs()[0].name

    input_dir = resolve_directory(input_images)
    patterns = ("*.png", "*.tif", "*.tiff", "*.jpg")
    img_paths = sorted(p for pat in patterns for p in input_dir.glob(pat))
    if not img_paths:
        raise FileNotFoundError(f"No input images found in {input_dir}")

    features: list[dict[str, Any]] = []
    for img_path in img_paths:
        batch, transform, crs = _preprocess_onnx_image(img_path, model_size=model_size)
        outputs = session.run(None, {input_name: batch})
        mask = _decode_segmentation_mask(outputs, confidence_threshold)
        if mask is None:
            continue
        features.extend(_vectorize_binary_mask(mask, transform, crs, confidence_threshold))
    return _build_feature_collection(features)


def infer_yolo_model(
    model_uri: str | Path | Any,
    input_path: str,
    prediction_path: str,
    output_dir: str,
    confidence: float = 0.5,
    model_cache_dir: str | None = None,
) -> dict[str, Any]:
    """Run YOLO instance-segmentation inference and return final GeoJSON content."""
    import ultralytics
    from hot_fair_utilities import predict

    cache = Path(model_cache_dir) if model_cache_dir else None

    if isinstance(model_uri, (str, Path)):
        checkpoint_path = resolve_model_href(str(model_uri), cache_dir=cache)
    elif isinstance(model_uri, bytes):
        checkpoint_file = Path(tempfile.mkdtemp()) / "finetuned.pt"
        checkpoint_file.write_bytes(model_uri)
        checkpoint_path = str(checkpoint_file)
    elif isinstance(model_uri, ultralytics.YOLO):
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            checkpoint_path = tmp.name
            model_uri.save(checkpoint_path)
    else:
        raise TypeError("model_uri must be a str, Path, bytes, or an ultralytics.YOLO model.")

    local_input = _resolve_input_directory(input_path, "input_path")
    predict(
        checkpoint_path=checkpoint_path,
        input_path=str(local_input),
        prediction_path=prediction_path,
        confidence=confidence,
    )

    out_dir_path = Path(output_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    output_geojson = str(out_dir_path / "prediction.geojson")
    return postprocess(prediction_path, output_geojson)


@step
def split_dataset(
    dataset_chips: str,
    dataset_labels: str,
    hyperparameters: dict[str, Any],
) -> Annotated[dict[str, Any], "split_info"]:
    split_info = _prepare_training_split(dataset_chips, dataset_labels, hyperparameters)
    log_metadata(metadata={"fair/split": {k: v for k, v in split_info.items() if not k.startswith("_")}})
    return split_info


@step
def run_preprocessing(input_path: str, output_path: str, p_val: float = 0.05) -> str:
    """STAC entrypoint wrapper: preprocess raw chips + labels."""
    return preprocess(input_path, output_path, p_val)


@step
def train_model(
    dataset_chips: str,
    dataset_labels: str,
    base_model_weights: str,
    hyperparameters: dict[str, Any],
    split_info: dict[str, Any],
    num_classes: int = 1,
    model_name: str | None = None,
    base_model_id: str | None = None,
    dataset_id: str | None = None,
) -> Annotated[bytes, "trained_model"]:
    _ = (num_classes, model_name, base_model_id, dataset_id)
    epochs = int(hyperparameters.get("training.epochs", hyperparameters.get("epochs", 20)))
    batch_size = int(hyperparameters.get("training.batch_size", hyperparameters.get("batch_size", 16)))
    pc = float(hyperparameters.get("training.pc", hyperparameters.get("pc", 2.0)))

    yolo_dir = Path(split_info["_yolo_dir"])
    if not (yolo_dir / "yolo_dataset.yaml").exists():
        split_info = _prepare_training_split(dataset_chips, dataset_labels, hyperparameters, force_rebuild=True)
        yolo_dir = Path(split_info["_yolo_dir"])

    weights_path = resolve_model_href(base_model_weights)
    model_path, iou_accuracy = train_yolo_model(
        data_base_path=split_info["_work_dir"],
        yolo_data_dir=str(yolo_dir),
        weights_path=weights_path,
        epochs=epochs,
        batch_size=batch_size,
        pc=pc,
    )
    log_metadata(metadata={"iou_accuracy_pct": float(iou_accuracy), "checkpoint": model_path})
    return Path(model_path).read_bytes()


@step
def evaluate_model(
    trained_model: Any,
    dataset_chips: str,
    dataset_labels: str,
    hyperparameters: dict[str, Any],
    split_info: dict[str, Any],
    class_names: list[str] | None = None,
) -> Annotated[dict[str, Any], "metrics"]:
    _ = class_names
    imgsz = int(
        hyperparameters.get(
            "training.imgsz",
            hyperparameters.get("imgsz", hyperparameters.get("chip_size", 256)),
        )
    )

    dataset_yaml = Path(split_info["_dataset_yaml"])
    if not dataset_yaml.exists():
        split_info = _prepare_training_split(dataset_chips, dataset_labels, hyperparameters, force_rebuild=True)
        dataset_yaml = Path(split_info["_dataset_yaml"])

    model = _restore_checkpoint(trained_model)
    results = model.val(data=str(dataset_yaml), imgsz=imgsz, split="val", verbose=False)
    metrics = getattr(results, "results_dict", None) or {}
    if not metrics:
        raise RuntimeError("YOLO validation produced no results")

    metrics_dict: dict[str, Any] = {
        "fair:accuracy": float(metrics.get("metrics/mAP50(M)", 0.0)),
        "fair:mean_iou": float(metrics.get("metrics/mAP50-95(M)", 0.0)),
        "fair:precision": float(metrics.get("metrics/precision(M)", 0.0)),
        "fair:recall": float(metrics.get("metrics/recall(M)", 0.0)),
    }
    log_metadata(metadata=metrics_dict)
    return metrics_dict


@step
def export_onnx(trained_model: Any) -> Annotated[bytes, "onnx_model"]:
    import onnx

    model = _restore_checkpoint(trained_model)
    onnx_path = model.export(format="onnx")
    onnx_path = Path(str(onnx_path))
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX export did not produce expected file: {onnx_path}")
    onnx.checker.check_model(str(onnx_path))
    return onnx_path.read_bytes()


@step
def run_inference(
    model_uri: str | Path | Any,
    input_images: str,
    prediction_path: str,
    output_dir: str,
    confidence: float = 0.5,
    model_cache_dir: str | None = None,
) -> Annotated[dict[str, Any], "predictions"]:
    """Inference wrapper preserving existing predict -> polygonize flow."""
    return infer_yolo_model(
        model_uri=model_uri,
        input_path=input_images,
        prediction_path=prediction_path,
        output_dir=output_dir,
        confidence=confidence,
        model_cache_dir=model_cache_dir,
    )


@step
def run_postprocessing(prediction_path: str, output_geojson: str) -> dict[str, Any]:
    """STAC entrypoint wrapper: polygonize prediction masks."""
    return postprocess(prediction_path, output_geojson)


@pipeline
def training_pipeline(
    base_model_weights: str,
    dataset_chips: str,
    dataset_labels: str,
    num_classes: int,
    hyperparameters: dict[str, Any],
) -> None:
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
    inference_params: dict[str, Any] | None = None,
    output_dir: str = "",
    chip_size: int = 256,
    num_classes: int = 1,
    confidence: float = 0.5,
    zenml_artifact_version_id: str = "",
    prediction_path: str = "",
) -> dict[str, Any]:
    _ = (chip_size, num_classes)
    resolved_output_dir = output_dir or str(Path(tempfile.mkdtemp(prefix="yolo_v8_seg_inference_")))
    resolved_confidence = float((inference_params or {}).get("confidence_threshold", confidence))
    prediction_dir = prediction_path or str(Path(resolved_output_dir) / "predictions")
    model = (
        load_model(model_uri=model_uri, zenml_artifact_version_id=zenml_artifact_version_id)
        if zenml_artifact_version_id
        else model_uri
    )
    return run_inference(
        model_uri=model,
        input_images=input_images,
        prediction_path=prediction_dir,
        output_dir=resolved_output_dir,
        confidence=resolved_confidence,
    )
