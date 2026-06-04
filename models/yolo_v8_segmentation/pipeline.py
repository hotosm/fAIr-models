"""ZenML pipeline for YOLOv8 building instance segmentation."""

import gc
import json
import shutil
import tempfile
from pathlib import Path
from typing import Annotated, Any, cast
from urllib.request import urlretrieve

from zenml import log_metadata, pipeline, step

from fair.zenml.instrumentation import log_evaluation_results, mlflow_training_context
from fair.zenml.materializers import CheckpointBytesMaterializer, ONNXMaterializer
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

    # Support both HTTP(S) URLs and fsspec-compatible URIs (e.g. s3:// in-cluster MinIO).
    # The training config may convert MinIO HTTP URLs to s3://... via http_url_to_s3_uri.
    if "://" in model_uri and not (model_uri.startswith("http://") or model_uri.startswith("https://")):
        from fair.utils.data import resolve_path

        cache_dir.mkdir(parents=True, exist_ok=True)
        resolved = resolve_path(model_uri, local_dir=cache_dir)
        if not resolved.is_file():
            raise FileNotFoundError(f"Resolved checkpoint is not a file: {resolved}")
        return str(resolved)

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


def _resolve_labels_geojson(dataset_labels: str) -> Path:
    """Resolve dataset_labels to exactly one GeoJSON/JSON file.

    Supports:
    - direct file path/URI
    - directory/prefix containing exactly one labels file
    """
    from fair.utils.data import resolve_directory, resolve_path

    label_patterns = ("*.geojson", "*.json")
    label_suffixes = (".geojson", ".json")
    labels_value = str(dataset_labels)

    def _pick_single_candidate(search_dir: Path) -> Path:
        for name in ("labels.geojson", "labels.json", "label.geojson", "label.json"):
            candidate = search_dir / name
            if candidate.is_file():
                return candidate

        for pattern in label_patterns:
            matches = sorted(p for p in search_dir.glob(pattern) if p.is_file())
            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                listed = ", ".join(str(p) for p in matches)
                raise ValueError(
                    f"dataset_labels must resolve to exactly one labels file, found {len(matches)} "
                    f"matching {pattern} in {search_dir}: {listed}. "
                    "Rename your labels file to labels.geojson (or pass an explicit file path)."
                )
        raise FileNotFoundError(
            f"No labels file found in {search_dir}. Expected labels.geojson or exactly one '*.geojson' file."
        )

    # Remote file or directory/prefix.
    if "://" in labels_value:
        if labels_value.lower().endswith(label_suffixes):
            file_candidate = resolve_path(labels_value)
            if file_candidate.suffix.lower() not in label_suffixes:
                raise ValueError(f"dataset_labels must point to a .geojson or .json file, got: {file_candidate}")
            return file_candidate
        for pattern in label_patterns:
            try:
                local_dir = resolve_directory(labels_value, pattern=pattern)
            except FileNotFoundError:
                continue
            return _pick_single_candidate(local_dir)
        raise FileNotFoundError(
            f"No labels file found at {labels_value}. Expected exactly one '*.geojson' or '*.json' file."
        )

    # Local directory/file fallback.
    local_path = Path(labels_value)
    if local_path.is_file():
        if local_path.suffix.lower() not in label_suffixes:
            raise ValueError(f"dataset_labels must be a .geojson or .json file, got: {local_path}")
        return local_path
    if local_path.is_dir():
        return _pick_single_candidate(local_path)

    raise FileNotFoundError(f"dataset_labels path not found: {local_path}. Provide a labels file or directory/prefix.")


def _ensure_labels_epsg4326(labels_dir: Path) -> None:
    """Ensure all per-chip label GeoJSONs are EPSG:4326 and JSON-serializable."""
    if not labels_dir.is_dir():
        return

    import geopandas as gpd
    import pandas as pd

    def is_lonlat(bounds: tuple[float, float, float, float]) -> bool:
        minx, miny, maxx, maxy = bounds
        return -180 <= minx <= 180 and -90 <= miny <= 90 and -180 <= maxx <= 180 and -90 <= maxy <= 90

    def jsonable(value: Any) -> Any:
        if value is None or pd.isna(value):
            return None
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        return value

    def make_jsonable(gdf: Any) -> Any:
        geom_name = gdf.geometry.name
        for col in gdf.columns:
            if col == geom_name:
                continue

            series = gdf[col]
            if pd.api.types.is_datetime64_any_dtype(series):
                gdf[col] = series.dt.strftime("%Y-%m-%dT%H:%M:%S%z").where(~series.isna(), None)
            elif getattr(series, "dtype", None) == "object":
                gdf[col] = series.map(jsonable)
        return gdf

    for path in sorted(labels_dir.glob("*.geojson")):
        try:
            gdf = gpd.read_file(path)
        except Exception as e:
            raise RuntimeError(f"Failed to read label GeoJSON {path}") from e

        if gdf.empty:
            continue

        try:
            if gdf.crs is None:
                inferred = "EPSG:4326" if is_lonlat(tuple(gdf.total_bounds)) else "EPSG:3857"
                gdf = gdf.set_crs(inferred)

            epsg = gdf.crs.to_epsg()
            if epsg != 4326:
                gdf = gdf.to_crs("EPSG:4326")

            path.write_text(make_jsonable(gdf).to_json(), encoding="utf-8")
        except Exception as e:
            raise RuntimeError(f"Failed to normalize labels to EPSG:4326 for {path}") from e


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
        epsg=4326,
    )
    return preprocessed_path


def postprocess(prediction_path: str, output_geojson: str) -> dict[str, Any]:
    """Merge predicted-mask GeoTIFF tiles into a building-footprint GeoJSON."""
    import numpy as np
    import rasterio
    from hot_fair_utilities import polygonize

    prediction_dir = Path(prediction_path)
    raster_paths = sorted(p for ext in ("*.tif", "*.tiff", "*.png") for p in prediction_dir.glob(ext))
    if not raster_paths:
        empty = {"type": "FeatureCollection", "features": []}
        Path(output_geojson).write_text(json.dumps(empty), encoding="utf-8")
        return empty

    has_positive_pixels = False
    for raster_path in raster_paths:
        with rasterio.open(raster_path) as src:
            mask = src.read(1)
        if np.any(mask > 0):
            has_positive_pixels = True
            break

    if not has_positive_pixels:
        empty = {"type": "FeatureCollection", "features": []}
        Path(output_geojson).write_text(json.dumps(empty), encoding="utf-8")
        return empty

    polygonize(
        input_path=prediction_path,
        output_path=output_geojson,
        remove_inputs=False,
    )

    with open(output_geojson, encoding="utf-8") as f:
        return json.load(f)


def _copy_labels_geojson(labels_path: Path, destination: Path) -> None:
    """Copy the dataset labels file to `labels.geojson` under the preprocess input folder."""
    if not labels_path.is_file():
        raise FileNotFoundError(f"dataset_labels must be a single GeoJSON file path, got: {labels_path}")
    shutil.copy2(labels_path, destination)


def _materialize_training_input(dataset_chips: str, dataset_labels: str, work_dir: Path) -> Path:
    """Create preprocess input folder with chip PNGs and labels.geojson."""
    chips_dir = _resolve_input_directory(dataset_chips, "dataset_chips")
    labels_path = _resolve_labels_geojson(dataset_labels)

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

    _copy_labels_geojson(labels_path, input_dir / "labels.geojson")
    return input_dir


def _prepare_training_split(
    dataset_chips: str,
    dataset_labels: str,
    hyperparameters: dict[str, Any],
    force_rebuild: bool = False,
    reuse_work_dir: str | None = None,
) -> dict[str, Any]:
    from hot_fair_utilities.preprocessing.yolo_v8 import yolo_format

    p_val = float(hyperparameters.get("val_ratio", hyperparameters.get("p_val", 0.2)))
    seed = int(hyperparameters.get("split_seed", 42))
    if not 0.0 < p_val < 1.0:
        raise ValueError("p_val/val_ratio must be in (0.0, 1.0)")

    if force_rebuild and reuse_work_dir:
        work_dir = Path(reuse_work_dir)
        if work_dir.exists():
            shutil.rmtree(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
    else:
        work_dir = Path(tempfile.mkdtemp(prefix="yolo_v8_seg_dataset_"))

    yolo_dir = work_dir / "yolo"
    dataset_yaml = yolo_dir / "yolo_dataset.yaml"

    if not dataset_yaml.exists():
        work_dir.mkdir(parents=True, exist_ok=True)
        input_dir = _materialize_training_input(dataset_chips, dataset_labels, work_dir)
        preprocessed_dir = Path(preprocess(str(input_dir), str(work_dir), p_val=p_val))
        _ensure_labels_epsg4326(preprocessed_dir / "labels")
        yolo_format(
            input_path=str(preprocessed_dir),
            output_path=str(yolo_dir),
            seed=seed,
            train_split=1.0 - p_val,
            val_split=p_val,
            test_split=0.0,
        )

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
    train_overrides: dict[str, Any] | None = None,
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

    import importlib

    train_mod = cast(Any, importlib.import_module("hot_fair_utilities.training.yolo_v8.train"))

    dataset_yaml = str(Path(yolo_data_dir) / "yolo_dataset.yaml")
    original_profile = dict(getattr(train_mod, "HYPERPARAM_CHANGES", {}))
    if train_overrides:
        # Runtime override of fAIr-utilities training profile so STAC controls
        # optimizer/LR without requiring an fAIr-utilities edit.
        train_mod.HYPERPARAM_CHANGES = {**original_profile, **train_overrides}
    try:
        model_path, iou_accuracy = train_mod.train(
            data=data_base_path,
            weights=weights_path,
            epochs=epochs,
            batch_size=batch_size,
            pc=pc,
            output_path=yolo_data_dir,
            dataset_yaml_path=dataset_yaml,
        )
    finally:
        train_mod.HYPERPARAM_CHANGES = original_profile
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


def _prepare_onnx_image(img_path: Path, input_width: int, input_height: int) -> tuple[Any, Any, Any]:
    import numpy as np
    import rasterio
    from PIL import Image
    from rasterio.crs import CRS
    from rasterio.transform import from_bounds

    with rasterio.open(img_path) as src:
        arr = src.read([1, 2, 3]).astype(np.float32) / 255.0
        transform = src.transform
        crs = src.crs
        src_height = src.height
        src_width = src.width

    # Many OAM TMS endpoints serve JPEG/PNG tiles (no embedded georeference).
    # If the chip has no CRS/transform, derive bounds from the OAM-{x}-{y}-{z} filename
    # so polygon outputs land in the correct lon/lat location.
    # Affine.is_identity is a bool property, not a method.
    if (crs is None) or getattr(transform, "is_identity", False):
        import math
        import re

        m = re.search(r"OAM-(\d+)-(\d+)-(\d+)\.", img_path.name)
        if m:
            x, y, z = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
            n = 2**z
            west = x / n * 360.0 - 180.0
            east = (x + 1) / n * 360.0 - 180.0

            def _lat_deg(tile_y: int) -> float:
                lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * tile_y / n)))
                return lat_rad * 180.0 / math.pi

            north = _lat_deg(y)
            south = _lat_deg(y + 1)
            transform = from_bounds(west, south, east, north, src_width, src_height)
            crs = CRS.from_epsg(4326)

    resized = [
        np.asarray(Image.fromarray(arr[c]).resize((input_width, input_height), Image.Resampling.BILINEAR))
        for c in range(arr.shape[0])
    ]
    batch = np.stack(resized, axis=0)[np.newaxis, ...].astype(np.float32)
    return batch, transform, (src_width, src_height, input_width, input_height, crs)


def _extract_yolo_shapes(session: Any) -> tuple[str, int, int]:
    input_meta = session.get_inputs()[0]
    input_name = input_meta.name
    shape = input_meta.shape
    if len(shape) != 4:
        raise RuntimeError(f"Unexpected ONNX input shape: {shape}")
    input_height = int(shape[2])
    input_width = int(shape[3])
    return input_name, input_width, input_height


def _decode_yoloseg_instances(
    box_output: Any,
    mask_output: Any,
    input_width: int,
    input_height: int,
    src_width: int,
    src_height: int,
    confidence_threshold: float,
    iou_threshold: float,
    num_masks: int = 32,
) -> list[dict[str, Any]]:
    import numpy as np
    from PIL import Image
    from predictor.yoloseg.utils import nms, sigmoid, xywh2xyxy

    predictions = np.squeeze(np.asarray(box_output)).T
    num_classes = int(np.asarray(box_output).shape[1]) - num_masks - 4
    class_scores = predictions[:, 4 : 4 + num_classes]
    class_ids = class_scores.argmax(axis=1)
    scores = class_scores.max(axis=1)

    keep = scores > confidence_threshold
    predictions = predictions[keep]
    scores = scores[keep]
    class_ids = class_ids[keep]
    if len(scores) == 0:
        return []

    box_predictions = predictions[..., : num_classes + 4]
    mask_predictions = predictions[..., num_classes + 4 :]
    boxes = box_predictions[:, :4]
    boxes = boxes / np.array([input_width, input_height, input_width, input_height], dtype=np.float32)
    boxes *= np.array([src_width, src_height, src_width, src_height], dtype=np.float32)
    boxes = xywh2xyxy(boxes)
    boxes[:, 0] = np.clip(boxes[:, 0], 0, src_width)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, src_height)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, src_width)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, src_height)

    indices = nms(boxes, scores, iou_threshold)
    if len(indices) == 0:
        return []
    boxes = boxes[indices]
    mask_predictions = mask_predictions[indices]
    scores = scores[indices]
    class_ids = class_ids[indices]

    proto = np.squeeze(np.asarray(mask_output))
    num_mask, mask_h, mask_w = proto.shape
    masks = sigmoid(mask_predictions @ proto.reshape((num_mask, -1))).reshape((-1, mask_h, mask_w))

    scale_boxes = boxes / np.array([src_width, src_height, src_width, src_height], dtype=np.float32)
    scale_boxes *= np.array([mask_w, mask_h, mask_w, mask_h], dtype=np.float32)

    instances: list[dict[str, Any]] = []
    for i in range(len(scale_boxes)):
        sx1, sy1 = int(np.floor(scale_boxes[i][0])), int(np.floor(scale_boxes[i][1]))
        sx2, sy2 = int(np.ceil(scale_boxes[i][2])), int(np.ceil(scale_boxes[i][3]))
        x1, y1 = int(np.floor(boxes[i][0])), int(np.floor(boxes[i][1]))
        x2, y2 = int(np.ceil(boxes[i][2])), int(np.ceil(boxes[i][3]))
        if x2 <= x1 or y2 <= y1:
            continue

        crop = masks[i][sy1:sy2, sx1:sx2]
        if crop.size == 0:
            continue
        resized_crop = np.asarray(
            Image.fromarray(crop).resize((x2 - x1, y2 - y1), Image.Resampling.BILINEAR),
            dtype=np.float32,
        )
        binary = (resized_crop > 0.5).astype(np.uint8)
        if not binary.any():
            continue

        instance_mask = np.zeros((src_height, src_width), dtype=np.uint8)
        instance_mask[y1:y2, x1:x2] = binary
        instances.append(
            {
                "confidence": float(scores[i]),
                "class": int(class_ids[i]),
                "mask": instance_mask,
            }
        )
    return instances


def _vectorize_instance_mask(
    mask: Any,
    transform: Any,
    crs: Any,
    properties: dict[str, Any],
) -> list[dict[str, Any]]:
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
                "properties": properties,
                "geometry": geom,
            }
        )
    return features


def predict(session: Any, input_images: str, params: dict[str, Any]) -> dict[str, Any]:
    from fair.utils.data import resolve_directory

    if "confidence_threshold" not in params:
        raise ValueError("params['confidence_threshold'] is required")
    confidence_threshold = float(params["confidence_threshold"])
    iou_threshold = float(params.get("iou_threshold", 0.3))
    input_name, input_width, input_height = _extract_yolo_shapes(session)

    input_dir = resolve_directory(input_images)
    patterns = ("*.png", "*.tif", "*.tiff", "*.jpg")
    img_paths = sorted(p for pat in patterns for p in input_dir.glob(pat))
    if not img_paths:
        raise FileNotFoundError(f"No input images found in {input_dir}")

    features: list[dict[str, Any]] = []
    for img_path in img_paths:
        batch, transform, meta = _prepare_onnx_image(img_path, input_width, input_height)
        src_width, src_height, _iw, _ih, crs = meta
        outputs = session.run(None, {input_name: batch})
        if len(outputs) < 2:
            continue
        for instance in _decode_yoloseg_instances(
            outputs[0],
            outputs[1],
            input_width=input_width,
            input_height=input_height,
            src_width=src_width,
            src_height=src_height,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
        ):
            properties = {
                "confidence": round(instance["confidence"], 4),
                "class": instance["class"],
                "source": img_path.name,
            }
            features.extend(
                _vectorize_instance_mask(instance["mask"], transform, crs, properties),
            )
    return _build_feature_collection(features)


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


@step(output_materializers={"trained_model_artifact": CheckpointBytesMaterializer})
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
) -> Annotated[bytes, "trained_model_artifact"]:
    _ = num_classes
    epochs = int(hyperparameters.get("epochs", 20))
    batch_size = int(hyperparameters.get("batch_size", 16))
    pc = float(hyperparameters.get("pc", 2.0))
    optimizer = str(hyperparameters.get("optimizer", "AdamW"))
    learning_rate = float(hyperparameters.get("learning_rate", 0.01))
    train_overrides: dict[str, Any] = {
        "optimizer": optimizer,
        # Ultralytics uses lr0 as the base learning rate argument.
        "lr0": learning_rate,
    }

    yolo_dir = Path(split_info["_yolo_dir"])
    if not (yolo_dir / "yolo_dataset.yaml").exists():
        split_info = _prepare_training_split(
            dataset_chips,
            dataset_labels,
            hyperparameters,
            force_rebuild=True,
            reuse_work_dir=split_info["_work_dir"],
        )
        yolo_dir = Path(split_info["_yolo_dir"])

    weights_path = resolve_model_href(base_model_weights)

    with mlflow_training_context(
        hyperparameters,
        model_name,
        base_model_id,
        dataset_id,
    ):
        model_path, iou_accuracy = train_yolo_model(
            data_base_path=split_info["_work_dir"],
            yolo_data_dir=str(yolo_dir),
            weights_path=weights_path,
            epochs=epochs,
            batch_size=batch_size,
            pc=pc,
            train_overrides=train_overrides,
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
    imgsz = int(hyperparameters.get("imgsz", 256))

    dataset_yaml = Path(split_info["_dataset_yaml"])
    if not dataset_yaml.exists():
        split_info = _prepare_training_split(
            dataset_chips,
            dataset_labels,
            hyperparameters,
            force_rebuild=True,
            reuse_work_dir=split_info["_work_dir"],
        )
        dataset_yaml = Path(split_info["_dataset_yaml"])

    model = _restore_checkpoint(trained_model)
    results = model.val(data=str(dataset_yaml), imgsz=imgsz, split="val", verbose=False)
    metrics = getattr(results, "results_dict", None) or {}
    if not metrics:
        raise RuntimeError("YOLO validation produced no results")

    metrics_dict: dict[str, Any] = {
        "accuracy": float(metrics.get("metrics/mAP50(M)", 0.0)),
        "mean_iou": float(metrics.get("metrics/mAP50-95(M)", 0.0)),
        "precision": float(metrics.get("metrics/precision(M)", 0.0)),
        "recall": float(metrics.get("metrics/recall(M)", 0.0)),
    }
    log_evaluation_results(metrics_dict)
    return metrics_dict


@step(output_materializers={"onnx_model": ONNXMaterializer})
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
    model_uri: str,
    input_images: str,
    inference_params: dict[str, Any],
) -> Annotated[dict[str, Any], "predictions"]:
    from fair.serve.base import load_session

    session = load_session(model_uri)
    return predict(session, input_images, inference_params)


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
    inference_params: dict[str, Any],
    zenml_artifact_version_id: str = "",
) -> dict[str, Any]:
    model = (
        load_model(model_uri=model_uri, zenml_artifact_version_id=zenml_artifact_version_id)
        if zenml_artifact_version_id
        else model_uri
    )
    if isinstance(model, bytes):
        onnx_path = Path(tempfile.mkdtemp(prefix="yolo_v8_seg_onnx_")) / "model.onnx"
        onnx_path.write_bytes(model)
        model = str(onnx_path)
    elif isinstance(model, Path):
        model = str(model)
    elif not isinstance(model, str):
        raise TypeError("inference_pipeline expects model_uri to resolve to an ONNX URI/path string")

    return run_inference(
        model_uri=model,
        input_images=input_images,
        inference_params=inference_params,
    )
