# ruff: noqa: UP006, UP007, UP035, UP045
"""ZenML pipeline for YOLOv8-v2 building instance segmentation.

Entrypoints referenced by models/yolo_v8_v2/stac-item.json.
Runtime: PyTorch / Ultralytics, hot-fair-utilities (preprocessing + training + inference).

Implements the fAIr entrypoints:
  - pre_processing_function  → preprocess()
  - post_processing_function → postprocess()
  - mlm:entrypoint (training) → training_pipeline()
  - inference (model from STAC mlm:model asset href) → inference_pipeline()

Model weights: Backend passes model_uri from STAC Item (assets.model.href).
Supports direct HTTP(S) URLs to .pt files and local paths.
Google Drive is not supported; weights should be published to HTTP or staged locally.

Inference uses **hot_fair_utilities.predict** which delegates to fairpredictor internally.
Postprocessing uses **hot_fair_utilities.polygonize** (geomltoolkits-based vectorization).

All heavy imports are lazy: this module is importable in the fAIr-models
host environment where PyTorch, Ultralytics, and GDAL are not installed.
"""

import json
from pathlib import Path
from typing import Annotated, Any, Dict, Optional, Union
from urllib.request import urlretrieve

from zenml import log_metadata, pipeline, step

_DEFAULT_WEIGHTS_CACHE = Path("/workspace/.yolo_weights_cache")
_DEFAULT_YOLO_WEIGHTS_URL = "https://github.com/hotosm/fAIr-utilities/raw/refs/heads/master/yolov8s_v2-seg.pt"


def _resolve_input_directory(path_value: str, purpose: str) -> Path:
    """Resolve local/remote dataset directories to a local path."""
    from fair.utils.data import resolve_directory

    if "://" in str(path_value):
        return resolve_directory(path_value, pattern="*")
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
    cache_dir: Optional[Path] = None,
) -> str:
    """Resolve model_uri to a local .pt checkpoint path.

    Supports:
      - Local path: returned as-is if it exists
      - Direct HTTP(S) URL to .pt file: downloaded, cached

    Returns the absolute path to the .pt checkpoint.
    """
    if not isinstance(model_uri, str):
        raise TypeError("model_uri must be a string")
    if cache_dir is not None and not isinstance(cache_dir, Path):
        raise TypeError("cache_dir must be a pathlib.Path or None")

    cache_dir = cache_dir or _DEFAULT_WEIGHTS_CACHE

    # Local path
    if not (model_uri.startswith("http://") or model_uri.startswith("https://")):
        resolved = _to_local_path(model_uri, "model_uri").resolve()
        if resolved.exists():
            return str(resolved)
        raise FileNotFoundError(f"Model path not found: {resolved}")

    # Remote URL — download and cache
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


def _load_hyperparams_from_stac(stac_item_path: str) -> dict:
    """Load mlm:hyperparameters from a STAC Item JSON file.

    Relative paths are resolved against /workspace (container) or cwd (local).
    """
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


def _get_model_href_from_stac(stac_item_path: str) -> str:
    """Load assets.model.href from a STAC Item JSON file."""
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
    assets = item.get("assets", {})
    model_asset = assets.get("model", {})
    href = model_asset.get("href")
    if not href:
        raise ValueError(f"No assets.model.href found in STAC Item at {stac_item_path}")
    return href


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
                if geometry.get("type") == "Polygon" and coords and coords[0] and coords[0][0] and len(coords[0][0]) >= 2:
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
                    polygons = geometry.get("coordinates", [])  # list[list[list[xy]]]
                elif geometry_type == "Polygon":
                    polygons = [geometry.get("coordinates", [])]  # list[list[xy]]
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
                    if ring and ring[0] and ring[-1] and len(ring[0]) >= 2 and len(ring[-1]) >= 2:
                        if ring[0][0] == ring[-1][0] and ring[0][1] == ring[-1][1]:
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
    yolo_format_module.write_yolo_file = _patched_write_yolo_file
    yolo_utils._fair_models_affine_patch_applied = True


# ---------------------------------------------------------------------------
# Processing-expression callables (referenced by STAC MLM items)
# ---------------------------------------------------------------------------


def preprocess(
    input_path: str,
    output_path: str,
    p_val: float = 0.05,
) -> str:
    """Preprocess OAM chips + labels into a georeferenced, clipped dataset.

    Step 1 — hot_fair_utilities.preprocess: georeference + rasterize + clip labels

    Returns the preprocessed directory path.
    """
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


def postprocess(prediction_path: str, output_geojson: str) -> Dict[str, Any]:
    """Merge predicted-mask GeoTIFF tiles into a single building-footprint GeoJSON.

    Delegates to hot_fair_utilities.polygonize which uses geomltoolkits:
      merge_rasters → vectorize_mask → merge_polygons

    CRS of the resulting GeoJSON is EPSG:4326.
    Returns the parsed GeoJSON content.
    """
    from hot_fair_utilities import polygonize

    polygonize(
        input_path=prediction_path,
        output_path=output_geojson,
        remove_inputs=False,
    )

    with open(output_geojson, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Plain-Python helpers (reusable outside ZenML)
# ---------------------------------------------------------------------------


def train_yolo_model(
    data_base_path: str,
    yolo_data_dir: str,
    weights_path: str,
    epochs: int = 20,
    batch_size: int = 16,
    pc: float = 2.0,
) -> tuple:
    """Fine-tune YOLOv8-v2 segmentation and return (model_path, iou_accuracy).

    Delegates to hot_fair_utilities.training.yolo_v8.train which handles:
      - Weight download if missing
      - YOLOSegWithPosWeight when pc != 0
      - Checkpoint management
      - IoU chart + ONNX export
    """
    # ================= FAIR UTILITIES COMPATIBILITY PATCH =================
    # get_yolo_iou_metrics divides by zero if precision or recall is 0.0 (e.g. in short smoke tests)
    import gc

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
    # ======================================================================

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
    return model_path, iou_accuracy


def infer_yolo_model(
    model_uri: Union[str, Path, Any],
    input_path: str,
    prediction_path: str,
    output_dir: str,
    confidence: float = 0.5,
    model_cache_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Run YOLO instance-segmentation inference and return final GeoJSON content.

    Delegates to hot_fair_utilities.predict which uses fairpredictor internally:
      run_prediction → georeference → move output TIFs

    Then calls postprocess() to generate and parse the GeoJSON.
    """
    from hot_fair_utilities import predict

    cache = Path(model_cache_dir) if model_cache_dir else None

    if isinstance(model_uri, (str, Path)):
        checkpoint_path = resolve_model_href(str(model_uri), cache_dir=cache)
    else:
        # If model_uri is exactly the model object from training, save it out to a temp dir
        import tempfile

        import ultralytics

        if isinstance(model_uri, ultralytics.YOLO):
            with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
                checkpoint_path = tmp.name
                model_uri.save(checkpoint_path)
        else:
            raise TypeError("model_uri must be a str, Path, or an ultralytics.YOLO model.")

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


# ---------------------------------------------------------------------------
# ZenML steps
# ---------------------------------------------------------------------------


@step
def split_dataset(
    preprocessed_path: str,
    output_path: str,
    hyperparameters: Optional[Dict[str, Any]] = None,
) -> Annotated[Dict[str, Any], "split_info"]:
    """Generate the YOLO train/val split and return split metadata.

    `hot_fair_utilities.preprocessing.yolo_v8.yolo_format` performs the actual split and writes:
    - `yolo_dataset.yaml`
    - `images/{train,val,test}` and `labels/{train,val,test}`
    """
    _patch_yolo_write_yolo_file_with_affine()
    from hot_fair_utilities.preprocessing.yolo_v8 import yolo_format

    hyperparameters = hyperparameters or {}
    p_val = float(hyperparameters.get("p_val", 0.2))
    seed = int(hyperparameters.get("split_seed", 42))

    if not 0.0 < p_val < 1.0:
        raise ValueError("p_val must be in (0.0, 1.0)")

    preprocessed_dir = _to_local_path(preprocessed_path, "preprocessed_path")
    if not preprocessed_dir.is_dir():
        raise FileNotFoundError(f"Preprocessed directory not found: {preprocessed_dir}")

    out_dir = _to_local_path(output_path, "output_path")
    yolo_dir = out_dir / "yolo"

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
    test_count = len(list((yolo_dir / "images" / "test").glob("*")))

    split_info: Dict[str, Any] = {
        "strategy": "ratio",
        "val_ratio": p_val,
        "seed": seed,
        "train_count": train_count,
        "val_count": val_count,
        "test_count": test_count,
        "yolo_data_dir": str(yolo_dir),
        "dataset_yaml": str(yolo_dir / "yolo_dataset.yaml"),
    }
    log_metadata(metadata={"fair/split": split_info})
    return split_info


@step
def run_preprocessing(
    input_path: str,
    output_path: str,
    p_val: float = 0.05,
) -> str:
    """Preprocess raw chips + labels. Returns the preprocessed directory path."""
    return preprocess(input_path, output_path, p_val)


@step
def train_model(
    data_base_path: str,
    yolo_data_dir: str,
    weights_path: str,
    epochs: int = 20,
    batch_size: int = 16,
    pc: float = 2.0,
) -> Any:
    """Fine-tune YOLOv8-v2 segmentation on building-footprint chips.

    Returns the loaded Ultralytics YOLO model object. IoU accuracy is logged as ZenML metadata.
    """
    import ultralytics

    model_path, iou_accuracy = train_yolo_model(
        data_base_path=data_base_path,
        yolo_data_dir=yolo_data_dir,
        weights_path=weights_path,
        epochs=epochs,
        batch_size=batch_size,
        pc=pc,
    )
    log_metadata(metadata={"iou_accuracy_pct": float(iou_accuracy), "checkpoint": model_path})
    return ultralytics.YOLO(model_path)


@step
def run_inference(
    model_uri: Union[str, Path, Any],
    input_path: str,
    prediction_path: str,
    output_dir: str,
    confidence: float = 0.5,
    model_cache_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Run YOLO instance-segmentation. model_uri from STAC or training. Returns GeoJSON content."""
    return infer_yolo_model(
        model_uri=model_uri,
        input_path=input_path,
        prediction_path=prediction_path,
        output_dir=output_dir,
        confidence=confidence,
        model_cache_dir=model_cache_dir,
    )


@step
def run_postprocessing(
    prediction_path: str,
    output_geojson: str,
) -> Dict[str, Any]:
    """Polygonize + merge predicted-mask tiles into building-footprint GeoJSON."""
    return postprocess(prediction_path, output_geojson)


# ---------------------------------------------------------------------------
# ZenML pipelines
# ---------------------------------------------------------------------------


@pipeline
def training_pipeline(
    input_path: str,
    output_path: str,
    stac_item_path: str = "models/yolo_v8_v2/stac-item.json",
) -> None:
    """Full training run: preprocess → YOLO format → fine-tune → log IoU.

    Hyperparameters (epochs, batch_size, pc, multimasks, p_val) and base model
    weights (assets.model.href) are loaded from the STAC Item at stac_item_path.
    """
    hyperparams = _load_hyperparams_from_stac(stac_item_path)
    model_href = _get_model_href_from_stac(stac_item_path)
    weights_path = resolve_model_href(model_href)

    epochs = hyperparams.get("epochs", 20)
    batch_size = hyperparams.get("batch_size", 16)
    pc = hyperparams.get("pc", 2.0)
    p_val = hyperparams.get("p_val", 0.2)
    split_seed = hyperparams.get("split_seed", 42)

    preprocessed_dir = run_preprocessing(
        input_path=input_path,
        output_path=output_path,
        p_val=p_val,
    )
    split_info = split_dataset(
        preprocessed_path=preprocessed_dir,
        output_path=output_path,
        hyperparameters={**hyperparams, "p_val": p_val, "split_seed": split_seed},
    )
    train_model(
        data_base_path=output_path,
        yolo_data_dir=split_info["yolo_data_dir"],
        weights_path=weights_path,
        epochs=epochs,
        batch_size=batch_size,
        pc=pc,
    )


@pipeline
def inference_pipeline(
    model_uri: Union[str, Path, Any],
    input_path: str,
    prediction_path: str,
    output_dir: str,
    confidence: Annotated[float, "0.0 <= confidence <= 1.0"] = 0.5,
    model_cache_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Inference: load model → predict → postprocess → return parsed GeoJSON."""
    final_geojson = run_inference(
        model_uri=model_uri,
        input_path=input_path,
        prediction_path=prediction_path,
        output_dir=output_dir,
        confidence=confidence,
        model_cache_dir=model_cache_dir,
    )
    return final_geojson
