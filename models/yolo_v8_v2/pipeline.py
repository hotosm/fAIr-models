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


# ---------------------------------------------------------------------------
# Processing-expression callables (referenced by STAC MLM items)
# ---------------------------------------------------------------------------


def preprocess(
    input_path: str,
    output_path: str,
    p_val: float = 0.05,
) -> str:
    """Preprocess OAM chips + labels, then convert to YOLO dataset format.

    Step 1 — hot_fair_utilities.preprocess: georeference + rasterize + clip labels
    Step 2 — hot_fair_utilities.preprocessing.yolo_v8.yolo_format: convert to YOLO layout

    Returns the YOLO dataset directory path.
    """
    from hot_fair_utilities import preprocess as _preprocess
    from hot_fair_utilities.preprocessing.yolo_v8 import yolo_format

    preprocessed_path = str(Path(output_path) / "preprocessed")
    _preprocess(
        input_path=input_path,
        output_path=preprocessed_path,
        rasterize=True,
        rasterize_options=["binary"],
        georeference_images=True,
        multimasks=False,
    )

    yolo_dir = str(Path(output_path) / "yolo")
    yolo_format(
        input_path=preprocessed_path,
        output_path=yolo_dir,
    )
    return yolo_dir


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

    predict(
        checkpoint_path=checkpoint_path,
        input_path=input_path,
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
def run_preprocessing(
    input_path: str,
    output_path: str,
    p_val: float = 0.05,
) -> str:
    """Preprocess raw chips + labels and write a YOLO dataset. Returns yolo_dir."""
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
    p_val = hyperparams.get("p_val", 0.05)

    yolo_dir = run_preprocessing(
        input_path=input_path,
        output_path=output_path,
        p_val=p_val,
    )
    train_model(
        data_base_path=output_path,
        yolo_data_dir=yolo_dir,
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
