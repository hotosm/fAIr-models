"""ZenML pipeline for YOLOv8-v1 building instance segmentation.

Entrypoints referenced by models/yolo_v8_v1/stac-item.json.
Implements fAIr 3.0 contract (FAIr_3.0_Optimized_Pipeline.md).

Model weights (model_uri): From STAC assets.model.href.
Supports local path, HTTP(S) URL, Google Drive. Resolved on first use.

Hyperparameters (epochs, batch_size, pc, multimasks, p_val): Loaded from STAC
properties.mlm:hyperparameters. Use stac_item_path to point at the Item JSON.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Annotated
from urllib.request import urlretrieve

from annotated_types import Ge, Le
from zenml import log_metadata, pipeline, step

_DEFAULT_WEIGHTS_CACHE = Path("/workspace/.yolo_weights_cache")
_GDRIVE_FILE_RE = re.compile(
    r"https?://drive\.google\.com/file/d/([a-zA-Z0-9_-]+)",
    re.IGNORECASE,
)


def resolve_model_href(model_uri: str, cache_dir: Path | None = None) -> str:
    """Resolve model_uri to local .pt checkpoint path.

    Supports: local path, HTTP(S) URL, Google Drive file URL.
    """
    cache_dir = cache_dir or _DEFAULT_WEIGHTS_CACHE
    path = Path(model_uri)

    if not (model_uri.startswith("http://") or model_uri.startswith("https://")):
        resolved = path.resolve()
        if resolved.is_file() and resolved.suffix == ".pt":
            return str(resolved)
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

    gdrive_match = _GDRIVE_FILE_RE.search(model_uri)
    if gdrive_match:
        try:
            import gdown

            gdown.download(id=gdrive_match.group(1), output=str(dest), quiet=True)
        except ImportError as e:
            raise ImportError("gdown required for Google Drive. Add to Dockerfile.") from e
    else:
        urlretrieve(model_uri, dest)

    if not dest.is_file():
        raise RuntimeError(f"Download failed for {model_uri}")
    return str(dest)


def _load_hyperparams_from_stac(stac_item_path: str) -> dict:
    """Load mlm:hyperparameters from a STAC Item JSON file.

    Relative paths are resolved against /workspace (container) or cwd (local).
    """
    path = Path(stac_item_path)
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
    path = Path(stac_item_path)
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
    multimasks: bool = True,
    p_val: float = 0.05,
) -> str:
    """Preprocess OAM chips + labels, then convert to YOLO dataset format.

    Step 1 — hot_fair_utilities.preprocess:
        Georeferences PNGs, reprojects labels to EPSG:3857, clips labels to
        chip extents, and (if multimasks=True) generates the three-channel
        RAMP-style multimask (footprint / boundary / contact).

    Step 2 — yolo_format:
        Converts the preprocessed chips and masks into an Ultralytics YOLO
        segmentation dataset (images/, labels/, yolo_dataset.yaml).

    Returns the path to the YOLO data directory.
    """
    from hot_fair_utilities import preprocess as _preprocess
    from hot_fair_utilities.preprocessing.yolo_v8_v1.yolo_format import yolo_format

    preprocessed_path = str(Path(output_path) / "preprocessed")
    _preprocess(
        input_path=input_path,
        output_path=preprocessed_path,
        rasterize=True,
        rasterize_options=["binary"],
        georeference_images=True,
        multimasks=multimasks,
    )

    yolo_dir = str(Path(output_path) / "yolo")
    yolo_format(
        preprocessed_dirs=preprocessed_path,
        yolo_dir=yolo_dir,
        multimask=multimasks,
        p_val=p_val,
    )
    return yolo_dir


def postprocess(prediction_path: str, output_geojson: str) -> str:
    """Merge predicted-mask GeoTIFF tiles into a single building-footprint GeoJSON.

    Uses the two-step AutoBFE algorithm from hot_fair_utilities:
        1. Polygonize each tile individually (GDAL/rasterio).
        2. Buffer + spatial-join neighbouring polygons to merge fragments
           that straddle tile boundaries (EPSG:4326 output).
    """
    from hot_fair_utilities import polygonize

    polygonize(
        input_path=prediction_path,
        output_path=output_geojson,
        remove_inputs=False,
    )
    return output_geojson


# ---------------------------------------------------------------------------
# ZenML steps
# ---------------------------------------------------------------------------


@step
def run_preprocessing(
    input_path: str,
    output_path: str,
    multimasks: bool = True,
    p_val: float = 0.05,
) -> str:
    """Preprocess raw chips + labels and write a YOLO dataset. Returns yolo_dir."""
    return preprocess(input_path, output_path, multimasks, p_val)


@step
def train_model(
    data_base_path: str,
    yolo_data_dir: str,
    weights_path: str,
    epochs: int,
    batch_size: int,
    pc: float,
) -> str:
    """Fine-tune YOLOv8-v1 segmentation on building-footprint chips.

    If weights_path does not exist on disk the base weights are downloaded
    automatically from the fAIr-utilities GitHub release.

    Returns the path to the best checkpoint (.pt file).
    IoU accuracy (0-100 %) is logged as ZenML step metadata.
    """
    from hot_fair_utilities.training.yolo_v8_v1.train import train as _train

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
    log_metadata(metadata={"iou_accuracy_pct": float(iou_accuracy), "checkpoint": model_path})
    return model_path


@step
def run_inference(
    model_uri: str,
    input_path: str,
    prediction_path: str,
    confidence: float = 0.5,
    model_cache_dir: str | None = None,
) -> str:
    """Run YOLO instance-segmentation. model_uri from STAC (local/HTTP/GDrive)."""
    from hot_fair_utilities import predict

    cache = Path(model_cache_dir) if model_cache_dir else None
    checkpoint_path = resolve_model_href(model_uri, cache_dir=cache)
    predict(
        checkpoint_path=checkpoint_path,
        input_path=input_path,
        prediction_path=prediction_path,
        confidence=confidence,
    )
    return prediction_path


@step
def run_postprocessing(
    prediction_path: str,
    output_geojson: str,
) -> str:
    """Polygonize + merge predicted-mask tiles into building-footprint GeoJSON."""
    return postprocess(prediction_path, output_geojson)


# ---------------------------------------------------------------------------
# ZenML pipelines
# ---------------------------------------------------------------------------


@pipeline
def training_pipeline(
    input_path: str,
    output_path: str,
    stac_item_path: str = "models/yolo_v8_v1/stac-item.json",
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
    multimasks = hyperparams.get("multimasks", True)
    p_val = hyperparams.get("p_val", 0.05)

    yolo_dir = run_preprocessing(
        input_path=input_path,
        output_path=output_path,
        multimasks=multimasks,
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
    model_uri: str,
    input_path: str,
    prediction_path: str,
    output_geojson: str,
    confidence: Annotated[float, Ge(0.0), Le(1.0)] = 0.5,
    model_cache_dir: str | None = None,
) -> None:
    """Inference: load model (from STAC) → predict → polygonize. model_uri=assets.model.href."""
    predictions = run_inference(
        model_uri=model_uri,
        input_path=input_path,
        prediction_path=prediction_path,
        confidence=confidence,
        model_cache_dir=model_cache_dir,
    )
    run_postprocessing(
        prediction_path=predictions,
        output_geojson=output_geojson,
    )
