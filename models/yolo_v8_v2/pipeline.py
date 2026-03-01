"""ZenML pipeline for YOLOv8-v2 building instance segmentation.

Entrypoints referenced by models/yolo_v8_v2/stac-item.json.
Implements fAIr 3.0 contract. model_uri from STAC assets.model.href.
"""

from __future__ import annotations

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
    """Resolve model_uri to local .pt checkpoint path."""
    cache_dir = cache_dir or _DEFAULT_WEIGHTS_CACHE
    path = Path(model_uri)
    if not (model_uri.startswith("http://") or model_uri.startswith("https://")):
        resolved = path.resolve()
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
            raise ImportError("gdown required for Google Drive.") from e
    else:
        urlretrieve(model_uri, dest)
    if not dest.is_file():
        raise RuntimeError(f"Download failed for {model_uri}")
    return str(dest)


# ---------------------------------------------------------------------------
# Processing-expression callables (referenced by STAC MLM items)
# ---------------------------------------------------------------------------


def preprocess(input_path: str, output_path: str, multimasks: bool = True) -> str:
    """Preprocess OAM chips + labels, then convert to YOLO dataset format."""
    from hot_fair_utilities import preprocess as _preprocess
    from hot_fair_utilities.preprocessing.yolo_v8_v2.yolo_format import yolo_format

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
        p_val=0.05,
    )
    return yolo_dir


def postprocess(prediction_path: str, output_geojson: str) -> str:
    """Merge predicted-mask GeoTIFF tiles into a single building-footprint GeoJSON."""
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
) -> str:
    """Preprocess raw chips + labels and write a YOLO dataset. Returns yolo_dir."""
    return preprocess(input_path, output_path, multimasks)


@step
def train_model(
    data_base_path: str,
    yolo_data_dir: str,
    weights_path: str,
    epochs: int,
    batch_size: int,
    pc: float,
) -> str:
    """Fine-tune YOLOv8-v2 segmentation on building-footprint chips."""
    from hot_fair_utilities.training.yolo_v8_v2.train import train as _train

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
    """Run YOLO instance-segmentation. model_uri from STAC."""
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
    weights_path: str,
    epochs: Annotated[int, Ge(1), Le(500)] = 20,
    batch_size: Annotated[int, Ge(1), Le(64)] = 16,
    pc: Annotated[float, Ge(0.0), Le(10.0)] = 2.0,
    multimasks: bool = True,
) -> None:
    """Full training run: preprocess → YOLO format → fine-tune → log IoU."""
    yolo_dir = run_preprocessing(
        input_path=input_path,
        output_path=output_path,
        multimasks=multimasks,
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
    """Inference: model from STAC → predict → polygonize."""
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
