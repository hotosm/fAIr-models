"""deepforest-tree: DeepForest RetinaNet-ResNet50 (NEON) tree-crown detector, finetuned on OAM chips.

Everything model-specific lives in the pinned `td-hot` library: scale-aware preprocessing, the ONNX contract (the graph
holds normalisation, anchors and box decoding), confidence filtering and NMS, georeferencing, label boxes, training and
scoring. This module only wires those calls into the ZenML steps and the fAIr serving contract. Heavy imports (torch and
the td_hot training code) stay inside functions, so the distroless inference image (onnxruntime, no torch) can import
this module and reach `predict`.
"""

import io
import os
import shutil
import tempfile
from pathlib import Path
from typing import Annotated, Any

from zenml import log_metadata, pipeline, step

from fair.utils.data import resolve_directory, resolve_path
from fair.zenml.instrumentation import log_evaluation_results, mlflow_training_context
from fair.zenml.materializers import CheckpointBytesMaterializer, ONNXMaterializer
from fair.zenml.metrics import log_loss_history

CLASS_VALUE = 1
CLASS_LABEL = "tree"
IOU_MATCH = 0.5  # box IoU at which a prediction counts as a true positive
MIN_BOX_PX = 2.0  # clipped label boxes thinner than this are degenerate for the box loss
CHIP_SUFFIXES = {".tif", ".tiff"}


def predict(session: Any, input_images: str, params: dict[str, Any]) -> dict[str, Any]:
    """Module-level predict for `fair.serve.base`: `td_hot.predict_session`, features renamed to the fAIr schema."""
    from td_hot import predict_session

    collection = predict_session(session, resolve_directory(input_images), params)
    for feature in collection["features"]:
        props = feature["properties"]
        feature["properties"] = {
            "class": CLASS_VALUE,
            "label": CLASS_LABEL,
            "confidence": props["confidence"],
            "source": props["source"],
        }
    return collection


def preprocess(image_path: Any, params: dict[str, Any] | None = None) -> Any:
    """Chip -> `(float32 tensor (3, H, W) in [0, 1] already scaled to the model's resolution, PreprocessMeta)`."""
    from td_hot import parse_preprocess_params, preprocess_chip

    return preprocess_chip(image_path, parse_preprocess_params(params))


def postprocess(raw_output: Any, params: dict[str, Any] | None = None) -> Any:
    """ONNX outputs of one chip -> detections in model-input pixels (confidence filter, NMS, cap); no CRS involved.

    `raw_output` is `(boxes, scores, labels)`: candidates sorted by score, before NMS.
    """
    from td_hot import decode_detections, parse_postprocess_params

    boxes, scores, labels = raw_output
    return decode_detections(boxes, scores, labels, parse_postprocess_params(params))


def _device() -> str:
    if os.environ.get("FAIR_FORCE_CPU", "").lower() in {"1", "true", "yes", "on"}:
        return "cpu"
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _local_weights(href: str) -> Path:
    if "://" not in href:
        return Path(href)
    from upath import UPath

    source = UPath(href)
    local = Path(tempfile.mkdtemp()) / (source.name or "weights.pt")
    local.write_bytes(source.read_bytes())
    return local


def _resolve_labels_geojson(labels_href: str) -> Path:
    if labels_href.endswith((".geojson", ".json")):
        path = resolve_path(labels_href)
    else:
        root = resolve_directory(labels_href, "*.geojson")
        files = sorted(root.rglob("*.geojson")) or sorted(root.rglob("*.json"))
        if len(files) != 1:
            raise ValueError(f"expected exactly one labels GeoJSON under {labels_href}, found {len(files)}")
        path = files[0]
    if path.suffix.lower() != ".geojson":  # td-hot reads a `.geojson` file
        path = Path(shutil.copy(path, Path(tempfile.mkdtemp()) / "labels.geojson"))
    return path


def _chip_names(chips_dir: Path) -> list[str]:
    return sorted(p.name for p in chips_dir.iterdir() if p.suffix.lower() in CHIP_SUFFIXES)


def _labeled_boxes(chips_dir: Path, labels_href: str) -> dict[str, list[dict[str, Any]]]:
    """Per chip, the pixel boxes of the label polygons (`td_hot.dataset.labels_for_chips`), degenerate ones dropped.

    Rebuilt from the dataset by every step, so no temporary file is shared between isolated steps.
    """
    from td_hot.dataset import labels_for_chips

    boxes = labels_for_chips(chips_dir, _resolve_labels_geojson(labels_href))
    return {
        name: [b for b in rows if b["x2"] - b["x1"] >= MIN_BOX_PX and b["y2"] - b["y1"] >= MIN_BOX_PX]
        for name, rows in boxes.items()
    }


def _labelled(names: list[str], boxes: dict[str, list[dict[str, Any]]]) -> list[str]:
    return sorted(name for name in names if boxes.get(name))


def _restore_model(checkpoint: bytes) -> Any:
    import torch
    from td_hot.export import build_retinanet

    model = build_retinanet()
    model.load_state_dict(torch.load(io.BytesIO(checkpoint), map_location="cpu", weights_only=True), strict=True)
    return model.eval()


@step
def split_dataset(
    dataset_chips: str,
    dataset_labels: str,
    hyperparameters: dict[str, Any],
) -> Annotated[dict[str, Any], "split_info"]:
    """Spatial block split of the chips on OAM tile coordinates (neighbouring chips share trees)."""
    from td_hot import parse_split_params, spatial_split

    cfg = parse_split_params(hyperparameters)
    chip_names = _chip_names(resolve_directory(dataset_chips, "*.tif*"))
    if not chip_names:
        raise FileNotFoundError(f"no .tif chips found at {dataset_chips}")
    train_names, val_names = spatial_split(chip_names, cfg.val_ratio, cfg.split_seed, block_size=cfg.block_size)

    info = {
        "strategy": "spatial",
        "val_ratio": cfg.val_ratio,
        "seed": cfg.split_seed,
        "block_size": cfg.block_size,
        "train_count": len(train_names),
        "val_count": len(val_names),
        "description": (
            f"Spatial block split on OAM-x-y-z tile coordinates ({cfg.block_size}x{cfg.block_size} tile blocks): "
            "whole blocks are assigned to train or validation. Chips must be named OAM-<x>-<y>-<z>.tif; "
            "any other name is rejected."
        ),
        "train_chip_names": train_names,
        "val_chip_names": val_names,
    }
    log_metadata(
        metadata={"fair/split": {k: v for k, v in info.items() if k not in ("train_chip_names", "val_chip_names")}}
    )
    return info


@step(output_materializers={"trained_model_artifact": CheckpointBytesMaterializer})
def train_model(
    dataset_chips: str,
    dataset_labels: str,
    base_model_weights: str,
    hyperparameters: dict[str, Any],
    split_info: dict[str, Any],
    num_classes: int,
    model_name: str | None = None,
    base_model_id: str | None = None,
    dataset_id: str | None = None,
) -> Annotated[bytes, "trained_model_artifact"]:
    """Finetune RetinaNet on the train split with `td_hot.train`; returns the best epoch as `checkpoint.pt` bytes."""
    import mlflow
    import torch
    from td_hot import parse_preprocess_params
    from td_hot.params import parse_train_params
    from td_hot.train import train_retinanet

    if num_classes != 1:
        raise ValueError(f"deepforest-tree is a single-class detector, got num_classes={num_classes}")

    chips_dir = resolve_directory(dataset_chips, "*.tif*")
    boxes = _labeled_boxes(chips_dir, dataset_labels)
    train_names = _labelled(split_info["train_chip_names"], boxes)
    if not train_names:
        raise ValueError("no labelled chips in the train split: check that the labels overlap the chips")
    val_names = _labelled(split_info["val_chip_names"], boxes)
    log_metadata(
        metadata={
            "fair/train_data": {
                "labelled_train_chips": len(train_names),
                "train_boxes": int(sum(len(boxes[n]) for n in train_names)),
                "labelled_val_chips": len(val_names),
            }
        }
    )

    def log_epoch(epoch: int, train_loss: float, val_loss: float | None) -> None:
        mlflow.log_metric("train_loss", train_loss, step=epoch - 1)  # ty: ignore[possibly-missing-attribute]
        if val_loss is not None:
            mlflow.log_metric("val_loss", val_loss, step=epoch - 1)  # ty: ignore[possibly-missing-attribute]

    with mlflow_training_context(hyperparameters, model_name, base_model_id, dataset_id):
        result = train_retinanet(
            train_images_dir=chips_dir,
            train_labels=boxes,
            train_names=train_names,
            val_images_dir=chips_dir if val_names else None,
            val_labels=boxes if val_names else None,
            val_names=val_names or None,
            cfg=parse_train_params({**hyperparameters, "seed": split_info["seed"], "device": _device()}),
            pre=parse_preprocess_params(hyperparameters),
            checkpoint_path=_local_weights(base_model_weights),
            on_epoch_end=log_epoch,
        )
    if result.history["val_loss"]:
        log_loss_history(result.history["train_loss"], result.history["val_loss"])

    buffer = io.BytesIO()
    torch.save(result.model.state_dict(), buffer)
    return buffer.getvalue()


@step
def evaluate_model(
    trained_model: Any,
    dataset_chips: str,
    dataset_labels: str,
    hyperparameters: dict[str, Any],
    split_info: dict[str, Any],
) -> Annotated[dict[str, Any], "metrics"]:
    """Box precision/recall/F1 at IoU 0.5 on the val split, scored through the exported ONNX (`score_session`)."""
    import onnxruntime as ort
    from td_hot.evaluate import score_session
    from td_hot.export import export_onnx_bytes

    chips_dir = resolve_directory(dataset_chips, "*.tif*")
    boxes = _labeled_boxes(chips_dir, dataset_labels)
    session = ort.InferenceSession(export_onnx_bytes(_restore_model(trained_model)), providers=["CPUExecutionProvider"])
    scores = score_session(
        session,
        images_dir=chips_dir,
        labels=boxes,
        names=_labelled(split_info["val_chip_names"], boxes),
        params=hyperparameters,
        iou_thresholds=(IOU_MATCH,),
    )[IOU_MATCH]

    metrics = {key: round(scores[key], 4) for key in ("precision", "recall", "f1")}
    log_evaluation_results(metrics)
    return metrics


@step(output_materializers={"onnx_model": ONNXMaterializer})
def export_onnx(
    trained_model: Any,
    hyperparameters: dict[str, Any],
) -> Annotated[bytes, "onnx_model"]:
    """ONNX with the td-hot contract: `image (3,H,W)` float32 in [0,1], outputs `boxes`, `scores`, `labels`."""
    from td_hot.export import export_onnx_bytes

    return export_onnx_bytes(_restore_model(trained_model))


@step
def run_inference(
    model_uri: str,
    input_images: str,
    inference_params: dict[str, Any],
) -> Annotated[dict[str, Any], "predictions"]:
    from fair.serve.base import load_session

    return predict(load_session(model_uri), input_images, inference_params)


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
    export_onnx(trained_model=trained_model, hyperparameters=hyperparameters)


@pipeline
def inference_pipeline(
    model_uri: str,
    input_images: str,
    inference_params: dict[str, Any],
) -> None:
    run_inference(
        model_uri=model_uri,
        input_images=input_images,
        inference_params=inference_params,
    )
