"""yolo11m-swimming-pools: YOLO11m swimming-pool detector on OAM chips, with the model logic in the pinned `spd-hot`
library; heavy imports stay inside functions so the onnxruntime-only inference image can import `predict`."""

import tempfile
from pathlib import Path
from typing import Annotated, Any

from zenml import log_metadata, pipeline, step

from fair.stac.inputs import IMAGE_INPUT
from fair.utils.data import resolve_directory, resolve_path
from fair.zenml.instrumentation import log_evaluation_results, mlflow_training_context
from fair.zenml.materializers import CheckpointBytesMaterializer, ONNXMaterializer
from fair.zenml.metrics import log_loss_history

CLASS_VALUE = 0
CLASS_LABEL = "swimming_pool"


def predict(session: Any, input_images: dict[str, str], params: dict[str, Any]) -> dict[str, Any]:
    """Module-level predict for `fair.serve.base`: `spd_hot.predict_session`, features renamed to the fAIr schema."""
    from spd_hot import predict_session

    collection = predict_session(session, resolve_directory(input_images[IMAGE_INPUT]), params)
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
    """Chip -> `(float32 tensor (3, S, S), PreprocessMeta)`: RGB, one resize to the model input, divided by 255."""
    from spd_hot import parse_preprocess_params, preprocess_chip

    return preprocess_chip(image_path, parse_preprocess_params(params))


def postprocess(raw_output: Any, params: dict[str, Any] | None = None) -> Any:
    """The ONNX output of one chip -> detections in model-input pixels (confidence filter, NMS, cap 300).

    No CRS is involved here: `spd_hot.predict_session` maps the boxes to chip pixels and georeferences them.
    """
    from spd_hot import decode_detections, parse_postprocess_params

    return decode_detections(raw_output, parse_postprocess_params(params))


def _chip_boxes(chips_dir: Path, labels_href: str) -> dict[str, list[dict[str, Any]]]:
    """The pixel boxes of every chip (`spd_hot.dataset.build_chip_labels`); a chip with no pool is a background example.

    Rebuilt from the dataset by every step, so no temporary file is shared between isolated steps.
    """
    from spd_hot.dataset import build_chip_labels

    if labels_href.endswith((".geojson", ".json")):
        labels = resolve_path(labels_href)
    else:
        labels = resolve_directory(labels_href, "*.geojson")
    return build_chip_labels(chips_dir, labels)


def _onnx_from_checkpoint(checkpoint: bytes) -> bytes:
    from spd_hot.export import export_onnx_bytes
    from ultralytics import YOLO

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "best.pt"
        path.write_bytes(checkpoint)
        return export_onnx_bytes(YOLO(str(path)))


@step
def split_dataset(
    dataset_chips: dict[str, str],
    dataset_labels: str,
    hyperparameters: dict[str, Any],
) -> Annotated[dict[str, Any], "split_info"]:
    """Spatial block split of the chips on OAM tile coordinates (neighbouring chips share pools and surroundings)."""
    from spd_hot import parse_split_params, spatial_split
    from spd_hot.dataset import list_chip_paths

    cfg = parse_split_params(hyperparameters)
    chip_names = [path.name for path in list_chip_paths(resolve_directory(dataset_chips[IMAGE_INPUT], "*.tif*"))]
    if not chip_names:
        raise FileNotFoundError(f"no .tif chips found at {dataset_chips[IMAGE_INPUT]}")
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
    dataset_chips: dict[str, str],
    dataset_labels: str,
    base_model_weights: str,
    hyperparameters: dict[str, Any],
    split_info: dict[str, Any],
    num_classes: int,
    model_name: str | None = None,
    base_model_id: str | None = None,
    dataset_id: str | None = None,
) -> Annotated[bytes, "trained_model_artifact"]:
    """Finetune YOLO11m on the train split with `spd_hot.train`; returns the best epoch as `best.pt` bytes.

    The dataset folder (links to the chips and the label files), the base weights and the runs are made in a temporary
    folder of this step, which is the pod that trains on them, and removed when the step ends.
    """
    import mlflow
    from spd_hot import parse_preprocess_params
    from spd_hot.params import parse_train_params
    from spd_hot.train import train
    from spd_hot.yolo_dataset import write_yolo_dataset
    from upath import UPath

    if num_classes != 1:
        raise ValueError(f"yolo11m-swimming-pools is a single-class detector, got num_classes={num_classes}")

    chips_dir = resolve_directory(dataset_chips[IMAGE_INPUT], "*.tif*")
    boxes = _chip_boxes(chips_dir, dataset_labels)
    train_names = list(split_info["train_chip_names"])
    val_names = list(split_info["val_chip_names"])
    log_metadata(
        metadata={
            "fair/train_data": {
                "train_chips": len(train_names),
                "labelled_train_chips": sum(1 for name in train_names if boxes.get(name)),
                "train_boxes": int(sum(len(boxes.get(name, [])) for name in train_names)),
                "val_chips": len(val_names),
                "labelled_val_chips": sum(1 for name in val_names if boxes.get(name)),
            }
        }
    )

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    def log_epoch(epoch: int, train_loss: float, val_loss: float | None) -> None:
        mlflow.log_metric("train_loss", train_loss, step=epoch - 1)  # ty: ignore[possibly-missing-attribute]
        history["train_loss"].append(train_loss)
        if val_loss is not None:
            mlflow.log_metric("val_loss", val_loss, step=epoch - 1)  # ty: ignore[possibly-missing-attribute]
            history["val_loss"].append(val_loss)

    with tempfile.TemporaryDirectory() as work:
        work_dir = Path(work)
        weights = work_dir / "base.pt"
        weights.write_bytes(UPath(base_model_weights).read_bytes())
        yolo_dir = write_yolo_dataset(
            chips_dir, boxes, train_names=train_names, val_names=val_names, out_dir=work_dir / "yolo"
        )
        with mlflow_training_context(hyperparameters, model_name, base_model_id, dataset_id):
            model = train(
                weights=weights,
                dataset_yaml=yolo_dir / "data.yaml",
                cfg=parse_train_params(hyperparameters),
                pre=parse_preprocess_params(hyperparameters),
                project_dir=work_dir / "runs",
                on_epoch_end=log_epoch,
            )
        if history["val_loss"]:
            log_loss_history(history["train_loss"], history["val_loss"])
        return Path(model.trainer.best).read_bytes()  # the best epoch, which Ultralytics keeps as best.pt


@step
def evaluate_model(
    trained_model: Any,
    dataset_chips: dict[str, str],
    dataset_labels: str,
    hyperparameters: dict[str, Any],
    split_info: dict[str, Any],
    class_names: list[str] | None = None,
) -> Annotated[dict[str, Any], "metrics"]:
    """Box precision/recall/F1 at IoU 0.5 and mAP on the val split, scored through the exported ONNX."""
    import onnxruntime as ort
    from spd_hot.evaluate import score_map, score_session

    chips_dir = resolve_directory(dataset_chips[IMAGE_INPUT], "*.tif*")
    boxes = _chip_boxes(chips_dir, dataset_labels)
    names = split_info["val_chip_names"]
    session = ort.InferenceSession(_onnx_from_checkpoint(trained_model), providers=["CPUExecutionProvider"])
    scores = {
        **score_session(session, images_dir=chips_dir, labels=boxes, names=names, params=hyperparameters)[0.5],
        **score_map(session, images_dir=chips_dir, labels=boxes, names=names, params=hyperparameters),
    }

    metrics = {key: round(scores[key], 4) for key in ("precision", "recall", "f1", "map50", "map50_95")}
    log_evaluation_results(metrics)
    return metrics


@step(output_materializers={"onnx_model": ONNXMaterializer})
def export_onnx(
    trained_model: Any,
    hyperparameters: dict[str, Any],
) -> Annotated[bytes, "onnx_model"]:
    """ONNX with the spd-hot contract: `images (1, 3, 512, 512)` float32 in [0, 1], output `(1, 5, 5376)`."""
    return _onnx_from_checkpoint(trained_model)


@step
def run_inference(
    model_uri: str,
    input_images: dict[str, str],
    inference_params: dict[str, Any],
) -> Annotated[dict[str, Any], "predictions"]:
    from fair.serve.base import load_session

    return predict(load_session(model_uri), input_images, inference_params)


@pipeline
def training_pipeline(
    base_model_weights: str,
    dataset_chips: dict[str, str],
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
    input_images: dict[str, str],
    inference_params: dict[str, Any],
) -> None:
    run_inference(
        model_uri=model_uri,
        input_images=input_images,
        inference_params=inference_params,
    )
