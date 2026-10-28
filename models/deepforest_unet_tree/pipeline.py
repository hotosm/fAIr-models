"""deepforest-unet-tree: tree-crown semantic segmentation on OAM chips (U-Net on the DeepForest NEON_OAM encoder).

The model logic lives in the pinned `ts-hot` library; this module is the fAIr adapter: the contract and the ZenML
wiring. Heavy imports stay inside functions so the onnxruntime-only inference image can import `predict`.
"""

import io
from pathlib import Path
from typing import Annotated, Any

from zenml import log_metadata, pipeline, step

from fair.stac.inputs import IMAGE_INPUT
from fair.utils.data import resolve_directory, resolve_path
from fair.zenml.instrumentation import log_evaluation_results, mlflow_training_context
from fair.zenml.materializers import CheckpointBytesMaterializer, ONNXMaterializer
from fair.zenml.metrics import log_loss_history

CLASS_VALUE = 1
CLASS_LABEL = "tree"
METRIC_KEYS = ("precision", "recall", "f1", "iou")


def _labels_path(dataset_labels: str) -> Path:
    """The labels GeoJSON, given as a file or as a folder holding exactly one."""
    if dataset_labels.endswith((".geojson", ".json")):
        return resolve_path(dataset_labels)
    return resolve_directory(dataset_labels, "*.geojson")


def _chip_shapes(chips_dir: Path, dataset_labels: str) -> dict[str, list[dict[str, Any]]]:
    """Label shapes of every chip (`ts_hot.dataset.build_chip_labels`), rebuilt by each step so none is shared."""
    from ts_hot.dataset import build_chip_labels

    return build_chip_labels(chips_dir, _labels_path(dataset_labels))


def _base_weights_kwargs(weights: bytes) -> dict[str, Any]:
    """The `ts_hot.train` argument for the base weights: a U-Net to continue from, or a DeepForest encoder to start.

    A U-Net checkpoint (saved from a `TreeUNet`) has `up_blocks.*` keys; the NEON_OAM RetinaNet has none.
    """
    import torch

    state = torch.load(io.BytesIO(weights), map_location="cpu", weights_only=True)
    if any(key.startswith("up_blocks.") for key in state):
        return {"checkpoint": io.BytesIO(weights)}
    return {"encoder_checkpoint": io.BytesIO(weights)}


def _onnx_from_checkpoint(checkpoint: bytes) -> bytes:
    from ts_hot.export import build_unet, export_onnx_bytes

    return export_onnx_bytes(build_unet(weights=io.BytesIO(checkpoint)))


def preprocess(image_path: Any, params: dict[str, Any] | None = None) -> Any:
    """Chip -> `(float32 tensor (3, H, W) in [0, 1] scaled to the model's resolution, PreprocessMeta)`."""
    from ts_hot import parse_preprocess_params, preprocess_chip

    return preprocess_chip(image_path, parse_preprocess_params(params))


def postprocess(raw_output: Any, params: dict[str, Any] | None = None) -> Any:
    """The probability map of one chip -> polygons in model-input pixels (threshold, trace, minimum area, simplify).

    No CRS is involved here: `ts_hot.predict_session` georeferences the polygons.
    """
    from ts_hot import parse_postprocess_params
    from ts_hot.decode import decode_polygons

    return decode_polygons(raw_output, parse_postprocess_params(params))


def predict(session: Any, input_images: dict[str, str], params: dict[str, Any]) -> dict[str, Any]:
    """Module-level predict for `fair.serve.base`: `ts_hot.predict_session`, features renamed to the fAIr schema."""
    from ts_hot import predict_session

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


@step
def split_dataset(
    dataset_chips: dict[str, str],
    dataset_labels: str,
    hyperparameters: dict[str, Any],
) -> Annotated[dict[str, Any], "split_info"]:
    """Spatial block split of the chips on OAM tile coordinates (neighbouring chips share canopy)."""
    from ts_hot import parse_split_params, spatial_split
    from ts_hot.dataset import list_chip_paths

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
    """Train the U-Net on the train split with `ts_hot.train`; returns the best epoch as state-dict bytes."""
    import mlflow
    import torch
    from ts_hot import parse_preprocess_params
    from ts_hot.params import parse_train_params
    from ts_hot.train import train
    from upath import UPath

    if num_classes != 1:
        raise ValueError(f"deepforest-unet-tree is a single-class segmenter, got num_classes={num_classes}")

    chips_dir = resolve_directory(dataset_chips[IMAGE_INPUT], "*.tif*")
    shapes = _chip_shapes(chips_dir, dataset_labels)
    train_names = list(split_info["train_chip_names"])
    val_names = list(split_info["val_chip_names"])
    log_metadata(
        metadata={
            "fair/train_data": {
                "train_chips": len(train_names),
                "train_chips_with_trees": sum(1 for name in train_names if shapes.get(name)),
                "val_chips": len(val_names),
            }
        }
    )

    def log_epoch(epoch: int, train_loss: float, val_loss: float | None) -> None:
        mlflow.log_metric("train_loss", train_loss, step=epoch - 1)  # ty: ignore[possibly-missing-attribute]
        if val_loss is not None:
            mlflow.log_metric("val_loss", val_loss, step=epoch - 1)  # ty: ignore[possibly-missing-attribute]

    base_weights = _base_weights_kwargs(UPath(base_model_weights).read_bytes())
    with mlflow_training_context(hyperparameters, model_name, base_model_id, dataset_id):
        result = train(
            train_images_dir=chips_dir,
            train_labels=shapes,
            train_names=train_names,
            val_images_dir=chips_dir if val_names else None,
            val_labels=shapes if val_names else None,
            val_names=val_names or None,
            cfg=parse_train_params({**hyperparameters, "seed": split_info["seed"]}),
            pre=parse_preprocess_params(hyperparameters),
            on_epoch_end=log_epoch,
            **base_weights,
        )
    if result.history["val_loss"]:
        log_loss_history(result.history["train_loss"], result.history["val_loss"])

    buffer = io.BytesIO()
    torch.save(result.model.state_dict(), buffer)
    return buffer.getvalue()


@step
def evaluate_model(
    trained_model: Any,
    dataset_chips: dict[str, str],
    dataset_labels: str,
    hyperparameters: dict[str, Any],
    split_info: dict[str, Any],
    class_names: list[str] | None = None,
) -> Annotated[dict[str, Any], "metrics"]:
    """Pixel precision, recall, F1 and IoU on the val split, scored through the exported ONNX (`score_session`)."""
    import onnxruntime as ort
    from ts_hot import parse_postprocess_params
    from ts_hot.evaluate import score_session

    chips_dir = resolve_directory(dataset_chips[IMAGE_INPUT], "*.tif*")
    shapes = _chip_shapes(chips_dir, dataset_labels)
    threshold = parse_postprocess_params(hyperparameters).threshold
    session = ort.InferenceSession(_onnx_from_checkpoint(trained_model), providers=["CPUExecutionProvider"])
    scores = score_session(
        session,
        images_dir=chips_dir,
        labels=shapes,
        names=split_info["val_chip_names"],
        params=hyperparameters,
        thresholds=(threshold,),
    )[threshold]

    metrics = {key: round(scores[key], 4) for key in METRIC_KEYS}
    log_evaluation_results(metrics)
    return metrics


@step(output_materializers={"onnx_model": ONNXMaterializer})
def export_onnx(
    trained_model: Any,
    hyperparameters: dict[str, Any],
) -> Annotated[bytes, "onnx_model"]:
    """ONNX with the ts-hot contract: `image (3, H, W)` float32 in [0, 1], output `probability (H, W)`."""
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
    inference_params: dict[str, Any] | None = None,
) -> None:
    run_inference(
        model_uri=model_uri,
        input_images=input_images,
        inference_params=inference_params or {},
    )
