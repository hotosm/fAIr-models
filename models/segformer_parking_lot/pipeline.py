"""ZenML pipeline for SegFormer parking-lot semantic segmentation.

Thin fAIr adapter over `pl-hot`:
- fAIr contract + ZenML orchestration live here
- model logic lives in `pl_hot.*`
"""

import io
import tempfile
from pathlib import Path
from typing import Annotated, Any

from zenml import log_metadata, pipeline, step

from fair.utils.data import resolve_directory
from fair.zenml.instrumentation import log_evaluation_results, mlflow_training_context
from fair.zenml.materializers import CheckpointBytesMaterializer, ONNXMaterializer
from fair.zenml.metrics import log_loss_history

MODEL_INPUT_SIZE = 512


def _get_device() -> str:
    """Use CUDA when the training image can see a GPU; otherwise CPU."""
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _download_checkpoint(url: str) -> Path:
    """Download an HTTPS/S3 checkpoint to this step's local filesystem."""
    from upath import UPath

    remote = UPath(url)
    local = Path(tempfile.mkdtemp(prefix="segformer_weights_")) / remote.name
    local.write_bytes(remote.read_bytes())
    return local


def _prepare_dataset(
    dataset_chips: str,
    dataset_labels: str,
    hyperparameters: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    """Resolve remote inputs and build a fresh deterministic train/val dataset."""
    from pl_hot.dataset import prepare_seg_dataset_from_geojson
    from pl_hot.params import parse_split_params

    chips_dir = resolve_directory(dataset_chips, "*.tif*")
    labels = resolve_directory(dataset_labels, "*.geojson")
    labels_dir = labels if labels.is_dir() else labels.parent
    prepared_dir = Path(tempfile.mkdtemp(prefix="segformer_parking_dataset_"))
    split_info = prepare_seg_dataset_from_geojson(
        chips_dir,
        labels_dir,
        prepared_dir,
        parse_split_params(hyperparameters),
    )
    return prepared_dir, split_info


def _check_rebuilt_split(rebuilt: dict[str, Any], expected: dict[str, Any]) -> None:
    """Guard that isolated ZenML steps reconstructed the exact same split."""
    for key in ("train_chip_names", "val_chip_names"):
        if rebuilt.get(key) != expected.get(key):
            raise RuntimeError(f"Rebuilt dataset changed {key}; split inputs are not deterministic")


def _serialize_trained_model(model: Any) -> bytes:
    """Serialize finetuned weights for transport between isolated ZenML steps."""
    import torch

    payload = {"state_dict": {f"model.{key}": value.detach().cpu() for key, value in model.state_dict().items()}}
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    return buffer.getvalue()


def _restore_checkpoint(trained_model: Any) -> Any:
    """Restore fine-tuned weights using pl-hot's bundled SegFormer-B5 config."""
    from pl_hot.checkpoint import load_segformer_from_checkpoint

    if hasattr(trained_model, "state_dict") and not isinstance(trained_model, (bytes, bytearray)):
        return trained_model
    if isinstance(trained_model, (bytes, bytearray)):
        checkpoint = Path(tempfile.mkdtemp(prefix="segformer_finetuned_")) / "finetuned_segformer_parking.ckpt"
        checkpoint.write_bytes(bytes(trained_model))
        return load_segformer_from_checkpoint(checkpoint, num_labels=2)
    return load_segformer_from_checkpoint(trained_model, num_labels=2)


def preprocess(image_path: Any) -> Any:
    """STAC preprocessing hook for the fixed 512x512 ONNX input contract."""
    from pl_hot.params import PreprocessParams
    from pl_hot.preprocess import preprocess_chip_for_onnx

    config = PreprocessParams(
        model_input_size=MODEL_INPUT_SIZE,
        normalize_01=True,
        imagenet_norm=True,
    )
    batch, _meta = preprocess_chip_for_onnx(image_path, config)
    return batch


def postprocess(raw_output: Any, params: dict[str, Any] | None = None) -> tuple[Any, Any]:
    """STAC postprocessing hook: decode ONNX logits -> binary mask + parking probability."""
    from pl_hot.decode import decode_segformer_onnx_output
    from pl_hot.params import parse_inference_params, parse_preprocess_params

    preprocess_config = parse_preprocess_params(params)
    inference_config = parse_inference_params(params)
    return decode_segformer_onnx_output(
        raw_output,
        threshold=inference_config.mask_threshold,
        spatial_size=preprocess_config.model_input_size,
    )


def predict(session: Any, input_images: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Serving entrypoint; `params` are STAC `inference.*` keys without the prefix."""
    from pl_hot.params import parse_postprocess_params, parse_preprocess_params
    from pl_hot.postprocess import mask_to_feature_collection
    from pl_hot.preprocess import preprocess_chip_for_onnx
    from pl_hot.serve import iter_image_paths

    images_dir = resolve_directory(input_images)
    preprocess_config = parse_preprocess_params(params)
    postprocess_config = parse_postprocess_params(params)
    input_name = session.get_inputs()[0].name

    features: list[dict[str, Any]] = []
    for image_path in iter_image_paths(images_dir):
        batch, metadata = preprocess_chip_for_onnx(image_path, preprocess_config)
        raw_output = session.run(None, {input_name: batch})[0]
        mask, probability = postprocess(raw_output, params)
        collection = mask_to_feature_collection(
            mask,
            metadata,
            postprocess_config,
            source_name=image_path.name,
            class_name="parking_lot",
            probability=probability,
        )
        features.extend(collection["features"])
    return {"type": "FeatureCollection", "features": features}


@step
def split_dataset(
    dataset_chips: str,
    dataset_labels: str,
    hyperparameters: dict[str, Any],
) -> Annotated[dict[str, Any], "split_info_artifact"]:
    _prepared_dir, split_info = _prepare_dataset(dataset_chips, dataset_labels, hyperparameters)
    split_info["description"] = "Spatial split by OAM tile blocks; whole blocks stay in train or validation."
    log_metadata(metadata={"fair/split": split_info})
    return split_info


@step(output_materializers={"trained_model_artifact": CheckpointBytesMaterializer})
def train_model(
    dataset_chips: str,
    dataset_labels: str,
    base_model_weights: str,
    hyperparameters: dict[str, Any],
    split_info: dict[str, Any],
    num_classes: int = 2,
    model_name: str | None = None,
    base_model_id: str | None = None,
    dataset_id: str | None = None,
) -> Annotated[bytes, "trained_model_artifact"]:
    from pl_hot.params import parse_train_params
    from pl_hot.train import train_segformer

    if num_classes not in (1, 2):
        raise ValueError(f"SegFormer parking expects 1 or 2 classes, got {num_classes}")

    prepared_dir, rebuilt = _prepare_dataset(dataset_chips, dataset_labels, hyperparameters)
    _check_rebuilt_split(rebuilt, split_info)
    train_config = parse_train_params({**hyperparameters, "device": _get_device()})
    checkpoint = _download_checkpoint(base_model_weights)

    with mlflow_training_context(hyperparameters, model_name, base_model_id, dataset_id):
        result = train_segformer(
            train_images_dir=prepared_dir / "train" / "images",
            train_masks_dir=prepared_dir / "train" / "masks",
            val_images_dir=prepared_dir / "val" / "images",
            val_masks_dir=prepared_dir / "val" / "masks",
            cfg=train_config,
            checkpoint_path=checkpoint,
        )
        history = getattr(result, "history", {}) or {}
        train_losses = history.get("train_loss", [])
        val_losses = history.get("val_loss", [])
        if train_losses or val_losses:
            log_loss_history(train_losses, val_losses)

        log_metadata(
            metadata={
                "fair/train": {
                    "best_epoch": int(result.best_epoch),
                    "best_val_loss": float(result.best_val_loss),
                    "device": train_config.device,
                }
            }
        )
    return _serialize_trained_model(result.model)


@step
def evaluate_model(
    trained_model: Any,
    dataset_chips: str,
    dataset_labels: str,
    hyperparameters: dict[str, Any],
    split_info: dict[str, Any],
    class_names: list[str] | None = None,
) -> Annotated[dict[str, Any], "metrics"]:
    from pl_hot.evaluate import evaluate_segformer
    from pl_hot.params import parse_inference_params, parse_preprocess_params

    del class_names
    prepared_dir, rebuilt = _prepare_dataset(dataset_chips, dataset_labels, hyperparameters)
    _check_rebuilt_split(rebuilt, split_info)
    model = _restore_checkpoint(trained_model)
    inference_config = parse_inference_params(hyperparameters)
    metrics = evaluate_segformer(
        model,
        images_dir=prepared_dir / "val" / "images",
        masks_dir=prepared_dir / "val" / "masks",
        threshold=inference_config.mask_threshold,
        preprocess_cfg=parse_preprocess_params(hyperparameters),
    )
    log_evaluation_results(metrics)
    return metrics


@step(output_materializers={"onnx_model": ONNXMaterializer})
def export_onnx(
    trained_model: Any,
    hyperparameters: dict[str, Any],
) -> Annotated[bytes, "onnx_model"]:
    from pl_hot.export import export_onnx_bytes
    from pl_hot.params import parse_train_params

    model = _restore_checkpoint(trained_model)
    model_input_size = parse_train_params(hyperparameters).model_input_size
    return export_onnx_bytes(model, model_input_size=model_input_size)


@step
def run_inference(
    model_uri: str,
    input_images: str,
    inference_params: dict[str, Any] | None = None,
) -> Annotated[dict[str, Any], "predictions"]:
    from pl_hot.params import parse_inference_params, parse_postprocess_params, parse_preprocess_params

    from fair.serve.base import load_session

    params = dict(inference_params or {})
    parse_preprocess_params(params)
    parse_inference_params(params)
    parse_postprocess_params(params)
    return predict(load_session(model_uri), input_images, params)


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
    inference_params: dict[str, Any] | None = None,
) -> None:
    run_inference(
        model_uri=model_uri,
        input_images=input_images,
        inference_params=inference_params or {},
    )
