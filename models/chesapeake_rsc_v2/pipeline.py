"""fAIr ZenML pipelines for a ChesapeakeRSC v2 RGB road model."""

import io
import tempfile
from pathlib import Path
from typing import Annotated, Any

from zenml import log_metadata, pipeline, step

from fair.utils.data import resolve_path
from fair.zenml.instrumentation import log_evaluation_results, mlflow_training_context
from fair.zenml.materializers import CheckpointBytesMaterializer, ONNXMaterializer

from .imagery import chip_paths
from .inference import postprocess as postprocess
from .inference import predict as predict
from .inference import preprocess as preprocess


@step
def split_dataset(
    dataset_chips: dict[str, str], dataset_labels: str, hyperparameters: dict[str, Any]
) -> Annotated[dict[str, Any], "split_info"]:
    from .training import spatial_split

    info = spatial_split(
        chip_paths(dataset_chips["image"]),
        hyperparameters.get("val_ratio", 0.2),
        hyperparameters.get("split_seed", 42),
        hyperparameters.get("block_size", 4),
    )
    log_metadata(metadata={"fair/split": info})
    return info


def _loader(chips: str, labels: str, names: list[str], params: dict[str, Any], shuffle: bool) -> Any:
    from torch.utils.data import DataLoader

    from .training import RoadChips, collate

    dataset = RoadChips(chips, labels, names, params.get("road_width_m", 4.0))
    return DataLoader(dataset, batch_size=params.get("batch_size", 4), shuffle=shuffle, collate_fn=collate)


@step(output_materializers={"trained_model_artifact": CheckpointBytesMaterializer})
def train_model(
    dataset_chips: dict[str, str],
    dataset_labels: str,
    base_model_weights: str,
    hyperparameters: dict[str, Any],
    split_info: dict[str, Any],
    num_classes: int = 2,
    model_name: str | None = None,
    base_model_id: str | None = None,
    dataset_id: str | None = None,
) -> Annotated[bytes, "trained_model_artifact"]:
    import os

    import mlflow
    import torch
    from torch.nn.functional import cross_entropy

    from .training import IGNORE, load_model

    if num_classes != 2:
        raise ValueError("Expected two classes: background and road")
    device = "cuda" if torch.cuda.is_available() and os.getenv("FAIR_FORCE_CPU") != "1" else "cpu"
    loader = _loader(dataset_chips["image"], dataset_labels, split_info["train_chip_names"], hyperparameters, True)
    model = load_model(resolve_path(base_model_weights)).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=hyperparameters.get("learning_rate", 1e-4))
    with mlflow_training_context(hyperparameters, model_name, base_model_id, dataset_id):
        for epoch in range(hyperparameters.get("epochs", 5)):
            total_loss = steps = 0
            for images, labels in loader:
                if not (labels != IGNORE).any():
                    continue
                optimizer.zero_grad()
                loss = cross_entropy(model(images.to(device)), labels.to(device), ignore_index=IGNORE)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                steps += 1
            if not steps:
                raise ValueError("No valid training pixels in the spatial training split")
            mlflow.log_metric("train_loss", total_loss / steps, step=epoch)  # ty: ignore[possibly-missing-attribute]
    output = io.BytesIO()
    torch.save(model.cpu().state_dict(), output)
    return output.getvalue()


@step
def evaluate_model(
    trained_model: bytes,
    dataset_chips: dict[str, str],
    dataset_labels: str,
    hyperparameters: dict[str, Any],
    split_info: dict[str, Any],
    class_names: list[str] | None = None,
) -> Annotated[dict[str, Any], "metrics"]:
    import torch

    from .training import IGNORE, load_model

    loader = _loader(dataset_chips["image"], dataset_labels, split_info["val_chip_names"], hyperparameters, False)
    model = load_model(io.BytesIO(trained_model)).eval()
    tp = predicted = actual = 0
    with torch.inference_mode():
        for images, labels in loader:
            valid = labels != IGNORE
            pred = (model(images).argmax(dim=1) == 1) & valid
            road = labels == 1
            tp += int((pred & road).sum())
            predicted += int(pred.sum())
            actual += int(road.sum())
    metrics = {"road_f1": 2 * tp / max(predicted + actual, 1), "road_iou": tp / max(predicted + actual - tp, 1)}
    log_evaluation_results(metrics)
    return metrics


@step(output_materializers={"onnx_model": ONNXMaterializer})
def export_onnx(trained_model: bytes) -> Annotated[bytes, "onnx_model"]:
    import onnx
    import torch

    from .training import load_model

    model = load_model(io.BytesIO(trained_model)).eval()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.onnx"
        spatial_axes = {0: "batch", 2: "height", 3: "width"}
        torch.onnx.export(
            model,
            (torch.zeros(1, 3, 256, 256),),
            str(path),
            opset_version=18,
            input_names=["image"],
            output_names=["logits"],
            dynamic_axes={"image": spatial_axes, "logits": spatial_axes},
            dynamo=False,
        )
        onnx.checker.check_model(str(path))
        return path.read_bytes()


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
    info = split_dataset(dataset_chips=dataset_chips, dataset_labels=dataset_labels, hyperparameters=hyperparameters)
    trained = train_model(
        dataset_chips=dataset_chips,
        dataset_labels=dataset_labels,
        base_model_weights=base_model_weights,
        hyperparameters=hyperparameters,
        split_info=info,
        num_classes=num_classes,
    )
    evaluate_model(
        trained_model=trained,
        dataset_chips=dataset_chips,
        dataset_labels=dataset_labels,
        hyperparameters=hyperparameters,
        split_info=info,
    )
    export_onnx(trained_model=trained)


@pipeline
def inference_pipeline(
    model_uri: str,
    input_images: dict[str, str],
    inference_params: dict[str, Any],
) -> None:
    run_inference(model_uri=model_uri, input_images=input_images, inference_params=inference_params)
