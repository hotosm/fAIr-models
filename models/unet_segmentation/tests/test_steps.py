"""Step tests for UNet building segmentation.

Each test runs the real @step entrypoint against toy OAM chips and GeoJSON
labels at production chip size (256). No pipeline-internal mocks.
Telemetry sinks (zenml/mlflow) are no-ops via
models/conftest.py::mock_instrumentation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest


@pytest.fixture(scope="session")
def pretrained_weights(tmp_path_factory: pytest.TempPathFactory) -> str:
    import torch
    from torchgeo.models import unet

    cache = tmp_path_factory.mktemp("unet_weights") / "pretrained.pth"
    torch.save(unet(weights=None, classes=2).state_dict(), cache)
    return str(cache)


def test_split_dataset(toy_chips: Path, toy_labels: Path, base_hyperparameters: dict[str, Any]) -> None:
    from models.unet_segmentation.pipeline import split_dataset

    info = split_dataset.entrypoint(
        dataset_chips=str(toy_chips),
        dataset_labels=str(toy_labels),
        hyperparameters=base_hyperparameters,
    )

    assert info["strategy"] == "spatial"
    assert info["train_count"] > 0
    assert info["val_count"] > 0


def test_train_model(
    toy_chips: Path,
    toy_labels: Path,
    base_hyperparameters: dict[str, Any],
    pretrained_weights: str,
) -> None:
    from models.unet_segmentation.pipeline import split_dataset, train_model

    split_info = split_dataset.entrypoint(
        dataset_chips=str(toy_chips),
        dataset_labels=str(toy_labels),
        hyperparameters=base_hyperparameters,
    )
    model = train_model.entrypoint(
        dataset_chips=str(toy_chips),
        dataset_labels=str(toy_labels),
        base_model_weights=pretrained_weights,
        hyperparameters=base_hyperparameters,
        split_info=split_info,
        num_classes=2,
    )

    assert model is not None
    assert hasattr(model, "parameters")
    assert next(model.parameters()).device.type == "cpu"


def test_evaluate_model(
    toy_chips: Path,
    toy_labels: Path,
    base_hyperparameters: dict[str, Any],
    pretrained_weights: str,
) -> None:
    from models.unet_segmentation.pipeline import evaluate_model, split_dataset, train_model

    split_info = split_dataset.entrypoint(
        dataset_chips=str(toy_chips),
        dataset_labels=str(toy_labels),
        hyperparameters=base_hyperparameters,
    )
    model = train_model.entrypoint(
        dataset_chips=str(toy_chips),
        dataset_labels=str(toy_labels),
        base_model_weights=pretrained_weights,
        hyperparameters=base_hyperparameters,
        split_info=split_info,
        num_classes=2,
    )
    metrics = evaluate_model.entrypoint(
        trained_model=model,
        dataset_chips=str(toy_chips),
        dataset_labels=str(toy_labels),
        hyperparameters=base_hyperparameters,
        split_info=split_info,
        num_classes=2,
    )

    assert set(metrics) >= {"accuracy", "mean_iou", "per_class_iou"}
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["mean_iou"] <= 1.0


def test_export_onnx(base_hyperparameters: dict[str, Any]) -> None:
    import onnx
    from torchgeo.models import unet

    from models.unet_segmentation.pipeline import export_onnx

    model = unet(weights=None, classes=2).cpu()
    onnx_bytes = export_onnx.entrypoint(
        trained_model=model,
        hyperparameters=base_hyperparameters,
        num_classes=2,
    )

    assert isinstance(onnx_bytes, bytes)
    loaded = onnx.load_from_string(onnx_bytes)
    assert len(loaded.graph.input) == 1
    assert len(loaded.graph.output) == 1
