"""Step tests for ResNet18 binary classification.

Each test runs the real @step entrypoint against toy chips and a CSV label
file at production chip size (256). No pipeline-internal mocks.
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
    from torchvision.models import resnet18

    cache = tmp_path_factory.mktemp("resnet_weights") / "resnet18.pth"
    torch.save(resnet18(weights=None).state_dict(), cache)
    return str(cache)


def test_split_dataset(toy_chips: Path, toy_labels: Path, base_hyperparameters: dict[str, Any]) -> None:
    from models.resnet18_classification.pipeline import split_dataset

    info = split_dataset.entrypoint(
        dataset_chips=str(toy_chips),
        dataset_labels=str(toy_labels),
        hyperparameters=base_hyperparameters,
    )

    assert info["strategy"] == "random"
    assert info["train_count"] > 0
    assert info["val_count"] > 0


def test_train_model(
    toy_chips: Path,
    toy_labels: Path,
    base_hyperparameters: dict[str, Any],
    pretrained_weights: str,
) -> None:
    from models.resnet18_classification.pipeline import split_dataset, train_model

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
    from models.resnet18_classification.pipeline import evaluate_model, split_dataset, train_model

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
    )

    assert set(metrics) == {"accuracy", "precision", "recall", "f1"}
    for value in metrics.values():
        assert 0.0 <= value <= 1.0


def test_export_onnx(base_hyperparameters: dict[str, Any]) -> None:
    import onnx
    import torch.nn as nn
    from torchvision.models import resnet18

    from models.resnet18_classification.pipeline import export_onnx

    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)

    onnx_bytes = export_onnx.entrypoint(
        trained_model=model,
        hyperparameters=base_hyperparameters,
    )

    assert isinstance(onnx_bytes, bytes)
    loaded = onnx.load_from_string(onnx_bytes)
    assert len(loaded.graph.input) == 1
    assert len(loaded.graph.output) == 1
