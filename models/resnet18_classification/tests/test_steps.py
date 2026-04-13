"""Step tests for ResNet18 binary building classification pipeline.

Uses toy dataset from each model's data/ directory in models/conftest.py.
Each test calls the step's .entrypoint() directly (no ZenML server needed).
"""

from __future__ import annotations

import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest


def _mock_download_checkpoint(_url: str) -> Path:
    import torch
    from torchvision.models import resnet18

    path = Path(tempfile.mkdtemp()) / "weights.pt"
    torch.save(resnet18(weights=None).state_dict(), path)
    return path


@pytest.fixture()
def split_info(toy_chips: Path, toy_labels: Path, base_hyperparameters: dict[str, Any]) -> dict[str, Any]:
    from models.resnet18_classification.pipeline import split_dataset

    with patch("models.resnet18_classification.pipeline.log_metadata"):
        return split_dataset.entrypoint(
            dataset_chips=str(toy_chips),
            dataset_labels=str(toy_labels),
            hyperparameters=base_hyperparameters,
        )


@pytest.fixture()
def trained_model(
    toy_chips: Path, toy_labels: Path, base_hyperparameters: dict[str, Any], split_info: dict[str, Any]
) -> Any:
    from models.resnet18_classification.pipeline import train_model

    with (
        patch("models.resnet18_classification.pipeline._download_checkpoint", _mock_download_checkpoint),
        patch("models.resnet18_classification.pipeline.mlflow_training_context", _noop_ctx()),
        patch("models.resnet18_classification.pipeline.log_metadata"),
        patch("mlflow.log_metric"),
    ):
        return train_model.entrypoint(
            dataset_chips=str(toy_chips),
            dataset_labels=str(toy_labels),
            base_model_weights="https://example.com/resnet18.pt",
            hyperparameters=base_hyperparameters,
            split_info=split_info,
        )


@contextmanager
def _noop_ctx():
    @contextmanager
    def inner(*_args: Any, **_kwargs: Any):
        yield

    yield from [inner]


@contextmanager
def _noop_mlflow_ctx(*_args: Any, **_kwargs: Any):
    yield


def test_split_dataset(toy_chips: Path, toy_labels: Path, base_hyperparameters: dict[str, Any]) -> None:
    from models.resnet18_classification.pipeline import split_dataset

    with patch("models.resnet18_classification.pipeline.log_metadata"):
        result = split_dataset.entrypoint(
            dataset_chips=str(toy_chips),
            dataset_labels=str(toy_labels),
            hyperparameters=base_hyperparameters,
        )

    assert result["strategy"] == "random"
    assert result["train_count"] > 0
    assert result["val_count"] > 0
    assert result["train_count"] + result["val_count"] == 6


def test_train_model(toy_chips: Path, toy_labels: Path, base_hyperparameters: dict[str, Any]) -> None:
    from models.resnet18_classification.pipeline import split_dataset, train_model

    with patch("models.resnet18_classification.pipeline.log_metadata"):
        si = split_dataset.entrypoint(
            dataset_chips=str(toy_chips),
            dataset_labels=str(toy_labels),
            hyperparameters=base_hyperparameters,
        )

    with (
        patch("models.resnet18_classification.pipeline._download_checkpoint", _mock_download_checkpoint),
        patch("models.resnet18_classification.pipeline.mlflow_training_context", _noop_mlflow_ctx),
        patch("models.resnet18_classification.pipeline.log_metadata"),
        patch("mlflow.log_metric"),
    ):
        model = train_model.entrypoint(
            dataset_chips=str(toy_chips),
            dataset_labels=str(toy_labels),
            base_model_weights="https://example.com/resnet18.pt",
            hyperparameters=base_hyperparameters,
            split_info=si,
        )

    assert model is not None


def test_evaluate_model(toy_chips: Path, toy_labels: Path, base_hyperparameters: dict[str, Any]) -> None:
    from models.resnet18_classification.pipeline import evaluate_model, split_dataset, train_model

    with patch("models.resnet18_classification.pipeline.log_metadata"):
        si = split_dataset.entrypoint(
            dataset_chips=str(toy_chips),
            dataset_labels=str(toy_labels),
            hyperparameters=base_hyperparameters,
        )

    with (
        patch("models.resnet18_classification.pipeline._download_checkpoint", _mock_download_checkpoint),
        patch("models.resnet18_classification.pipeline.mlflow_training_context", _noop_mlflow_ctx),
        patch("models.resnet18_classification.pipeline.log_metadata"),
        patch("mlflow.log_metric"),
    ):
        model = train_model.entrypoint(
            dataset_chips=str(toy_chips),
            dataset_labels=str(toy_labels),
            base_model_weights="https://example.com/resnet18.pt",
            hyperparameters=base_hyperparameters,
            split_info=si,
        )

    with patch("models.resnet18_classification.pipeline.log_evaluation_results"):
        metrics = evaluate_model.entrypoint(
            trained_model=model,
            dataset_chips=str(toy_chips),
            dataset_labels=str(toy_labels),
            hyperparameters=base_hyperparameters,
            split_info=si,
        )

    assert set(metrics.keys()) == {"accuracy", "precision", "recall", "f1"}
    for value in metrics.values():
        assert 0.0 <= value <= 1.0


def test_export_onnx(toy_chips: Path, toy_labels: Path, base_hyperparameters: dict[str, Any]) -> None:
    import onnx

    from models.resnet18_classification.pipeline import export_onnx, split_dataset, train_model

    with patch("models.resnet18_classification.pipeline.log_metadata"):
        si = split_dataset.entrypoint(
            dataset_chips=str(toy_chips),
            dataset_labels=str(toy_labels),
            hyperparameters=base_hyperparameters,
        )

    with (
        patch("models.resnet18_classification.pipeline._download_checkpoint", _mock_download_checkpoint),
        patch("models.resnet18_classification.pipeline.mlflow_training_context", _noop_mlflow_ctx),
        patch("models.resnet18_classification.pipeline.log_metadata"),
        patch("mlflow.log_metric"),
    ):
        model = train_model.entrypoint(
            dataset_chips=str(toy_chips),
            dataset_labels=str(toy_labels),
            base_model_weights="https://example.com/resnet18.pt",
            hyperparameters=base_hyperparameters,
            split_info=si,
        )

    onnx_bytes = export_onnx.entrypoint(trained_model=model, hyperparameters=base_hyperparameters)
    assert isinstance(onnx_bytes, bytes)
    loaded = onnx.load_from_string(onnx_bytes)
    assert len(loaded.graph.input) == 1
    assert len(loaded.graph.output) == 1
