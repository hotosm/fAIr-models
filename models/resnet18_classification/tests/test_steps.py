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
        patch("models.resnet18_classification.pipeline.mlflow_training_context", _noop_mlflow_ctx),
        patch("models.resnet18_classification.pipeline.log_metadata"),
        patch("fair.zenml.metrics.log_metadata"),
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


def test_trained_model_fixture_returns_model(trained_model: Any) -> None:
    assert trained_model is not None


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
        patch("fair.zenml.metrics.log_metadata"),
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
        patch("fair.zenml.metrics.log_metadata"),
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


def test_pipeline_entrypoints_wire_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    from models.resnet18_classification.pipeline import inference_pipeline, training_pipeline

    calls: list[str] = []

    def fake_split_dataset(**kwargs: Any) -> dict[str, Any]:
        calls.append("split")
        return {"val_ratio": 0.2, "seed": 42}

    def fake_train_model(**kwargs: Any) -> object:
        calls.append("train")
        return object()

    def fake_evaluate_model(**kwargs: Any) -> dict[str, float]:
        calls.append("evaluate")
        return {"accuracy": 1.0}

    def fake_export_onnx(**kwargs: Any) -> bytes:
        calls.append("export")
        return b"onnx"

    def fake_run_inference(**kwargs: Any) -> dict[str, Any]:
        calls.append("infer")
        return {"type": "FeatureCollection", "features": []}

    monkeypatch.setattr("models.resnet18_classification.pipeline.split_dataset", fake_split_dataset)
    monkeypatch.setattr("models.resnet18_classification.pipeline.train_model", fake_train_model)
    monkeypatch.setattr("models.resnet18_classification.pipeline.evaluate_model", fake_evaluate_model)
    monkeypatch.setattr("models.resnet18_classification.pipeline.export_onnx", fake_export_onnx)
    monkeypatch.setattr("models.resnet18_classification.pipeline.run_inference", fake_run_inference)

    training_pipeline.entrypoint(
        base_model_weights="weights.pt",
        dataset_chips="chips",
        dataset_labels="labels.csv",
        num_classes=2,
        hyperparameters={"epochs": 1},
    )
    inference_pipeline.entrypoint(
        model_uri="https://example.com/model.onnx",
        input_images="chips",
        inference_params={"confidence_threshold": 0.5},
    )

    assert calls == ["split", "train", "evaluate", "export", "infer"]


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
        patch("fair.zenml.metrics.log_metadata"),
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
