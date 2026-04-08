"""End-to-end pipeline test for ResNet18 classification.

Verifies the full pipeline contract: split -> train -> evaluate -> export_onnx.
Uses synthetic data and mocks ZenML/MLflow instrumentation.
"""

from __future__ import annotations

import csv
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

torch = pytest.importorskip("torch")
onnx = pytest.importorskip("onnx")


@pytest.fixture()
def synthetic_dataset(tmp_path: Path) -> tuple[str, str]:
    chips_dir = tmp_path / "chips"
    chips_dir.mkdir()

    labels_csv = tmp_path / "labels.csv"
    rows: list[dict[str, str]] = []

    for i in range(20):
        img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        fname = f"chip_{i:03d}.png"
        from PIL import Image

        Image.fromarray(img).save(chips_dir / fname)
        label = "building" if i < 10 else "no_building"
        rows.append({"filename": fname, "class_name": label})

    with open(labels_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "class_name"])
        writer.writeheader()
        writer.writerows(rows)

    return str(chips_dir), str(labels_csv)


@contextmanager
def _noop_ctx(*_args, **_kwargs):
    yield


@pytest.mark.slow
@patch("models.resnet18_classification.pipeline.mlflow_training_context", _noop_ctx)
@patch("models.resnet18_classification.pipeline.log_evaluation_results")
@patch("models.resnet18_classification.pipeline.log_metadata")
@patch("mlflow.log_metric")
def test_resnet18_train_evaluate_export(
    _mock_mlflow_metric,
    _mock_log_meta,
    _mock_log_eval,
    synthetic_dataset: tuple[str, str],
    tmp_path: Path,
) -> None:
    from models.resnet18_classification.pipeline import evaluate_model, export_onnx, split_dataset, train_model

    chips_path, labels_path = synthetic_dataset
    hyperparameters = {
        "epochs": 2,
        "batch_size": 4,
        "learning_rate": 0.01,
        "chip_size": 32,
        "val_ratio": 0.3,
        "split_seed": 42,
        "scheduler": "cosine",
        "max_grad_norm": 1.0,
    }

    split_info = split_dataset.entrypoint(
        dataset_chips=chips_path,
        dataset_labels=labels_path,
        hyperparameters=hyperparameters,
    )
    assert split_info["strategy"] == "random"
    assert split_info["train_count"] > 0
    assert split_info["val_count"] > 0
    assert split_info["train_count"] + split_info["val_count"] == 20

    trained_model = train_model.entrypoint(
        dataset_chips=chips_path,
        dataset_labels=labels_path,
        base_model_weights="IMAGENET1K_V1",
        hyperparameters=hyperparameters,
        split_info=split_info,
    )
    assert trained_model is not None

    metrics = evaluate_model.entrypoint(
        trained_model=trained_model,
        dataset_chips=chips_path,
        dataset_labels=labels_path,
        hyperparameters=hyperparameters,
        split_info=split_info,
    )
    assert set(metrics.keys()) == {"accuracy", "precision", "recall", "f1"}
    for value in metrics.values():
        assert 0.0 <= value <= 1.0

    onnx_path = export_onnx.entrypoint(
        trained_model=trained_model,
        hyperparameters=hyperparameters,
    )
    assert Path(onnx_path).exists()
    model = onnx.load(onnx_path)
    assert len(model.graph.input) == 1
    assert len(model.graph.output) == 1
