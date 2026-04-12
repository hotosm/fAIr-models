"""Step tests for YOLOv11n building detection pipeline.

Uses toy PNG chips and COCO JSON labels from each model's data/ directory.
Each test calls the step's .entrypoint() directly (no ZenML server needed).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch


def test_split_dataset(toy_chips: Path, toy_labels: Path, base_hyperparameters: dict[str, Any]) -> None:
    from models.yolo11n_detection.pipeline import split_dataset

    with patch("models.yolo11n_detection.pipeline.log_metadata"):
        result = split_dataset.entrypoint(
            dataset_chips=str(toy_chips),
            dataset_labels=str(toy_labels),
            hyperparameters=base_hyperparameters,
        )

    assert result["strategy"] == "random"
    assert result["train_count"] > 0
    assert result["val_count"] > 0
    assert "_yolo_dir" in result
    yolo_dir = Path(result["_yolo_dir"])
    assert (yolo_dir / "data.yaml").exists()
    assert (yolo_dir / "images" / "train").is_dir()
    assert (yolo_dir / "images" / "val").is_dir()


def test_train_model(toy_chips: Path, toy_labels: Path, base_hyperparameters: dict[str, Any]) -> None:
    from models.yolo11n_detection.pipeline import split_dataset, train_model

    with patch("models.yolo11n_detection.pipeline.log_metadata"):
        si = split_dataset.entrypoint(
            dataset_chips=str(toy_chips),
            dataset_labels=str(toy_labels),
            hyperparameters=base_hyperparameters,
        )

    from contextlib import contextmanager

    @contextmanager
    def _noop_mlflow_ctx(*_args: Any, **_kwargs: Any):
        yield

    with (
        patch("models.yolo11n_detection.pipeline.mlflow_training_context", _noop_mlflow_ctx),
        patch("models.yolo11n_detection.pipeline.log_metadata"),
    ):
        model_bytes = train_model.entrypoint(
            dataset_chips=str(toy_chips),
            dataset_labels=str(toy_labels),
            base_model_weights="yolo11n.pt",
            hyperparameters=base_hyperparameters,
            split_info=si,
            num_classes=1,
        )

    assert isinstance(model_bytes, bytes)
    assert len(model_bytes) > 0


def test_evaluate_model(toy_chips: Path, toy_labels: Path, base_hyperparameters: dict[str, Any]) -> None:
    from models.yolo11n_detection.pipeline import evaluate_model, split_dataset, train_model

    with patch("models.yolo11n_detection.pipeline.log_metadata"):
        si = split_dataset.entrypoint(
            dataset_chips=str(toy_chips),
            dataset_labels=str(toy_labels),
            hyperparameters=base_hyperparameters,
        )

    from contextlib import contextmanager

    @contextmanager
    def _noop_mlflow_ctx(*_args: Any, **_kwargs: Any):
        yield

    with (
        patch("models.yolo11n_detection.pipeline.mlflow_training_context", _noop_mlflow_ctx),
        patch("models.yolo11n_detection.pipeline.log_metadata"),
    ):
        model_bytes = train_model.entrypoint(
            dataset_chips=str(toy_chips),
            dataset_labels=str(toy_labels),
            base_model_weights="yolo11n.pt",
            hyperparameters=base_hyperparameters,
            split_info=si,
            num_classes=1,
        )

    with patch("models.yolo11n_detection.pipeline.log_evaluation_results"):
        metrics = evaluate_model.entrypoint(
            trained_model=model_bytes,
            dataset_chips=str(toy_chips),
            dataset_labels=str(toy_labels),
            hyperparameters=base_hyperparameters,
            split_info=si,
        )

    expected_keys = {"accuracy", "mean_iou", "precision", "recall"}
    assert expected_keys == set(metrics.keys())


def test_export_onnx(toy_chips: Path, toy_labels: Path, base_hyperparameters: dict[str, Any]) -> None:
    import onnx

    from models.yolo11n_detection.pipeline import export_onnx, split_dataset, train_model

    with patch("models.yolo11n_detection.pipeline.log_metadata"):
        si = split_dataset.entrypoint(
            dataset_chips=str(toy_chips),
            dataset_labels=str(toy_labels),
            hyperparameters=base_hyperparameters,
        )

    from contextlib import contextmanager

    @contextmanager
    def _noop_mlflow_ctx(*_args: Any, **_kwargs: Any):
        yield

    with (
        patch("models.yolo11n_detection.pipeline.mlflow_training_context", _noop_mlflow_ctx),
        patch("models.yolo11n_detection.pipeline.log_metadata"),
    ):
        model_bytes = train_model.entrypoint(
            dataset_chips=str(toy_chips),
            dataset_labels=str(toy_labels),
            base_model_weights="yolo11n.pt",
            hyperparameters=base_hyperparameters,
            split_info=si,
            num_classes=1,
        )

    onnx_path = export_onnx.entrypoint(trained_model=model_bytes)
    assert Path(onnx_path).exists()
    loaded = onnx.load(onnx_path)
    assert len(loaded.graph.input) == 1
    assert len(loaded.graph.output) == 1
