"""Step tests for YOLOv11n building detection pipeline.

Uses toy PNG chips and COCO JSON labels from each model's data/ directory.
Each test calls the step's .entrypoint() directly (no ZenML server needed).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch


def _mock_download_checkpoint(url: str) -> Path:
    return Path(url)


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
        patch("models.yolo11n_detection.pipeline._download_checkpoint", _mock_download_checkpoint),
        patch("models.yolo11n_detection.pipeline.mlflow_training_context", _noop_mlflow_ctx),
        patch("models.yolo11n_detection.pipeline.log_metadata"),
        patch("fair.zenml.metrics.log_metadata"),
        patch("mlflow.log_metric"),
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
        patch("models.yolo11n_detection.pipeline._download_checkpoint", _mock_download_checkpoint),
        patch("models.yolo11n_detection.pipeline.mlflow_training_context", _noop_mlflow_ctx),
        patch("models.yolo11n_detection.pipeline.log_metadata"),
        patch("fair.zenml.metrics.log_metadata"),
        patch("mlflow.log_metric"),
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


def test_pipeline_entrypoints_wire_steps(monkeypatch: Any) -> None:
    from models.yolo11n_detection.pipeline import inference_pipeline, training_pipeline

    calls: list[str] = []

    def fake_split_dataset(**kwargs: Any) -> dict[str, Any]:
        calls.append("split")
        return {"_yolo_dir": "/tmp/demo", "val_ratio": 0.2}

    def fake_train_model(**kwargs: Any) -> bytes:
        calls.append("train")
        return b"weights"

    def fake_evaluate_model(**kwargs: Any) -> dict[str, float]:
        calls.append("evaluate")
        return {"accuracy": 1.0}

    def fake_export_onnx(**kwargs: Any) -> bytes:
        calls.append("export")
        return b"onnx"

    def fake_run_inference(**kwargs: Any) -> dict[str, Any]:
        calls.append("infer")
        return {"type": "FeatureCollection", "features": []}

    monkeypatch.setattr("models.yolo11n_detection.pipeline.split_dataset", fake_split_dataset)
    monkeypatch.setattr("models.yolo11n_detection.pipeline.train_model", fake_train_model)
    monkeypatch.setattr("models.yolo11n_detection.pipeline.evaluate_model", fake_evaluate_model)
    monkeypatch.setattr("models.yolo11n_detection.pipeline.export_onnx", fake_export_onnx)
    monkeypatch.setattr("models.yolo11n_detection.pipeline.run_inference", fake_run_inference)

    training_pipeline.entrypoint(
        base_model_weights="weights.pt",
        dataset_chips="chips",
        dataset_labels="labels.json",
        num_classes=1,
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
        patch("models.yolo11n_detection.pipeline._download_checkpoint", _mock_download_checkpoint),
        patch("models.yolo11n_detection.pipeline.mlflow_training_context", _noop_mlflow_ctx),
        patch("models.yolo11n_detection.pipeline.log_metadata"),
        patch("fair.zenml.metrics.log_metadata"),
        patch("mlflow.log_metric"),
    ):
        model_bytes = train_model.entrypoint(
            dataset_chips=str(toy_chips),
            dataset_labels=str(toy_labels),
            base_model_weights="yolo11n.pt",
            hyperparameters=base_hyperparameters,
            split_info=si,
            num_classes=1,
        )

    onnx_bytes = export_onnx.entrypoint(trained_model=model_bytes)
    assert isinstance(onnx_bytes, bytes)
    loaded = onnx.load_from_string(onnx_bytes)
    assert len(loaded.graph.input) == 1
    assert len(loaded.graph.output) == 1
