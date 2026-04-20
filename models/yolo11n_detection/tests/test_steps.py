"""Step tests for YOLOv11n building detection.

Each test runs the real @step entrypoint against toy OAM chips (256x256)
and COCO labels; YOLO's trainer handles internal resize to imgsz. No
pipeline-internal mocks. Telemetry sinks (zenml/mlflow) are no-ops via
models/conftest.py::mock_instrumentation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

_PRETRAINED_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt"


@pytest.fixture(scope="session")
def pretrained_weights(tmp_path_factory: pytest.TempPathFactory) -> str:
    from upath import UPath

    cache = tmp_path_factory.mktemp("yolo_weights") / "yolo11n.pt"
    cache.write_bytes(UPath(_PRETRAINED_URL).read_bytes())
    return str(cache)


def test_split_dataset(toy_chips: Path, toy_labels: Path, base_hyperparameters: dict[str, Any]) -> None:
    from models.yolo11n_detection.pipeline import split_dataset

    info = split_dataset.entrypoint(
        dataset_chips=str(toy_chips),
        dataset_labels=str(toy_labels),
        hyperparameters=base_hyperparameters,
    )

    assert info["strategy"] == "random"
    assert info["train_count"] > 0
    assert info["val_count"] > 0
    yolo_dir = Path(info["_yolo_dir"])
    assert (yolo_dir / "data.yaml").exists()
    assert (yolo_dir / "images" / "train").is_dir()
    assert (yolo_dir / "images" / "val").is_dir()


def test_train_model(
    toy_chips: Path,
    toy_labels: Path,
    base_hyperparameters: dict[str, Any],
    pretrained_weights: str,
) -> None:
    from models.yolo11n_detection.pipeline import split_dataset, train_model

    split_info = split_dataset.entrypoint(
        dataset_chips=str(toy_chips),
        dataset_labels=str(toy_labels),
        hyperparameters=base_hyperparameters,
    )
    model_bytes = train_model.entrypoint(
        dataset_chips=str(toy_chips),
        dataset_labels=str(toy_labels),
        base_model_weights=pretrained_weights,
        hyperparameters=base_hyperparameters,
        split_info=split_info,
        num_classes=1,
    )

    assert isinstance(model_bytes, bytes)
    assert len(model_bytes) > 0


def test_evaluate_model(
    toy_chips: Path,
    toy_labels: Path,
    base_hyperparameters: dict[str, Any],
    pretrained_weights: str,
) -> None:
    from models.yolo11n_detection.pipeline import evaluate_model, split_dataset, train_model

    split_info = split_dataset.entrypoint(
        dataset_chips=str(toy_chips),
        dataset_labels=str(toy_labels),
        hyperparameters=base_hyperparameters,
    )
    model_bytes = train_model.entrypoint(
        dataset_chips=str(toy_chips),
        dataset_labels=str(toy_labels),
        base_model_weights=pretrained_weights,
        hyperparameters=base_hyperparameters,
        split_info=split_info,
        num_classes=1,
    )
    metrics = evaluate_model.entrypoint(
        trained_model=model_bytes,
        dataset_chips=str(toy_chips),
        dataset_labels=str(toy_labels),
        hyperparameters=base_hyperparameters,
        split_info=split_info,
    )

    assert set(metrics) == {"accuracy", "mean_iou", "precision", "recall"}


def test_export_onnx(
    toy_chips: Path,
    toy_labels: Path,
    base_hyperparameters: dict[str, Any],
    pretrained_weights: str,
) -> None:
    import onnx

    from models.yolo11n_detection.pipeline import export_onnx, split_dataset, train_model

    split_info = split_dataset.entrypoint(
        dataset_chips=str(toy_chips),
        dataset_labels=str(toy_labels),
        hyperparameters=base_hyperparameters,
    )
    model_bytes = train_model.entrypoint(
        dataset_chips=str(toy_chips),
        dataset_labels=str(toy_labels),
        base_model_weights=pretrained_weights,
        hyperparameters=base_hyperparameters,
        split_info=split_info,
        num_classes=1,
    )
    onnx_bytes = export_onnx.entrypoint(trained_model=model_bytes)

    assert isinstance(onnx_bytes, bytes)
    loaded = onnx.load_from_string(onnx_bytes)
    assert len(loaded.graph.input) == 1
    assert len(loaded.graph.output) == 1
