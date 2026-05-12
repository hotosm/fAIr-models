"""Step tests for YOLOv8 segmentation pipeline.

Each test runs the real @step entrypoint flow on toy data:
split -> train -> evaluate -> export_onnx.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

_PRETRAINED_URL = "https://github.com/hotosm/fAIr-utilities/raw/refs/heads/master/yolov8s_v2-seg.pt"


@pytest.fixture(scope="session")
def pretrained_weights(tmp_path_factory: pytest.TempPathFactory) -> str:
    from upath import UPath

    cache = tmp_path_factory.mktemp("yolov8_weights") / "yolov8s_v2-seg.pt"
    cache.write_bytes(UPath(_PRETRAINED_URL).read_bytes())
    return str(cache)


@pytest.fixture(scope="session")
def toy_labels_geojson(toy_labels: Path) -> Path:
    labels_geojson = toy_labels / "labels.geojson"
    assert labels_geojson.is_file()
    return labels_geojson


def test_split_dataset(toy_chips: Path, toy_labels_geojson: Path, base_hyperparameters: dict[str, Any]) -> None:
    from models.yolo_v8_segmentation.pipeline import split_dataset

    hyperparameters = dict(base_hyperparameters)
    hyperparameters.update(
        {
            "training.epochs": 1,
            "training.batch_size": 1,
            "training.val_ratio": 0.25,
            "training.split_seed": 42,
        }
    )
    result = split_dataset.entrypoint(
        dataset_chips=str(toy_chips),
        dataset_labels=str(toy_labels_geojson),
        hyperparameters=hyperparameters,
    )

    assert result["strategy"] == "random"
    assert result["train_count"] > 0
    assert result["val_count"] > 0
    assert "_yolo_dir" in result
    assert "_dataset_yaml" in result
    assert Path(result["_dataset_yaml"]).exists()
    assert (Path(result["_yolo_dir"]) / "images" / "train").is_dir()
    assert (Path(result["_yolo_dir"]) / "images" / "val").is_dir()


def test_train_model(
    toy_chips: Path,
    toy_labels_geojson: Path,
    base_hyperparameters: dict[str, Any],
    pretrained_weights: str,
) -> None:
    from models.yolo_v8_segmentation.pipeline import split_dataset, train_model

    hyperparameters = dict(base_hyperparameters)
    hyperparameters.update(
        {
            "training.epochs": 1,
            "training.batch_size": 1,
            "training.pc": 2.0,
            "training.val_ratio": 0.25,
        }
    )
    split_info = split_dataset.entrypoint(
        dataset_chips=str(toy_chips),
        dataset_labels=str(toy_labels_geojson),
        hyperparameters=hyperparameters,
    )
    model_bytes = train_model.entrypoint(
        dataset_chips=str(toy_chips),
        dataset_labels=str(toy_labels_geojson),
        base_model_weights=pretrained_weights,
        hyperparameters=hyperparameters,
        split_info=split_info,
        num_classes=2,
    )

    assert isinstance(model_bytes, bytes)
    assert len(model_bytes) > 0


def test_evaluate_model(
    toy_chips: Path,
    toy_labels_geojson: Path,
    base_hyperparameters: dict[str, Any],
    pretrained_weights: str,
) -> None:
    from models.yolo_v8_segmentation.pipeline import evaluate_model, split_dataset, train_model

    hyperparameters = dict(base_hyperparameters)
    hyperparameters.update(
        {
            "training.epochs": 1,
            "training.batch_size": 1,
            "training.pc": 2.0,
            "training.val_ratio": 0.25,
            "training.imgsz": 256,
        }
    )
    split_info = split_dataset.entrypoint(
        dataset_chips=str(toy_chips),
        dataset_labels=str(toy_labels_geojson),
        hyperparameters=hyperparameters,
    )
    model_bytes = train_model.entrypoint(
        dataset_chips=str(toy_chips),
        dataset_labels=str(toy_labels_geojson),
        base_model_weights=pretrained_weights,
        hyperparameters=hyperparameters,
        split_info=split_info,
        num_classes=2,
    )
    metrics = evaluate_model.entrypoint(
        trained_model=model_bytes,
        dataset_chips=str(toy_chips),
        dataset_labels=str(toy_labels_geojson),
        hyperparameters=hyperparameters,
        split_info=split_info,
    )

    expected = {"fair:accuracy", "fair:mean_iou", "fair:precision", "fair:recall"}
    assert set(metrics.keys()) == expected


def test_export_onnx(
    toy_chips: Path,
    toy_labels_geojson: Path,
    base_hyperparameters: dict[str, Any],
    pretrained_weights: str,
) -> None:
    import onnx

    from models.yolo_v8_segmentation.pipeline import export_onnx, split_dataset, train_model

    hyperparameters = dict(base_hyperparameters)
    hyperparameters.update(
        {
            "training.epochs": 1,
            "training.batch_size": 1,
            "training.pc": 2.0,
            "training.val_ratio": 0.25,
        }
    )
    split_info = split_dataset.entrypoint(
        dataset_chips=str(toy_chips),
        dataset_labels=str(toy_labels_geojson),
        hyperparameters=hyperparameters,
    )
    model_bytes = train_model.entrypoint(
        dataset_chips=str(toy_chips),
        dataset_labels=str(toy_labels_geojson),
        base_model_weights=pretrained_weights,
        hyperparameters=hyperparameters,
        split_info=split_info,
        num_classes=2,
    )
    exported = export_onnx.entrypoint(trained_model=model_bytes)

    assert isinstance(exported, bytes)
    loaded = onnx.load_from_string(exported)
    onnx.checker.check_model(loaded)
