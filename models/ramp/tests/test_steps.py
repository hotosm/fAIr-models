"""Step tests for the RAMP pipeline.

Each test calls ``step.entrypoint(...)`` directly. Heavy RAMP/TF operations
(``train_ramp_model``, SavedModel loading, tf2onnx conversion) are patched so
tests run quickly and do not require GPU resources.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch


@contextmanager
def _noop_mlflow_ctx(*_args: Any, **_kwargs: Any):
    yield


def test_split_dataset(toy_chips: Path, toy_labels: Path, base_hyperparameters: dict[str, Any]) -> None:
    from models.ramp.pipeline import split_dataset

    hyperparameters = dict(base_hyperparameters)
    hyperparameters.update(
        {
            "epochs": 1,
            "batch_size": 1,
            "val_fraction": 0.25,
            "split_seed": 42,
            "boundary_width": 1,
            "contact_spacing": 2,
        }
    )

    with patch("models.ramp.pipeline.log_metadata"):
        result = split_dataset.entrypoint(
            dataset_chips=str(toy_chips),
            dataset_labels=str(toy_labels),
            hyperparameters=hyperparameters,
        )

    assert result["strategy"] == "random"
    assert result["train_count"] > 0
    assert result["val_count"] > 0
    assert "_ramp_train_dir" in result
    assert "_preprocessed_dir" in result
    ramp_dir = Path(result["_ramp_train_dir"])
    assert ramp_dir.exists()
    assert (ramp_dir / "chips").is_dir()
    assert (ramp_dir / "val-chips").is_dir()


def test_train_model(toy_chips: Path, toy_labels: Path, base_hyperparameters: dict[str, Any], tmp_path: Path) -> None:
    from models.ramp.pipeline import train_model

    ramp_train_dir = tmp_path / "ramp_training_work"
    (ramp_train_dir / "chips").mkdir(parents=True)
    (ramp_train_dir / "val-chips").mkdir(parents=True)

    fake_saved_model_dir = tmp_path / "saved_model"
    fake_saved_model_dir.mkdir()
    (fake_saved_model_dir / "saved_model.pb").write_bytes(b"\x08\x01")  # magic-enough stub

    split_info = {
        "_work_dir": str(tmp_path),
        "_preprocessed_dir": str(tmp_path / "preprocessed"),
        "_ramp_train_dir": str(ramp_train_dir),
        "strategy": "random",
        "val_ratio": 0.25,
        "seed": 42,
        "train_count": 3,
        "val_count": 1,
        "description": "test split",
    }
    hyperparameters = dict(base_hyperparameters)
    hyperparameters.update({"epochs": 1, "batch_size": 1})

    with (
        patch("models.ramp.pipeline.mlflow_training_context", _noop_mlflow_ctx, create=True),
        patch("models.ramp.pipeline.train_ramp_model", return_value=fake_saved_model_dir),
        patch("models.ramp.pipeline._zip_savedmodel_dir", return_value=b"fake-savedmodel-zip"),
        patch("models.ramp.pipeline.log_metadata"),
    ):
        model_bytes = train_model.entrypoint(
            dataset_chips=str(toy_chips),
            dataset_labels=str(toy_labels),
            base_model_weights="https://example.com/baseline.zip",
            hyperparameters=hyperparameters,
            split_info=split_info,
            num_classes=4,
        )

    assert isinstance(model_bytes, bytes)
    assert model_bytes == b"fake-savedmodel-zip"


def test_evaluate_model(
    toy_chips: Path, toy_labels: Path, base_hyperparameters: dict[str, Any], tmp_path: Path
) -> None:
    from models.ramp.pipeline import evaluate_model

    ramp_train_dir = tmp_path / "ramp_training_work"
    (ramp_train_dir / "chips").mkdir(parents=True)
    (ramp_train_dir / "val-chips").mkdir(parents=True)
    (ramp_train_dir / "val-multimasks").mkdir(parents=True)

    split_info = {
        "_work_dir": str(tmp_path),
        "_preprocessed_dir": str(tmp_path / "preprocessed"),
        "_ramp_train_dir": str(ramp_train_dir),
        "strategy": "random",
        "val_ratio": 0.25,
        "seed": 42,
        "train_count": 3,
        "val_count": 1,
        "description": "test split",
    }

    fake_saved_model_dir = tmp_path / "saved_model"
    fake_saved_model_dir.mkdir()
    (fake_saved_model_dir / "saved_model.pb").write_bytes(b"\x08\x01")

    with (
        patch("models.ramp.pipeline.mlflow_training_context", _noop_mlflow_ctx, create=True),
        patch("models.ramp.pipeline._restore_checkpoint", return_value=fake_saved_model_dir),
        patch("models.ramp.pipeline.log_metadata"),
    ):
        metrics = evaluate_model.entrypoint(
            trained_model=b"fake",
            dataset_chips=str(toy_chips),
            dataset_labels=str(toy_labels),
            hyperparameters=base_hyperparameters,
            split_info=split_info,
        )

    expected = {"fair:accuracy", "fair:mean_iou", "fair:precision", "fair:recall"}
    assert set(metrics.keys()) == expected
    for key in expected:
        assert isinstance(metrics[key], float)


def test_export_onnx(tmp_path: Path) -> None:
    import onnx
    from onnx import TensorProto, helper

    from models.ramp.pipeline import export_onnx

    # Build a toy ONNX model and capture its bytes.
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1])
    node = helper.make_node("Identity", ["input"], ["output"])
    graph = helper.make_graph([node], "toy", [x], [y])
    toy_model = helper.make_model(graph, producer_name="test")
    toy_bytes = toy_model.SerializeToString()

    fake_saved_model_dir = tmp_path / "saved_model"
    fake_saved_model_dir.mkdir()
    (fake_saved_model_dir / "saved_model.pb").write_bytes(b"\x08\x01")

    with (
        patch("models.ramp.pipeline._restore_checkpoint", return_value=fake_saved_model_dir),
        patch("models.ramp.pipeline._convert_savedmodel_to_onnx_bytes", return_value=toy_bytes),
        patch("models.ramp.pipeline.log_metadata"),
    ):
        exported = export_onnx.entrypoint(trained_model=b"fake")

    assert isinstance(exported, bytes)
    loaded = onnx.load_from_string(exported)
    onnx.checker.check_model(loaded)
