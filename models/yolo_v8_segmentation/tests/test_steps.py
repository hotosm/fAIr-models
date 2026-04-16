"""Step tests for YOLOv8 segmentation pipeline.

Each test calls step.entrypoint(...) directly and patches instrumentation
and heavy operations where needed.
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
    from models.yolo_v8_segmentation.pipeline import split_dataset

    hyperparameters = dict(base_hyperparameters)
    hyperparameters.update({"epochs": 1, "batch_size": 1, "p_val": 0.25, "split_seed": 42})

    with patch("models.yolo_v8_segmentation.pipeline.log_metadata"):
        result = split_dataset.entrypoint(
            dataset_chips=str(toy_chips),
            dataset_labels=str(toy_labels),
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


def test_train_model(toy_chips: Path, toy_labels: Path, base_hyperparameters: dict[str, Any], tmp_path: Path) -> None:
    from models.yolo_v8_segmentation.pipeline import train_model

    yolo_dir = tmp_path / "yolo"
    yolo_dir.mkdir(parents=True)
    dataset_yaml = yolo_dir / "yolo_dataset.yaml"
    dataset_yaml.write_text("path: .\ntrain: images/train\nval: images/val\nnc: 1\nnames: ['building']\n")

    fake_checkpoint = tmp_path / "best.pt"
    fake_checkpoint.write_bytes(b"fake-pt-weights")

    split_info = {
        "_work_dir": str(tmp_path),
        "_yolo_dir": str(yolo_dir),
        "_dataset_yaml": str(dataset_yaml),
        "strategy": "random",
        "val_ratio": 0.25,
        "seed": 42,
        "train_count": 3,
        "val_count": 1,
        "description": "test split",
    }
    hyperparameters = dict(base_hyperparameters)
    hyperparameters.update({"epochs": 1, "batch_size": 1, "pc": 2.0})

    with (
        patch("models.yolo_v8_segmentation.pipeline.mlflow_training_context", _noop_mlflow_ctx, create=True),
        patch("models.yolo_v8_segmentation.pipeline.resolve_model_href", return_value=str(fake_checkpoint)),
        patch("models.yolo_v8_segmentation.pipeline.train_yolo_model", return_value=(str(fake_checkpoint), 87.1)),
        patch("models.yolo_v8_segmentation.pipeline.log_metadata"),
    ):
        model_bytes = train_model.entrypoint(
            dataset_chips=str(toy_chips),
            dataset_labels=str(toy_labels),
            base_model_weights="https://example.com/base.pt",
            hyperparameters=hyperparameters,
            split_info=split_info,
            num_classes=2,
        )

    assert isinstance(model_bytes, bytes)
    assert model_bytes == b"fake-pt-weights"


def test_evaluate_model(
    toy_chips: Path, toy_labels: Path, base_hyperparameters: dict[str, Any], tmp_path: Path
) -> None:
    from models.yolo_v8_segmentation.pipeline import evaluate_model

    class _MockResult:
        from typing import ClassVar

        results_dict: ClassVar[dict[str, float]] = {
            "metrics/mAP50(M)": 0.77,
            "metrics/mAP50-95(M)": 0.51,
            "metrics/precision(M)": 0.81,
            "metrics/recall(M)": 0.74,
        }

    class _MockModel:
        def val(self, **_kwargs: Any) -> _MockResult:
            return _MockResult()

    yolo_dir = tmp_path / "yolo"
    yolo_dir.mkdir(parents=True)
    dataset_yaml = yolo_dir / "yolo_dataset.yaml"
    dataset_yaml.write_text("path: .\ntrain: images/train\nval: images/val\nnc: 1\nnames: ['building']\n")

    split_info = {
        "_work_dir": str(tmp_path),
        "_yolo_dir": str(yolo_dir),
        "_dataset_yaml": str(dataset_yaml),
        "strategy": "random",
        "val_ratio": 0.25,
        "seed": 42,
        "train_count": 3,
        "val_count": 1,
        "description": "test split",
    }

    with (
        patch("models.yolo_v8_segmentation.pipeline.mlflow_training_context", _noop_mlflow_ctx, create=True),
        patch("models.yolo_v8_segmentation.pipeline._restore_checkpoint", return_value=_MockModel()),
        patch("models.yolo_v8_segmentation.pipeline.log_metadata"),
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


def test_export_onnx() -> None:
    import onnx
    from onnx import TensorProto, helper

    from models.yolo_v8_segmentation.pipeline import export_onnx

    class _MockModel:
        def __init__(self, output_path: Path):
            self.output_path = output_path

        def export(self, format: str = "onnx") -> str:
            assert format == "onnx"
            return str(self.output_path)

    with patch("tempfile.mkdtemp", return_value=str(Path.cwd() / ".tmp_onnx_test")):
        out_dir = Path(".tmp_onnx_test")
        out_dir.mkdir(exist_ok=True)
        onnx_path = out_dir / "model.onnx"

        x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1])
        y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1])
        node = helper.make_node("Identity", ["input"], ["output"])
        graph = helper.make_graph([node], "toy", [x], [y])
        model = helper.make_model(graph, producer_name="test")
        onnx.save(model, str(onnx_path))

        with patch("models.yolo_v8_segmentation.pipeline._restore_checkpoint", return_value=_MockModel(onnx_path)):
            exported = export_onnx.entrypoint(trained_model=b"fake")

    assert Path(exported).exists()
