"""Step tests for segformer_parking_lot."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pytest


def _hyperparameters(base_hyperparameters: dict[str, Any]) -> dict[str, Any]:
    hp = dict(base_hyperparameters)
    hp.update(
        {
            "epochs": 1,
            "batch_size": 1,
            "chip_size": 32,
            "model_input_size": 32,
            "sample_fraction": 1.0,
            "val_ratio": 0.33,
            "split_seed": 7,
            "block_size": 4,
            "device": "cpu",
        }
    )
    return hp


def test_split_dataset(toy_chips: Path, toy_labels: Path, base_hyperparameters: dict[str, Any]) -> None:
    from models.segformer_parking_lot.pipeline import split_dataset

    info = split_dataset.entrypoint(
        dataset_chips=str(toy_chips),
        dataset_labels=str(toy_labels),
        hyperparameters=_hyperparameters(base_hyperparameters),
    )

    assert info["strategy"] == "spatial"
    assert info["train_count"] > 0
    assert info["val_count"] > 0
    assert info["train_chip_names"]
    assert info["val_chip_names"]
    assert "_prepared_dir" not in info
    assert "labels_geojson" not in info
    assert "labels_crs" not in info


def test_train_model(
    toy_chips: Path,
    toy_labels: Path,
    base_hyperparameters: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import torch

    from models.segformer_parking_lot import pipeline

    split_info = pipeline.split_dataset.entrypoint(
        dataset_chips=str(toy_chips),
        dataset_labels=str(toy_labels),
        hyperparameters=_hyperparameters(base_hyperparameters),
    )
    checkpoint = tmp_path / "base.ckpt"
    checkpoint.write_bytes(b"placeholder")

    class _FakeModel:
        def state_dict(self) -> dict[str, Any]:
            return {"decode_head.classifier.weight": torch.zeros((2, 64, 1, 1), dtype=torch.float32)}

    class _FakeTrainResult:
        def __init__(self) -> None:
            self.model = _FakeModel()
            self.best_epoch = 0
            self.best_val_loss = 0.1
            self.history = {"train_loss": [0.2], "val_loss": [0.1]}

    monkeypatch.setattr(pipeline, "_download_checkpoint", lambda href: Path(href))
    monkeypatch.setattr("pl_hot.train.train_segformer", lambda **_kwargs: _FakeTrainResult())
    model_bytes = pipeline.train_model.entrypoint(
        dataset_chips=str(toy_chips),
        dataset_labels=str(toy_labels),
        base_model_weights=str(checkpoint),
        hyperparameters=_hyperparameters(base_hyperparameters),
        split_info=split_info,
        num_classes=2,
    )
    assert isinstance(model_bytes, bytes)
    assert len(model_bytes) > 0


def test_evaluate_model(
    toy_chips: Path,
    toy_labels: Path,
    base_hyperparameters: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from models.segformer_parking_lot import pipeline

    split_info = pipeline.split_dataset.entrypoint(
        dataset_chips=str(toy_chips),
        dataset_labels=str(toy_labels),
        hyperparameters=_hyperparameters(base_hyperparameters),
    )
    monkeypatch.setattr(pipeline, "_restore_checkpoint", lambda _trained: object())
    monkeypatch.setattr(
        "pl_hot.evaluate.evaluate_segformer",
        lambda *_args, **_kwargs: {"accuracy": 0.9, "mean_iou": 0.7, "iou_background": 0.8, "iou_parking_lot": 0.6},
    )
    metrics = pipeline.evaluate_model.entrypoint(
        trained_model=b"fake",
        dataset_chips=str(toy_chips),
        dataset_labels=str(toy_labels),
        hyperparameters=_hyperparameters(base_hyperparameters),
        split_info=split_info,
    )
    assert "accuracy" in metrics
    assert "mean_iou" in metrics


def test_export_onnx(monkeypatch: pytest.MonkeyPatch) -> None:
    import onnx
    from onnx import TensorProto, helper

    from models.segformer_parking_lot import pipeline

    def _minimal_onnx_bytes() -> bytes:
        x = helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 3, 32, 32])
        y = helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 2, 8, 8])
        node = helper.make_node("Identity", inputs=["image"], outputs=["logits"])
        graph = helper.make_graph([node], "segformer", [x], [y])
        model = helper.make_model(graph)
        buf = io.BytesIO()
        buf.write(model.SerializeToString())
        return buf.getvalue()

    monkeypatch.setattr(pipeline, "_restore_checkpoint", lambda _trained: object())
    monkeypatch.setattr("pl_hot.export.export_onnx_bytes", lambda *_args, **_kwargs: _minimal_onnx_bytes())

    onnx_bytes = pipeline.export_onnx.entrypoint(trained_model=b"fake", hyperparameters={"chip_size": 32})
    loaded = onnx.load_from_string(onnx_bytes)
    assert len(loaded.graph.input) == 1
    assert len(loaded.graph.output) == 1


def test_predict_uses_stac_params_and_postprocess(
    toy_chips: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import numpy as np

    from models.segformer_parking_lot import pipeline

    params = {
        "model_input_size": 32,
        "confidence_threshold": 0.7,
        "min_area_m2": 75,
        "hole_area_m2": 80,
        "simplify_m": 2,
    }
    seen: dict[str, Any] = {}
    preprocess_config = object()
    postprocess_config = object()

    monkeypatch.setattr("pl_hot.params.parse_preprocess_params", lambda actual: preprocess_config)
    monkeypatch.setattr("pl_hot.params.parse_postprocess_params", lambda actual: postprocess_config)

    def _preprocess(image_path: Path, config: Any) -> tuple[Any, Any]:
        seen["preprocess"] = (image_path, config)
        return np.zeros((1, 3, 32, 32), dtype=np.float32), object()

    monkeypatch.setattr("pl_hot.preprocess.preprocess_chip_for_onnx", _preprocess)
    monkeypatch.setattr(
        pipeline,
        "postprocess",
        lambda raw, actual: (
            seen.update({"postprocess": (raw, actual)}) or np.ones((32, 32), dtype=np.uint8),
            np.ones((32, 32), dtype=np.float32),
        ),
    )

    def _polygonize(mask: Any, metadata: Any, config: Any, **kwargs: Any) -> dict[str, Any]:
        seen["polygonize"] = (mask, metadata, config, kwargs)
        return {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {}, "geometry": None}]}

    monkeypatch.setattr("pl_hot.postprocess.mask_to_feature_collection", _polygonize)

    class _Input:
        name = "image"

    class _Session:
        def get_inputs(self) -> list[_Input]:
            return [_Input()]

        def run(self, _outputs: Any, feed: dict[str, Any]) -> list[Any]:
            seen["feed"] = feed
            return [np.zeros((1, 2, 8, 8), dtype=np.float32)]

    result = pipeline.predict(_Session(), str(toy_chips), params)

    assert result["type"] == "FeatureCollection"
    assert seen["preprocess"][1] is preprocess_config
    assert seen["postprocess"][1] is params
    assert seen["polygonize"][2] is postprocess_config
    assert set(seen["feed"]) == {"image"}
