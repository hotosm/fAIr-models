from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from affine import Affine


@pytest.mark.parametrize(
    ("mps_available", "cuda_available", "expected"),
    [
        (True, False, "mps"),
        (False, True, "cuda"),
        (False, False, "cpu"),
    ],
)
def test_get_device_prefers_available_accelerator(
    monkeypatch: pytest.MonkeyPatch,
    mps_available: bool,
    cuda_available: bool,
    expected: str,
) -> None:
    import torch

    from models.yolo11n_detection.pipeline import _get_device

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps_available)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)

    assert _get_device() == expected


def test_yolo_helper_functions_cover_geometry_and_decoding(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from models.yolo11n_detection.pipeline import (
        _build_feature_collection,
        _decode_yolo_output,
        _log_yolo_loss_history,
        _nms,
        _pixel_bbox_to_geo_feature,
    )

    feature = _pixel_bbox_to_geo_feature(
        [0.0, 0.0, 2.0, 2.0],
        Affine.identity(),
        None,
        {"label": "building"},
    )
    assert feature["geometry"]["type"] == "Polygon"
    feature_collection = _build_feature_collection([feature])
    assert feature_collection["type"] == "FeatureCollection"

    boxes = np.array([[0.0, 0.0, 2.0, 2.0], [0.2, 0.2, 2.2, 2.2]])
    scores = np.array([0.9, 0.6])
    kept = _nms(boxes, scores, 0.5)
    assert kept == [0]

    output = np.zeros((1, 5, 10), dtype=np.float32)
    output[0, 0, 0] = 128.0
    output[0, 1, 0] = 128.0
    output[0, 2, 0] = 64.0
    output[0, 3, 0] = 64.0
    output[0, 4, 0] = 0.95
    detections = _decode_yolo_output(output, confidence_threshold=0.5, iou_threshold=0.5)
    assert len(detections) == 1
    assert detections[0]["confidence"] > 0.5

    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "results.csv").write_text("train/box_loss,val/box_loss\n1.0,2.0\n")

    logged: list[tuple[str, float, int]] = []
    monkeypatch.setattr("mlflow.log_metric", lambda name, value, step: logged.append((name, value, step)))
    monkeypatch.setattr("fair.zenml.metrics.log_loss_history", lambda train, val: None)
    model = type("Model", (), {"trainer": type("Trainer", (), {"save_dir": results_dir})()})()
    _log_yolo_loss_history(model)
    assert logged == [("train_loss", 1.0, 0), ("val_loss", 2.0, 0)]


def test_yolo_loss_history_early_returns(tmp_path: Path) -> None:
    from models.yolo11n_detection.pipeline import _log_yolo_loss_history

    _log_yolo_loss_history(object())

    model = type("Model", (), {"trainer": type("Trainer", (), {"save_dir": tmp_path})()})()
    _log_yolo_loss_history(model)


def test_restore_preprocess_postprocess_and_predict_errors(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import torch

    from models.yolo11n_detection.pipeline import _restore_checkpoint, postprocess, predict, preprocess

    created: list[str] = []

    class FakeYOLO:
        def __init__(self, path: str) -> None:
            created.append(path)

    import ultralytics

    monkeypatch.setattr(ultralytics, "YOLO", FakeYOLO)

    direct = FakeYOLO("existing.pt")
    assert _restore_checkpoint(direct) is direct
    restored = _restore_checkpoint(b"weights")
    assert isinstance(restored, FakeYOLO)
    assert Path(created[-1]).exists()
    restored = _restore_checkpoint("from-path.pt")
    assert isinstance(restored, FakeYOLO)

    input_path = tmp_path / "chip.tif"
    import rasterio

    with rasterio.open(
        input_path,
        "w",
        driver="GTiff",
        height=16,
        width=16,
        count=3,
        dtype="uint8",
    ) as dataset:
        dataset.write(np.ones((3, 16, 16), dtype=np.uint8))

    tensor = preprocess(input_path, chip_size=32)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape[-2:] == (32, 32)

    box = type(
        "Box",
        (),
        {
            "xyxy": [torch.tensor([1.0, 2.0, 3.0, 4.0])],
            "conf": torch.tensor([0.8]),
            "cls": torch.tensor([1.0]),
        },
    )()
    result = type("Result", (), {"boxes": [box]})()
    detections = postprocess([result])
    assert detections[0]["class"] == 1

    class FakeSession:
        def get_inputs(self) -> list[Any]:
            return [type("Input", (), {"name": "images"})()]

    with pytest.raises(ValueError):
        predict(FakeSession(), str(tmp_path), {})

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        predict(FakeSession(), str(empty_dir), {"confidence_threshold": 0.5})


def test_dataset_cache_predict_success_and_inference_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import rasterio

    from models.yolo11n_detection.pipeline import _dataset_cache_dir, _prepare_yolo_dataset, predict, run_inference

    cache_dir = _dataset_cache_dir("chips-dir", "labels.json")
    for split in ("train", "val"):
        (cache_dir / "images" / split).mkdir(parents=True, exist_ok=True)
    (cache_dir / "images" / "train" / "train-1.png").write_bytes(b"train")
    (cache_dir / "images" / "val" / "val-1.png").write_bytes(b"val")
    (cache_dir / "data.yaml").write_text("path: cached\n", encoding="utf-8")

    resolved_dir, train_count, val_count = _prepare_yolo_dataset("chips-dir", "labels.json", chip_size=640)
    assert resolved_dir == cache_dir
    assert train_count == 1
    assert val_count == 1

    input_path = tmp_path / "chip.tif"
    with rasterio.open(
        input_path,
        "w",
        driver="GTiff",
        height=32,
        width=32,
        count=3,
        dtype="uint8",
        crs="EPSG:4326",
        transform=Affine.identity(),
    ) as dataset:
        dataset.write(np.ones((3, 32, 32), dtype=np.uint8))

    class SuccessfulSession:
        def get_inputs(self) -> list[Any]:
            return [type("Input", (), {"name": "images"})()]

        def run(self, *_args: Any, **_kwargs: Any) -> list[np.ndarray]:
            output = np.zeros((1, 5, 10), dtype=np.float32)
            output[0, 0, 0] = 128.0
            output[0, 1, 0] = 128.0
            output[0, 2, 0] = 64.0
            output[0, 3, 0] = 64.0
            output[0, 4, 0] = 0.95
            return [output]

    result = predict(SuccessfulSession(), str(tmp_path), {"confidence_threshold": 0.5, "iou_threshold": 0.5})
    assert result["type"] == "FeatureCollection"
    assert len(result["features"]) == 1

    expected = {"type": "FeatureCollection", "features": []}
    monkeypatch.setattr("fair.serve.base.load_session", lambda model_uri: object())
    monkeypatch.setattr("models.yolo11n_detection.pipeline.predict", lambda session, input_images, params: expected)

    inference_result = run_inference.entrypoint(
        model_uri="https://example.com/model.onnx",
        input_images=str(tmp_path),
        inference_params={"confidence_threshold": 0.5},
    )
    assert inference_result == expected


def test_train_and_evaluate_rebuild_cached_dataset(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from contextlib import contextmanager

    from models.yolo11n_detection.pipeline import evaluate_model, train_model

    rebuilt_dir = tmp_path / "rebuilt"
    rebuilt_dir.mkdir()
    (rebuilt_dir / "data.yaml").write_text("path: rebuilt\n", encoding="utf-8")

    rebuild_calls: list[tuple[str, str, int, float]] = []

    def fake_prepare(chips_path: str, coco_json_path: str, chip_size: int, val_ratio: float = 0.2, seed: int = 42):
        rebuild_calls.append((chips_path, coco_json_path, chip_size, val_ratio))
        return rebuilt_dir, 3, 1

    @contextmanager
    def noop_ctx(*_args: Any, **_kwargs: Any):
        yield

    class FakeTrainModel:
        def train(self, **kwargs: Any) -> Any:
            return type("Results", (), {"results_dict": {"train/box_loss": 0.25}})()

        def save(self, path: str) -> None:
            Path(path).write_bytes(b"weights")

    class FakeEvalModel:
        def val(self, *, data: str, imgsz: int, verbose: bool) -> Any:
            return type(
                "EvalResults",
                (),
                {
                    "results_dict": {
                        "metrics/mAP50(B)": 0.9,
                        "metrics/mAP50-95(B)": 0.8,
                        "metrics/precision(B)": 0.7,
                        "metrics/recall(B)": 0.6,
                    }
                },
            )()

    import ultralytics

    weights_path = tmp_path / "weights.pt"
    weights_path.write_bytes(b"weights")

    monkeypatch.setattr("models.yolo11n_detection.pipeline._prepare_yolo_dataset", fake_prepare)
    monkeypatch.setattr("models.yolo11n_detection.pipeline._download_checkpoint", lambda url: weights_path)
    monkeypatch.setattr("models.yolo11n_detection.pipeline._get_device", lambda: "cpu")
    monkeypatch.setattr("models.yolo11n_detection.pipeline._log_yolo_loss_history", lambda model: None)
    monkeypatch.setattr("models.yolo11n_detection.pipeline.mlflow_training_context", noop_ctx)
    monkeypatch.setattr("models.yolo11n_detection.pipeline.log_metadata", lambda metadata: None)
    monkeypatch.setattr(ultralytics, "YOLO", lambda path: FakeTrainModel())
    monkeypatch.setattr(ultralytics, "settings", SimpleNamespace(update=lambda values: None))

    trained = train_model.entrypoint(
        dataset_chips=str(tmp_path / "chips"),
        dataset_labels=str(tmp_path / "labels.json"),
        base_model_weights=str(weights_path),
        hyperparameters={"epochs": 1, "chip_size": 640},
        split_info={"_yolo_dir": str(tmp_path / "missing-train"), "val_ratio": 0.25},
        num_classes=1,
    )
    assert trained == b"weights"

    monkeypatch.setattr("models.yolo11n_detection.pipeline._restore_checkpoint", lambda trained_model: FakeEvalModel())
    metrics = evaluate_model.entrypoint(
        trained_model=b"weights",
        dataset_chips=str(tmp_path / "chips"),
        dataset_labels=str(tmp_path / "labels.json"),
        hyperparameters={"chip_size": 640},
        split_info={"_yolo_dir": str(tmp_path / "missing-eval"), "val_ratio": 0.25},
    )

    assert metrics["accuracy"] == 0.9
    assert len(rebuild_calls) == 2


def test_evaluate_model_raises_when_yolo_validation_is_empty(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from models.yolo11n_detection.pipeline import evaluate_model

    (tmp_path / "data.yaml").write_text("path: ready\n", encoding="utf-8")

    class FakeModel:
        def val(self, *, data: str, imgsz: int, verbose: bool) -> object:
            return object()

    monkeypatch.setattr("models.yolo11n_detection.pipeline._restore_checkpoint", lambda trained_model: FakeModel())

    with pytest.raises(RuntimeError, match="YOLO validation produced no results"):
        evaluate_model.entrypoint(
            trained_model=b"weights",
            dataset_chips=str(tmp_path),
            dataset_labels=str(tmp_path / "labels.json"),
            hyperparameters={"chip_size": 640},
            split_info={"_yolo_dir": str(tmp_path), "val_ratio": 0.2},
        )
