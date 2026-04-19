from __future__ import annotations

from pathlib import Path
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
