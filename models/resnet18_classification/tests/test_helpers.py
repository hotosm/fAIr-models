from __future__ import annotations

from pathlib import Path

import pytest
from rasterio.crs import CRS


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

    from models.resnet18_classification.pipeline import _get_device

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps_available)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)

    assert _get_device() == expected


def test_classification_helpers_cover_geo_tensor_and_download_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch
    import upath

    from models.resnet18_classification.pipeline import (
        _bounds_to_geo_feature,
        _build_feature_collection,
        _download_checkpoint,
        _sigmoid,
        postprocess,
        preprocess,
    )

    batch = {
        "image": torch.ones((1, 3, 4, 4), dtype=torch.float32),
        "label": torch.tensor([1.0]),
    }
    images, labels = preprocess(batch)
    assert images.shape == (1, 3, 4, 4)
    assert labels.tolist() == [1.0]

    logits = torch.tensor([[2.0], [-2.0]], dtype=torch.float32)
    predictions = postprocess(logits)
    assert predictions.tolist() == [[1], [0]]
    assert _sigmoid(0.0) == pytest.approx(0.5)

    feature = _bounds_to_geo_feature(
        0.0,
        0.0,
        111319.49,
        111319.49,
        CRS.from_epsg(3857),
        {"source": "chip.tif"},
    )
    assert feature["geometry"]["type"] == "Polygon"
    assert feature["geometry"]["coordinates"][0][0] == feature["geometry"]["coordinates"][0][-1]

    feature_collection = _build_feature_collection([feature])
    assert feature_collection["type"] == "FeatureCollection"
    assert len(feature_collection["features"]) == 1

    class DummyUPath:
        def __init__(self, path: str) -> None:
            self.path = path

        @property
        def name(self) -> str:
            return Path(self.path).name

        def read_bytes(self) -> bytes:
            return b"weights"

    monkeypatch.setattr(upath, "UPath", DummyUPath)
    checkpoint = _download_checkpoint("s3://bucket/weights.bin")
    assert checkpoint.exists()
    assert checkpoint.read_bytes() == b"weights"


def test_run_inference_loads_session_and_predicts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from models.resnet18_classification.pipeline import run_inference

    fake_session = object()
    expected = {"type": "FeatureCollection", "features": []}

    monkeypatch.setattr("fair.serve.base.load_session", lambda model_uri: fake_session)
    monkeypatch.setattr(
        "models.resnet18_classification.pipeline.predict",
        lambda session, input_images, params: expected,
    )

    result = run_inference.entrypoint(
        model_uri="https://example.com/model.onnx",
        input_images=str(tmp_path),
        inference_params={"confidence_threshold": 0.5},
    )

    assert result == expected
