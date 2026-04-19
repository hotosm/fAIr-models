from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from rasterio.transform import from_origin


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

    from models.unet_segmentation.pipeline import _get_device

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps_available)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)

    assert _get_device() == expected


def test_segmentation_helpers_cover_preprocess_and_postprocess() -> None:
    import torch

    from models.unet_segmentation.pipeline import _build_feature_collection, _softmax, postprocess, preprocess

    batch = {
        "image": torch.full((1, 3, 2, 2), 255.0),
        "mask": torch.ones((1, 1, 2, 2), dtype=torch.int64),
    }
    images, masks = preprocess(batch)
    assert images.max().item() == 1.0
    assert masks.shape == (1, 2, 2)

    logits = torch.tensor([[[[0.0, 3.0], [5.0, 1.0]], [[2.0, 1.0], [1.0, 4.0]]]])
    prediction = postprocess(logits)
    assert prediction.dtype == np.uint8
    assert prediction.shape == (1, 2, 2)

    probs = _softmax(np.array([[1.0, 2.0], [3.0, 4.0]]), axis=0)
    assert np.allclose(probs.sum(axis=0), np.array([1.0, 1.0]))

    feature_collection = _build_feature_collection([{"type": "Feature"}])
    assert feature_collection["type"] == "FeatureCollection"
    assert len(feature_collection["features"]) == 1


def test_vectorize_mask_download_checkpoint_and_predict_errors(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from models.unet_segmentation.pipeline import _download_checkpoint, _vectorize_segmentation_mask, predict

    mask = np.array([[0, 1], [1, 1]], dtype=np.uint8)
    transform = from_origin(0, 2, 1, 1)
    features = _vectorize_segmentation_mask(mask, transform, None)
    assert features
    assert features[0]["properties"]["class"] == 1

    class DummyUPath:
        def __init__(self, path: str) -> None:
            self.path = path

        @property
        def name(self) -> str:
            return "weights.bin"

        def read_bytes(self) -> bytes:
            return b"weights"

    import upath

    monkeypatch.setattr(upath, "UPath", DummyUPath)
    checkpoint = _download_checkpoint("s3://bucket/weights.bin")
    assert checkpoint.exists()
    assert checkpoint.read_bytes() == b"weights"

    class FakeSession:
        def get_inputs(self) -> list[Any]:
            return [type("Input", (), {"name": "images"})()]

    with pytest.raises(ValueError):
        predict(FakeSession(), str(tmp_path), {})

    with pytest.raises(FileNotFoundError):
        predict(FakeSession(), str(tmp_path), {"confidence_threshold": 0.5})
