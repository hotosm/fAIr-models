"""Serving-path tests for UNet segmentation predict()."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin


def _write_dummy_geotiff(path: Path, size: int = 256) -> None:
    data = (np.random.rand(3, size, size) * 255).astype(np.uint8)
    transform = from_origin(0, size, 1, 1)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=size,
        width=size,
        count=3,
        dtype="uint8",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data)


@pytest.fixture()
def fake_session() -> Any:
    session = MagicMock()
    session.get_inputs.return_value = [MagicMock(name="input")]
    session.get_inputs.return_value[0].name = "input"
    session.run.return_value = [np.random.rand(1, 2, 256, 256).astype(np.float32)]
    return session


def test_predict_returns_feature_collection(fake_session: Any, tmp_path: Path) -> None:
    from models.unet_segmentation.pipeline import predict

    _write_dummy_geotiff(tmp_path / "chip1.tif")
    result = predict(
        fake_session,
        str(tmp_path),
        {"confidence_threshold": 0.5, "min_class_value": 1},
    )
    assert result["type"] == "FeatureCollection"
    assert isinstance(result["features"], list)


def test_predict_raises_when_no_inputs(fake_session: Any, tmp_path: Path) -> None:
    from models.unet_segmentation.pipeline import predict

    with pytest.raises(FileNotFoundError):
        predict(fake_session, str(tmp_path), {"confidence_threshold": 0.5})


def test_predict_raises_when_confidence_threshold_missing(fake_session: Any, tmp_path: Path) -> None:
    from models.unet_segmentation.pipeline import predict

    _write_dummy_geotiff(tmp_path / "chip1.tif")
    with pytest.raises(ValueError, match="confidence_threshold"):
        predict(fake_session, str(tmp_path), {})
