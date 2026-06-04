"""Serve smoke test: real merge + sliding window + watershed + vectorise, stubbed ONNX session."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import shapely.geometry as sgeom
from dinov3_hot.serve import predict_session


class _StubSession:
    def get_inputs(self) -> list[Any]:
        return [SimpleNamespace(name="image")]

    def run(self, _outputs: list[str] | None, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        batch = feeds["image"]
        n, _, h, w = batch.shape
        logits = np.full((n, 3, h, w), -10.0, dtype=np.float32)
        logits[:, 0, 64:192, 64:192] = 5.0
        return [logits]


def test_predict_returns_feature_collection(toy_chips: Path) -> None:
    result = predict_session(_StubSession(), toy_chips, {"confidence_threshold": 0.5})

    assert result["type"] == "FeatureCollection"
    assert len(result["features"]) > 0
    for feature in result["features"]:
        assert feature["type"] == "Feature"
        assert "score" in feature["properties"]
        assert 0.0 <= feature["properties"]["score"] <= 1.0
        geom = sgeom.shape(feature["geometry"])
        assert geom.geom_type in ("Polygon", "MultiPolygon")
        minx, miny, maxx, maxy = geom.bounds
        assert minx >= -180.0 and maxx <= 180.0
        assert miny >= -90.0 and maxy <= 90.0


def test_predict_requires_confidence_threshold(toy_chips: Path) -> None:
    with pytest.raises(ValueError, match="confidence_threshold"):
        predict_session(_StubSession(), toy_chips, {})
