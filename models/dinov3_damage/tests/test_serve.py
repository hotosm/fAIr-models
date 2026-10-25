"""predict on toy chips with a stubbed ONNX session: real mosaics, sliding window and pooling."""

from typing import Any

import numpy as np
import pytest

from models.dinov3_damage.pipeline import predict


class _DestroyedSession:
    """Damage logits that favour `destroyed` everywhere."""

    def run(self, _outputs: list[str], feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        _, _, height, width = feeds["post"].shape
        logits = np.zeros((1, 4, height, width), dtype=np.float32)
        logits[:, 3] = 5.0
        return [logits]


def test_predict_grades_every_footprint_with_the_declared_variables(
    predict_images: dict[str, str], predict_params: dict[str, Any]
) -> None:
    result = predict(_DestroyedSession(), predict_images, predict_params)

    assert result["type"] == "FeatureCollection"
    assert len(result["features"]) == 8
    for feature in result["features"]:
        properties = feature["properties"]
        assert set(properties) == {"damage_class", "damage", "damage_confidence", "osm_id"}
        assert properties["osm_id"] is None
        assert properties["damage"] == "destroyed"
        assert properties["damage_class"] == 3
        assert properties["damage_confidence"] > 0.9
        assert feature["geometry"]["type"] == "Polygon"


def test_predict_drops_buildings_below_the_confidence_threshold(
    predict_images: dict[str, str], predict_params: dict[str, Any]
) -> None:
    result = predict(_DestroyedSession(), predict_images, {**predict_params, "confidence_threshold": 0.999})

    assert result["features"] == []


def test_predict_requires_pre_event_imagery(predict_images: dict[str, str], predict_params: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="'pre'"):
        predict(_DestroyedSession(), {"post": predict_images["post"]}, predict_params)
