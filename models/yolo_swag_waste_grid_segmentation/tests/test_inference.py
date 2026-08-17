"""Focused contracts for the published classifier and grid prediction output."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest


class _StubSession:
    """Small ONNX Runtime stand-in with a fixed two-class probability vector."""

    def __init__(self, probabilities: list[float]) -> None:
        self._probabilities = np.asarray([probabilities], dtype=np.float32)
        self.batches: list[np.ndarray] = []

    def get_inputs(self) -> list[Any]:
        return [SimpleNamespace(name="input")]

    def run(self, _output_names: Any, feeds: dict[str, Any]) -> list[Any]:
        self.batches.append(feeds["input"])
        return [self._probabilities]


def _probabilities_for_waste_confidence(waste_confidence: float) -> list[float]:
    from models.yolo_swag_waste_grid_segmentation.pipeline import CLASS_NAMES

    probabilities = [1.0 - waste_confidence, waste_confidence]
    waste_index = CLASS_NAMES.index("waste")
    return probabilities if waste_index == 1 else list(reversed(probabilities))


def test_predict_returns_unique_wgs84_polygons(toy_chips: Path) -> None:
    """Prediction turns each grid cell into one valid GeoJSON polygon."""
    from models.yolo_swag_waste_grid_segmentation.pipeline import predict

    session = _StubSession(_probabilities_for_waste_confidence(0.9))
    predictions = predict(session, str(toy_chips), {"confidence_threshold": 0.5})

    assert predictions["type"] == "FeatureCollection"
    assert predictions["features"]
    cell_ids = [feature["properties"]["cell_id"] for feature in predictions["features"]]
    assert len(cell_ids) == len(set(cell_ids))
    feature = predictions["features"][0]
    ring = feature["geometry"]["coordinates"][0]
    assert feature["geometry"]["type"] == "Polygon"
    assert len(ring) == 5
    assert ring[0] == ring[-1]


@pytest.mark.parametrize(
    ("waste_confidence", "expected_label"),
    [(0.1, "background"), (0.5, "waste"), (0.9, "waste")],
)
def test_predict_applies_the_waste_confidence_threshold(
    toy_chips: Path,
    waste_confidence: float,
    expected_label: str,
) -> None:
    """The threshold is inclusive and controls the output label."""
    from models.yolo_swag_waste_grid_segmentation.pipeline import predict

    session = _StubSession(_probabilities_for_waste_confidence(waste_confidence))
    predictions = predict(session, str(toy_chips), {"confidence_threshold": 0.5})

    assert {feature["properties"]["label"] for feature in predictions["features"]} == {expected_label}


def test_predict_requires_a_confidence_threshold(toy_chips: Path) -> None:
    """Inference rejects an ambiguous request without a confidence threshold."""
    from models.yolo_swag_waste_grid_segmentation.pipeline import predict

    with pytest.raises(ValueError, match="confidence_threshold"):
        predict(_StubSession(_probabilities_for_waste_confidence(0.9)), str(toy_chips), {})
