"""Tree points as square crowns, so OSM tree points train the detector like crown polygons."""

import json
from pathlib import Path
from typing import Any

import pytest

from models.deepforest_tree.pipeline import METRES_PER_DEGREE, _labeled_boxes, _point_crown, _with_point_crowns


def _point(lon: float, lat: float, properties: dict[str, Any]) -> dict[str, Any]:
    return {"type": "Feature", "properties": properties, "geometry": {"type": "Point", "coordinates": [lon, lat]}}


def _side_degrees(feature: dict[str, Any]) -> float:
    ring = feature["geometry"]["coordinates"][0]
    return ring[2][1] - ring[0][1]


def test_a_tree_point_becomes_a_square_crown_of_its_diameter() -> None:
    crown = _point_crown(_point(0.0, 0.0, {"tags": {"natural": "tree", "diameter_crown": "10"}}))
    assert crown["geometry"]["type"] == "Polygon"
    assert _side_degrees(crown) == pytest.approx(10 / METRES_PER_DEGREE)


@pytest.mark.parametrize(
    "diameter", [None, "3 m", "nan", "-3", "0"], ids=["missing", "free text", "nan", "negative", "zero"]
)
def test_a_tree_point_without_a_usable_diameter_takes_the_default_crown(diameter: str | None) -> None:
    tags = {"natural": "tree"} if diameter is None else {"natural": "tree", "diameter_crown": diameter}
    assert _side_degrees(_point_crown(_point(0.0, 0.0, {"tags": tags}))) == pytest.approx(5 / METRES_PER_DEGREE)


def test_a_crown_away_from_the_equator_is_square_on_the_ground() -> None:
    ring = _point_crown(_point(0.0, 60.0, {"tags": {"diameter_crown": "10"}}))["geometry"]["coordinates"][0]
    assert (ring[1][0] - ring[0][0]) == pytest.approx(2 * (ring[2][1] - ring[1][1]))


def test_polygon_labels_are_used_as_given(toy_labels: Path) -> None:
    labels = toy_labels / "labels.geojson"
    assert _with_point_crowns(labels) == labels


def test_tree_points_give_training_boxes_on_their_chips(toy_chips: Path, toy_labels: Path, tmp_path: Path) -> None:
    polygons = json.loads((toy_labels / "labels.geojson").read_text())["features"]
    points = []
    for feature in polygons:
        ring = feature["geometry"]["coordinates"][0]
        lon = sum(x for x, _ in ring[:-1]) / (len(ring) - 1)
        lat = sum(y for _, y in ring[:-1]) / (len(ring) - 1)
        points.append(_point(lon, lat, {"tags": {"natural": "tree", "diameter_crown": "4"}}))
    labels = tmp_path / "labels.geojson"
    labels.write_text(json.dumps({"type": "FeatureCollection", "features": points}))

    boxes = _labeled_boxes(toy_chips, str(labels))

    assert sum(len(rows) for rows in boxes.values()) == len(points)
