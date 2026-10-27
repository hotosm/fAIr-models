"""`_chip_boxes`: training boxes for polygon labels and for OSM tree points, only on the chips that have a label."""

import json
from pathlib import Path

from models.deepforest_tree.pipeline import _chip_boxes


def test_polygon_labels_give_boxes_on_labelled_chips_only(toy_chips: Path, toy_labels: Path) -> None:
    boxes = _chip_boxes(toy_chips, str(toy_labels / "labels.geojson"))
    assert boxes and all(rows for rows in boxes.values())


def test_tree_points_give_training_boxes_on_their_chips(toy_chips: Path, toy_labels: Path, tmp_path: Path) -> None:
    polygons = json.loads((toy_labels / "labels.geojson").read_text())["features"]
    points = []
    for feature in polygons:
        ring = feature["geometry"]["coordinates"][0]
        lon = sum(x for x, _ in ring[:-1]) / (len(ring) - 1)
        lat = sum(y for _, y in ring[:-1]) / (len(ring) - 1)
        points.append(
            {
                "type": "Feature",
                "properties": {"tags": {"natural": "tree", "diameter_crown": "4"}},
                "geometry": {"type": "Point", "coordinates": [lon, lat]},
            }
        )
    labels = tmp_path / "labels.geojson"
    labels.write_text(json.dumps({"type": "FeatureCollection", "features": points}))

    boxes = _chip_boxes(toy_chips, str(labels))

    assert sum(len(rows) for rows in boxes.values()) == len(points)
