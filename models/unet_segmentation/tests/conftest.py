from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

CHIP_COUNT = 6
CHIP_SIZE = 256


@pytest.fixture(scope="session")
def generate_toy_dataset(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("toy_segmentation")
    chips_dir = root / "chips"
    chips_dir.mkdir()

    for i in range(CHIP_COUNT):
        img = np.random.randint(0, 255, (CHIP_SIZE, CHIP_SIZE, 3), dtype=np.uint8)
        Image.fromarray(img).save(chips_dir / f"chip_{i:03d}.png")

    labels_dir = root / "labels"
    labels_dir.mkdir()
    features = [
        {
            "type": "Feature",
            "properties": {"building": "yes"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]],
            },
        }
        for _ in range(CHIP_COUNT)
    ]
    geojson = {"type": "FeatureCollection", "features": features}
    (labels_dir / "buildings.geojson").write_text(json.dumps(geojson))

    stac_item = _build_dataset_stac_item(chips_dir, labels_dir)
    stac_path = root / "dataset-stac-item.json"
    stac_path.write_text(json.dumps(stac_item, indent=2))

    return {"chips": chips_dir, "labels": labels_dir, "dataset_stac_item": stac_path}


def _build_dataset_stac_item(chips_dir: Path, labels_dir: Path) -> dict[str, Any]:
    return {
        "type": "Feature",
        "stac_version": "1.1.0",
        "stac_extensions": [
            "https://stac-extensions.github.io/label/v1.0.1/schema.json",
        ],
        "id": "toy-segmentation",
        "geometry": None,
        "bbox": None,
        "properties": {
            "datetime": "2024-01-01T00:00:00Z",
            "label:type": "vector",
            "label:tasks": ["segmentation"],
            "label:classes": [{"name": "building", "classes": ["yes"]}],
        },
        "assets": {
            "chips": {"href": str(chips_dir), "type": "image/png", "roles": ["data"]},
            "labels": {"href": str(labels_dir), "type": "application/geo+json", "roles": ["labels"]},
        },
        "links": [],
    }
