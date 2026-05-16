from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds

CHIP_COUNT = 6
CHIP_SIZE = 256
BASE_LON, BASE_LAT, STEP = 85.5, 27.6, 0.001
_WEST, _SOUTH = BASE_LON, BASE_LAT
_EAST, _NORTH = BASE_LON + 3 * STEP, BASE_LAT + 2 * STEP
_GEOMETRY = {
    "type": "Polygon",
    "coordinates": [[[_WEST, _SOUTH], [_EAST, _SOUTH], [_EAST, _NORTH], [_WEST, _NORTH], [_WEST, _SOUTH]]],
}
_BBOX = [_WEST, _SOUTH, _EAST, _NORTH]


def create_toy_data(root: Path) -> dict[str, Path]:
    chips_dir = root / "chips"
    chips_dir.mkdir()
    labels_dir = root / "labels"
    labels_dir.mkdir()

    features: list[dict[str, Any]] = []
    for i in range(CHIP_COUNT):
        lon = BASE_LON + (i % 3) * STEP
        lat = BASE_LAT + (i // 3) * STEP
        transform = from_bounds(lon, lat, lon + STEP, lat + STEP, CHIP_SIZE, CHIP_SIZE)
        with rasterio.open(
            chips_dir / f"chip_{i:03d}.tif",
            "w",
            driver="GTiff",
            width=CHIP_SIZE,
            height=CHIP_SIZE,
            count=3,
            dtype="uint8",
            crs=CRS.from_epsg(4326),
            transform=transform,
        ) as dst:
            dst.write(np.random.randint(0, 255, (3, CHIP_SIZE, CHIP_SIZE), dtype=np.uint8))

        # Even chips get a centered building feature; odd chips get none.
        # The pipeline derives binary labels by chip-vs-feature intersection.
        if i % 2 == 0:
            cx, cy = lon + STEP / 2, lat + STEP / 2
            half = STEP / 6
            features.append(
                {
                    "type": "Feature",
                    "properties": {"building": "yes"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [
                                [cx - half, cy - half],
                                [cx + half, cy - half],
                                [cx + half, cy + half],
                                [cx - half, cy + half],
                                [cx - half, cy - half],
                            ]
                        ],
                    },
                }
            )

    (labels_dir / "labels.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": features}))

    stac_path = root / "dataset-stac-item.json"
    stac_path.write_text(json.dumps(_build_dataset_stac_item(chips_dir, labels_dir), indent=2))
    return {"chips": chips_dir, "labels": labels_dir, "dataset_stac_item": stac_path}


@pytest.fixture(scope="session")
def generate_toy_dataset(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    return create_toy_data(tmp_path_factory.mktemp("toy_classification"))


def _build_dataset_stac_item(chips_dir: Path, labels_dir: Path) -> dict[str, Any]:
    return {
        "type": "Feature",
        "stac_version": "1.1.0",
        "stac_extensions": [
            "https://stac-extensions.github.io/label/v1.0.1/schema.json",
        ],
        "id": "toy-classification",
        "geometry": _GEOMETRY,
        "bbox": _BBOX,
        "properties": {
            "datetime": "2024-01-01T00:00:00Z",
            "description": "Toy classification dataset",
            "label:type": "vector",
            "label:tasks": ["classification"],
            "label:classes": [{"name": "building", "classes": ["yes"]}],
            "label:description": "Building polygons; per-chip binary labels derived at runtime",
            "keywords": ["building"],
            "fair:user_id": "test",
            "version": "1",
            "deprecated": False,
            "license": "CC-BY-4.0",
            "providers": [{"name": "HOTOSM", "roles": ["producer"], "url": "https://www.hotosm.org"}],
        },
        "assets": {
            "chips": {"href": str(chips_dir), "type": "image/tiff", "roles": ["data"]},
            "labels": {"href": str(labels_dir), "type": "application/geo+json", "roles": ["labels"]},
        },
        "links": [],
    }
