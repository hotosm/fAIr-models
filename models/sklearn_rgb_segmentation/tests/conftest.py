"""Deterministic toy chips and labels for the sklearn-rgb-segmentation tests."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds

CHIPS_PER_SIDE = 2
CHIP_PIXELS = 32
STEP_DEG = 0.001
BASE_LON, BASE_LAT = 85.5, 27.6
_EAST, _NORTH = BASE_LON + CHIPS_PER_SIDE * STEP_DEG, BASE_LAT + CHIPS_PER_SIDE * STEP_DEG
_GEOMETRY = {
    "type": "Polygon",
    "coordinates": [
        [[BASE_LON, BASE_LAT], [_EAST, BASE_LAT], [_EAST, _NORTH], [BASE_LON, _NORTH], [BASE_LON, BASE_LAT]]
    ],
}
_BBOX = [BASE_LON, BASE_LAT, _EAST, _NORTH]


def create_toy_data(root: Path) -> dict[str, Path]:
    """Write four RGB chips whose left half is bright, plus labels over the bright half."""
    chips_dir = root / "chips"
    chips_dir.mkdir(parents=True)
    building_polygons = []

    for row in range(CHIPS_PER_SIDE):
        for col in range(CHIPS_PER_SIDE):
            west = BASE_LON + col * STEP_DEG
            south = BASE_LAT + row * STEP_DEG
            east, north = west + STEP_DEG, south + STEP_DEG
            transform = from_bounds(west, south, east, north, CHIP_PIXELS, CHIP_PIXELS)
            pixels = np.full((3, CHIP_PIXELS, CHIP_PIXELS), 32, dtype=np.uint8)
            pixels[:, :, : CHIP_PIXELS // 2] = 224
            with rasterio.open(
                chips_dir / f"OAM-{row:02d}-{col:02d}.tif",
                "w",
                driver="GTiff",
                width=CHIP_PIXELS,
                height=CHIP_PIXELS,
                count=3,
                dtype="uint8",
                crs=CRS.from_epsg(4326),
                transform=transform,
            ) as dst:
                dst.write(pixels)
            mid = west + STEP_DEG / 2
            building_polygons.append(
                {
                    "type": "Feature",
                    "properties": {"label": 1},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[west, south], [mid, south], [mid, north], [west, north], [west, south]]],
                    },
                }
            )

    labels_dir = root / "labels"
    labels_dir.mkdir()
    (labels_dir / "labels.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": building_polygons}))

    stac_path = root / "dataset-stac-item.json"
    stac_path.write_text(json.dumps(_build_dataset_stac_item(chips_dir, labels_dir), indent=2))
    return {"chips": chips_dir, "labels": labels_dir, "dataset_stac_item": stac_path}


@pytest.fixture
def generate_toy_dataset(tmp_path: Path) -> dict[str, Path]:
    return create_toy_data(tmp_path)


def _build_dataset_stac_item(chips_dir: Path, labels_dir: Path) -> dict[str, Any]:
    return {
        "type": "Feature",
        "stac_version": "1.1.0",
        "stac_extensions": ["https://stac-extensions.github.io/label/v1.0.1/schema.json"],
        "id": "toy-sklearn-rgb-segmentation",
        "geometry": _GEOMETRY,
        "bbox": _BBOX,
        "properties": {
            "datetime": "2026-05-18T00:00:00Z",
            "description": "Toy sklearn RGB segmentation dataset",
            "label:type": "vector",
            "label:tasks": ["segmentation"],
            "label:classes": [{"name": "building", "classes": ["yes"]}],
            "label:description": "Segmentation labels",
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
