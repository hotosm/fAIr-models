"""Deterministic toy chips and road polygons for the unet-roads tests."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds

CHIPS_PER_SIDE = 2
CHIP_PIXELS = 64
STEP_DEG = 0.0005
ROAD_HALF_WIDTH_PX = 4
BASE_LON, BASE_LAT = 85.5, 27.6
_EAST, _NORTH = BASE_LON + CHIPS_PER_SIDE * STEP_DEG, BASE_LAT + CHIPS_PER_SIDE * STEP_DEG
_GEOMETRY = {
    "type": "Polygon",
    "coordinates": [
        [[BASE_LON, BASE_LAT], [_EAST, BASE_LAT], [_EAST, _NORTH], [BASE_LON, _NORTH], [BASE_LON, BASE_LAT]]
    ],
}
_BBOX = [BASE_LON, BASE_LAT, _EAST, _NORTH]


def create_toy_data(root: Path) -> dict[str, Any]:
    """Write four dark RGB chips crossed by a bright east-west road, plus the road surface polygons."""
    chips_dir = root / "chips"
    chips_dir.mkdir(parents=True)
    roads = []

    for row in range(CHIPS_PER_SIDE):
        south = BASE_LAT + row * STEP_DEG
        for col in range(CHIPS_PER_SIDE):
            west = BASE_LON + col * STEP_DEG
            transform = from_bounds(west, south, west + STEP_DEG, south + STEP_DEG, CHIP_PIXELS, CHIP_PIXELS)
            pixels = np.full((3, CHIP_PIXELS, CHIP_PIXELS), 40, dtype=np.uint8)
            mid = CHIP_PIXELS // 2
            pixels[:, mid - ROAD_HALF_WIDTH_PX : mid + ROAD_HALF_WIDTH_PX, :] = 200
            with rasterio.open(
                chips_dir / f"OAM-{col}-{row}-19.tif",
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
        centre = south + STEP_DEG / 2
        half = ROAD_HALF_WIDTH_PX * STEP_DEG / CHIP_PIXELS
        ring = [
            [BASE_LON, centre - half],
            [_EAST, centre - half],
            [_EAST, centre + half],
            [BASE_LON, centre + half],
            [BASE_LON, centre - half],
        ]
        roads.append(
            {"type": "Feature", "properties": {"label": 1}, "geometry": {"type": "Polygon", "coordinates": [ring]}}
        )

    labels_dir = root / "labels"
    labels_dir.mkdir()
    (labels_dir / "labels.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": roads}))

    stac_path = root / "dataset-stac-item.json"
    stac_path.write_text(json.dumps(_build_dataset_stac_item(chips_dir, labels_dir), indent=2))
    return {"chips": chips_dir, "labels": labels_dir, "dataset_stac_item": stac_path}


@pytest.fixture(scope="session")
def generate_toy_dataset(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    return create_toy_data(tmp_path_factory.mktemp("toy_unet_roads"))


def _build_dataset_stac_item(chips_dir: Path, labels_dir: Path) -> dict[str, Any]:
    return {
        "type": "Feature",
        "stac_version": "1.1.0",
        "stac_extensions": ["https://stac-extensions.github.io/label/v1.0.1/schema.json"],
        "id": "toy-unet-roads",
        "geometry": _GEOMETRY,
        "bbox": _BBOX,
        "properties": {
            "datetime": "2026-11-01T00:00:00Z",
            "description": "Toy road surface dataset",
            "label:type": "vector",
            "label:tasks": ["segmentation"],
            "label:classes": [{"name": "highway", "classes": ["*"]}],
            "label:description": "Road surface polygons",
            "keywords": ["road", "polygon"],
            "fair:user_id": "test",
            "fair:geometry_type": "polygon",
            "version": "1",
            "deprecated": False,
            "license": "ODbL-1.0",
            "providers": [{"name": "HOTOSM", "roles": ["producer"], "url": "https://www.hotosm.org"}],
        },
        "assets": {
            "chips-image": {"href": str(chips_dir), "type": "image/tiff", "roles": ["data"]},
            "labels": {"href": str(labels_dir), "type": "application/geo+json", "roles": ["labels"]},
        },
        "links": [],
    }
