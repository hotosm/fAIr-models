from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mercantile
import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds

CHIP_COUNT = 4
CHIP_SIZE = 128
# Web Mercator zoom and tile grid used by hot_fair_utilities label clipping
# (see clip_labels._bounding_box_from_filename).
_OAM_ZOOM = 18
# Four neighbouring tiles — x,y must be in [0, 2**z) for mercantile (avoids
# FutureWarning on tile 0,0,0 edge cases).
_TILE_XY = [(100, 200), (101, 200), (100, 201), (101, 201)]

_bounds = mercantile.bounds(mercantile.Tile(x=_TILE_XY[0][0], y=_TILE_XY[0][1], z=_OAM_ZOOM))
_WEST, _SOUTH, _EAST, _NORTH = _bounds.west, _bounds.south, _bounds.east, _bounds.north
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
        tx, ty = _TILE_XY[i]
        tile = mercantile.Tile(x=tx, y=ty, z=_OAM_ZOOM)
        b = mercantile.bounds(tile)
        transform = from_bounds(b.west, b.south, b.east, b.north, CHIP_SIZE, CHIP_SIZE)
        # Filename must match OAM-{x}-{y}-{z} so clip_labels can parse tile ids (hot_fair_utilities).
        chip_name = f"OAM-{tx}-{ty}-{_OAM_ZOOM}.tif"
        with rasterio.open(
            chips_dir / chip_name,
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

        # Tiny square polygon inside each chip footprint (degrees, EPSG:4326).
        w, h = b.east - b.west, b.north - b.south
        cx = b.west + w * 0.35
        cy = b.south + h * 0.35
        s = min(w, h) * 0.25
        features.append(
            {
                "type": "Feature",
                "properties": {"building": 1},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[cx, cy], [cx + s, cy], [cx + s, cy + s], [cx, cy + s], [cx, cy]]],
                },
            }
        )

    labels_geojson = labels_dir / "labels.geojson"
    labels_geojson.write_text(json.dumps({"type": "FeatureCollection", "features": features}))

    stac_path = root / "dataset-stac-item.json"
    stac_path.write_text(json.dumps(_build_dataset_stac_item(chips_dir, labels_dir), indent=2))
    return {"chips": chips_dir, "labels": labels_dir, "dataset_stac_item": stac_path}


@pytest.fixture(scope="session")
def generate_toy_dataset(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    return create_toy_data(tmp_path_factory.mktemp("toy_yolo_v8_segmentation"))


def _build_dataset_stac_item(chips_dir: Path, labels_dir: Path) -> dict[str, Any]:
    return {
        "type": "Feature",
        "stac_version": "1.1.0",
        "stac_extensions": [
            "https://stac-extensions.github.io/label/v1.0.1/schema.json",
        ],
        "id": "toy-yolo-v8-segmentation",
        "geometry": _GEOMETRY,
        "bbox": _BBOX,
        "properties": {
            "datetime": "2024-01-01T00:00:00Z",
            "description": "Toy instance segmentation dataset",
            "label:type": "vector",
            "label:tasks": ["segmentation"],
            "label:classes": [{"name": "building", "classes": ["building"]}],
            "label:description": "Toy polygon labels",
            "keywords": ["building"],
            "providers": [
                {
                    "name": "HOTOSM",
                    "roles": ["producer"],
                    "url": "https://www.hotosm.org",
                    "description": "Humanitarian OpenStreetMap Team",
                }
            ],
            "fair:user_id": "test",
            "version": "1",
            "deprecated": False,
            "license": "CC-BY-4.0",
        },
        "assets": {
            "chips": {"href": str(chips_dir), "type": "image/tiff", "roles": ["data"]},
            "labels": {"href": str(labels_dir), "type": "application/geo+json", "roles": ["labels"]},
        },
        "links": [],
    }
