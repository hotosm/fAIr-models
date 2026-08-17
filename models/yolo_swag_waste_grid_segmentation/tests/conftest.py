"""Deterministic toy chips and labels for waste-grid pipeline tests."""

from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds

CHIPS_PER_SIDE = 2
CHIP_PIXELS = 64
STEP_DEG = 0.0002
BASE_LON, BASE_LAT = 85.5, 27.6

_WEST = BASE_LON
_EAST = BASE_LON + CHIPS_PER_SIDE * STEP_DEG
_SOUTH = BASE_LAT
_NORTH = BASE_LAT + CHIPS_PER_SIDE * STEP_DEG
_GEOMETRY = {
    "type": "Polygon",
    "coordinates": [[[_WEST, _SOUTH], [_EAST, _SOUTH], [_EAST, _NORTH], [_WEST, _NORTH], [_WEST, _SOUTH]]],
}
_BBOX = [_WEST, _SOUTH, _EAST, _NORTH]
_PRETRAINED_URL = (
    "https://raw.githubusercontent.com/GIScience/solid-waste-detection-for-fAIr/"
    "9f62fd1e4de6905a38620c195a6e62bcef280956/data/checkpoint/checkpoint_v1_extra_large.pt"
)


def create_toy_data(root: Path) -> dict[str, Path]:
    chips_dir = root / "chips"
    chips_dir.mkdir()

    for row in range(CHIPS_PER_SIDE):
        for col in range(CHIPS_PER_SIDE):
            west = BASE_LON + col * STEP_DEG
            south = BASE_LAT + row * STEP_DEG
            east = west + STEP_DEG
            north = south + STEP_DEG
            transform = from_bounds(west, south, east, north, CHIP_PIXELS, CHIP_PIXELS)
            chip_path = chips_dir / f"OAM-{row:02d}-{col:02d}.tif"
            with rasterio.open(
                chip_path,
                "w",
                driver="GTiff",
                width=CHIP_PIXELS,
                height=CHIP_PIXELS,
                count=3,
                dtype="uint8",
                crs=CRS.from_epsg(4326),
                transform=transform,
            ) as dst:
                pixel_value = 224 if col == 0 else 32
                dst.write(np.full((3, CHIP_PIXELS, CHIP_PIXELS), pixel_value, dtype=np.uint8))

    labels_dir = root / "labels"
    labels_dir.mkdir()
    midpoint = BASE_LON + (CHIPS_PER_SIDE * STEP_DEG) / 2
    row_edges = [_SOUTH, _SOUTH + (_NORTH - _SOUTH) / 3, _SOUTH + 2 * (_NORTH - _SOUTH) / 3, _NORTH]
    features = []
    for south, north in pairwise(row_edges):
        for west, east, label in ((_WEST, midpoint, 1), (midpoint, _EAST, 0)):
            features.append(
                {
                    "type": "Feature",
                    "properties": {"label": label},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[west, south], [east, south], [east, north], [west, north], [west, south]]],
                    },
                }
            )
    (labels_dir / "labels.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": features}))

    stac_path = root / "dataset-stac-item.json"
    stac_path.write_text(json.dumps(_build_dataset_stac_item(chips_dir, labels_dir), indent=2))
    return {"chips": chips_dir, "labels": labels_dir, "dataset_stac_item": stac_path}


@pytest.fixture(scope="session")
def generate_toy_dataset(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    return create_toy_data(tmp_path_factory.mktemp("toy_waste_grid"))


@pytest.fixture(scope="session")
def pretrained_weights(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Download the real two-class classifier used as the training base."""
    from upath import UPath

    checkpoint = tmp_path_factory.mktemp("yolo26xcls_weights") / "best.pt"
    checkpoint.write_bytes(UPath(_PRETRAINED_URL).read_bytes())
    return str(checkpoint)


def _build_dataset_stac_item(chips_dir: Path, labels_dir: Path) -> dict[str, Any]:
    return {
        "type": "Feature",
        "stac_version": "1.1.0",
        "stac_extensions": [
            "https://stac-extensions.github.io/label/v1.0.1/schema.json",
        ],
        "id": "toy-waste-grid",
        "geometry": _GEOMETRY,
        "bbox": _BBOX,
        "properties": {
            "datetime": "2024-01-01T00:00:00Z",
            "description": "Toy waste-grid dataset",
            "label:type": "vector",
            "label:tasks": ["segmentation"],
            "label:classes": [{"name": "waste", "classes": ["yes"]}],
            "label:description": "Polygon labels covering the left half of the mosaic",
            "keywords": ["waste"],
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
