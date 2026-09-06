"""Deterministic toy chips and labels for the sklearn-rgb-segmentation tests."""

import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds

CHIPS_PER_SIDE = 2
CHIP_PIXELS = 32
STEP_DEG = 0.001
BASE_LON, BASE_LAT = 85.5, 27.6


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

    labels_path = root / "labels.geojson"
    labels_path.write_text(json.dumps({"type": "FeatureCollection", "features": building_polygons}))
    return {"chips": chips_dir, "labels": labels_path}


@pytest.fixture
def generate_toy_dataset(tmp_path: Path) -> dict[str, Path]:
    return create_toy_data(tmp_path)
