import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds

CHIP_COUNT = 8
CHIP_SIZE = 256
TREE_PX = 36
TREE_ORIGINS_PX = ((40, 40), (120, 150), (200, 70))
BASE_LON, BASE_LAT, STEP = 85.5, 27.6, 0.001
_GEOMETRY = {
    "type": "Polygon",
    "coordinates": [
        [
            [BASE_LON, BASE_LAT],
            [BASE_LON + CHIP_COUNT * STEP, BASE_LAT],
            [BASE_LON + CHIP_COUNT * STEP, BASE_LAT + STEP],
            [BASE_LON, BASE_LAT + STEP],
            [BASE_LON, BASE_LAT],
        ]
    ],
}
_BBOX = [BASE_LON, BASE_LAT, BASE_LON + CHIP_COUNT * STEP, BASE_LAT + STEP]


def tree_polygon(lon: float, lat: float, px: int, py: int, size_px: int = TREE_PX) -> dict[str, Any]:
    """Square label of `size_px` pixels whose top-left pixel is (px, py) in the chip with origin (lon, lat)."""
    px_deg = STEP / CHIP_SIZE
    west, east = lon + px * px_deg, lon + (px + size_px) * px_deg
    north, south = lat + STEP - py * px_deg, lat + STEP - (py + size_px) * px_deg
    return {
        "type": "Polygon",
        "coordinates": [[[west, south], [east, south], [east, north], [west, north], [west, south]]],
    }


def create_toy_data(root: Path) -> dict[str, Path]:
    chips_dir = root / "chips"
    chips_dir.mkdir()
    rng = np.random.default_rng(0)
    for i in range(CHIP_COUNT):
        lon = BASE_LON + i * STEP
        transform = from_bounds(lon, BASE_LAT, lon + STEP, BASE_LAT + STEP, CHIP_SIZE, CHIP_SIZE)
        with rasterio.open(
            chips_dir / f"OAM-{i:04d}-0000-0018.tif",
            "w",
            driver="GTiff",
            width=CHIP_SIZE,
            height=CHIP_SIZE,
            count=3,
            dtype="uint8",
            crs=CRS.from_epsg(4326),
            transform=transform,
        ) as dst:
            dst.write(rng.integers(0, 255, (3, CHIP_SIZE, CHIP_SIZE), dtype=np.uint8))

    labels_dir = root / "labels"
    labels_dir.mkdir()
    features = [
        {
            "type": "Feature",
            "properties": {"label": 1},
            "geometry": tree_polygon(BASE_LON + i * STEP, BASE_LAT, px, py),
        }
        for i in range(CHIP_COUNT)
        for px, py in TREE_ORIGINS_PX
    ]
    (labels_dir / "labels.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": features}))

    stac_path = root / "dataset-stac-item.json"
    stac_path.write_text(json.dumps(_build_dataset_stac_item(chips_dir, labels_dir), indent=2))
    return {"chips": chips_dir, "labels": labels_dir, "dataset_stac_item": stac_path}


@pytest.fixture(scope="session")
def generate_toy_dataset(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    return create_toy_data(tmp_path_factory.mktemp("toy_deepforest_tree"))


def _build_dataset_stac_item(chips_dir: Path, labels_dir: Path) -> dict[str, Any]:
    return {
        "type": "Feature",
        "stac_version": "1.1.0",
        "stac_extensions": ["https://stac-extensions.github.io/label/v1.0.1/schema.json"],
        "id": "toy-deepforest-tree",
        "geometry": _GEOMETRY,
        "bbox": _BBOX,
        "properties": {
            "datetime": "2026-10-01T00:00:00Z",
            "description": "Toy tree-crown detection dataset",
            "label:type": "vector",
            "label:tasks": ["object-detection"],
            "label:classes": [{"name": "tree", "classes": ["yes"]}],
            "label:description": "Tree crown polygons",
            "keywords": ["tree", "object-detection", "polygon"],
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
