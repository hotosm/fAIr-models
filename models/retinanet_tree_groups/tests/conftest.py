import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds

EARTH_RADIUS_M = 6378137.0
ORIGIN_SHIFT = math.pi * EARTH_RADIUS_M
ZOOM = 20
CHIP_PX = 256
# Real z20 tiles in Freetown. They span two 4x4 tile blocks, so the spatial split has a val block, and no two
# touch, so each chip is a mosaic of its own and is predicted in one window.
TILE_X0, TILE_Y0 = 485728, 499508
TILE_OFFSETS = ((0, 0), (2, 0), (0, 2), (2, 2), (4, 0), (6, 2))
TREE_CENTRES_PX = ((50, 60), (180, 50), (64, 190))
CROWN_M = "5"  # OSM tags are text
GROUP_BOX_PX = (128, 128, 220, 210)


TILE_M = 2 * ORIGIN_SHIFT / 2**ZOOM


def _tile_bounds_3857(x: int, y: int) -> tuple[float, float, float, float]:
    west, north = -ORIGIN_SHIFT + x * TILE_M, ORIGIN_SHIFT - y * TILE_M
    return west, north - TILE_M, west + TILE_M, north


def _lon_lat(x: int, y: int, px: float, py: float) -> list[float]:
    """Longitude and latitude of pixel (px, py) of tile (x, y)."""
    west, _, _, north = _tile_bounds_3857(x, y)
    mx, my = west + px / CHIP_PX * TILE_M, north - py / CHIP_PX * TILE_M
    return [math.degrees(mx / EARTH_RADIUS_M), math.degrees(math.atan(math.sinh(my / EARTH_RADIUS_M)))]


def _features(x: int, y: int, first_id: int) -> list[dict[str, Any]]:
    """Three tree points with a crown diameter and one tree_group polygon, in OSM's tag form."""
    trees = [
        {
            "type": "Feature",
            "properties": {
                "osm_id": first_id + k,
                "osm_type": "nodes",
                "tags": {"natural": "tree", "diameter_crown": CROWN_M},
            },
            "geometry": {"type": "Point", "coordinates": _lon_lat(x, y, px, py)},
        }
        for k, (px, py) in enumerate(TREE_CENTRES_PX)
    ]
    x1, y1, x2, y2 = GROUP_BOX_PX
    ring = [_lon_lat(x, y, px, py) for px, py in ((x1, y2), (x2, y2), (x2, y1), (x1, y1), (x1, y2))]
    group = {
        "type": "Feature",
        "properties": {"osm_id": first_id + len(trees), "osm_type": "ways_poly", "tags": {"natural": "tree_group"}},
        "geometry": {"type": "Polygon", "coordinates": [ring]},
    }
    return [*trees, group]


def create_toy_data(root: Path) -> dict[str, Path]:
    chips_dir = root / "chips"
    chips_dir.mkdir()
    rng = np.random.default_rng(0)
    features = []
    for k, (dx, dy) in enumerate(TILE_OFFSETS):
        x, y = TILE_X0 + dx, TILE_Y0 + dy
        with rasterio.open(
            chips_dir / f"OAM-{x}-{y}-{ZOOM}.tif",
            "w",
            driver="GTiff",
            width=CHIP_PX,
            height=CHIP_PX,
            count=3,
            dtype="uint8",
            crs=CRS.from_epsg(3857),
            transform=from_bounds(*_tile_bounds_3857(x, y), CHIP_PX, CHIP_PX),
        ) as dst:
            dst.write(rng.integers(0, 255, (3, CHIP_PX, CHIP_PX), dtype=np.uint8))
        features += _features(x, y, first_id=10 * k + 1)

    labels_dir = root / "labels"
    labels_dir.mkdir()
    (labels_dir / "labels.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": features}))

    stac_path = root / "dataset-stac-item.json"
    stac_path.write_text(json.dumps(_build_dataset_stac_item(chips_dir, labels_dir), indent=2))
    return {"chips": chips_dir, "labels": labels_dir, "dataset_stac_item": stac_path}


@pytest.fixture(scope="session")
def generate_toy_dataset(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    return create_toy_data(tmp_path_factory.mktemp("toy_retinanet_tree_groups"))


def _build_dataset_stac_item(chips_dir: Path, labels_dir: Path) -> dict[str, Any]:
    corners = [
        _lon_lat(TILE_X0 + dx, TILE_Y0 + dy, px, py)
        for dx, dy in TILE_OFFSETS
        for px in (0, CHIP_PX)
        for py in (0, CHIP_PX)
    ]
    west, east = min(c[0] for c in corners), max(c[0] for c in corners)
    south, north = min(c[1] for c in corners), max(c[1] for c in corners)
    return {
        "type": "Feature",
        "stac_version": "1.1.0",
        "stac_extensions": ["https://stac-extensions.github.io/label/v1.0.1/schema.json"],
        "id": "toy-retinanet-tree-groups",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[west, south], [east, south], [east, north], [west, north], [west, south]]],
        },
        "bbox": [west, south, east, north],
        "properties": {
            "datetime": "2026-11-13T00:00:00Z",
            "title": "toy-retinanet-tree-groups",
            "description": "Toy tree and tree-group detection dataset",
            "label:type": "vector",
            "label:tasks": ["object-detection"],
            "label:classes": [{"name": "natural", "classes": ["tree", "tree_group"]}],
            "label:description": "OSM natural=tree points with diameter_crown and natural=tree_group polygons",
            "keywords": ["tree", "object-detection", "polygon"],
            "fair:user_id": "test",
            "version": "1",
            "deprecated": False,
            "license": "CC-BY-4.0",
            "providers": [{"name": "HOTOSM", "roles": ["producer"], "url": "https://www.hotosm.org"}],
        },
        "assets": {
            "chips-image": {"href": str(chips_dir), "type": "image/tiff", "roles": ["data"]},
            "labels": {"href": str(labels_dir), "type": "application/geo+json", "roles": ["labels"]},
        },
        "links": [],
    }
