"""Toy OAM chips and labels for the step tests.

Small georeferenced GeoTIFFs at zoom 18 in Freetown, named `OAM-<x>-<y>-18.tif`, with GeoJSON pool polygons. The chips
are 64 px so the tests stay fast on a CPU. The weights of the tests are random, so these tests check the contract and
the wiring, not accuracy.
"""

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin

WEB_MERCATOR_HALF_M = 20037508.342789244
ZOOM = 18
CHIP_PX = 64
POOL_HALF_M = 15.0  # a 30 m square pool, about 12 px at this scale
FREETOWN = (-13.23723, 8.47532)  # lon, lat


def lonlat_to_tile(lon: float, lat: float, zoom: int) -> tuple[int, int]:
    n = 2**zoom
    x = int((lon + 180) / 360 * n)
    y = int((1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n)
    return x, y


def tile_bounds_3857(x: int, y: int, zoom: int) -> tuple[float, float, float, float]:
    size = 2 * WEB_MERCATOR_HALF_M / 2**zoom
    left = -WEB_MERCATOR_HALF_M + x * size
    top = WEB_MERCATOR_HALF_M - y * size
    return left, top - size, left + size, top


def _write_chip(path: Path, x: int, y: int) -> None:
    left, _, right, top = tile_bounds_3857(x, y, ZOOM)
    pixel = (right - left) / CHIP_PX
    rgb = np.random.default_rng(x * 7 + y).integers(60, 200, size=(3, CHIP_PX, CHIP_PX), dtype=np.uint8)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=CHIP_PX,
        height=CHIP_PX,
        count=3,
        dtype="uint8",
        crs="EPSG:3857",
        transform=from_origin(left, top, pixel, pixel),
    ) as dst:
        dst.write(rgb)


def _pool_polygon(x: int, y: int) -> dict[str, Any]:
    """A square pool in the middle of the tile, as a lon/lat polygon."""
    left, bottom, right, top = tile_bounds_3857(x, y, ZOOM)
    cx, cy = (left + right) / 2, (bottom + top) / 2
    to_lonlat = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    corners = ((-1, -1), (1, -1), (1, 1), (-1, 1), (-1, -1))
    ring = [to_lonlat.transform(cx + dx * POOL_HALF_M, cy + dy * POOL_HALF_M) for dx, dy in corners]
    return {"type": "Polygon", "coordinates": [[list(point) for point in ring]]}


def create_toy_data(root: Path) -> dict[str, Any]:
    """Four chips in four different spatial blocks, a pool in three of them; the fourth is a background example.

    Returns the chips folder, the labels folder and the path of a dataset STAC item that points at them.
    """
    chips = root / "chips"
    labels = root / "labels"
    chips.mkdir(parents=True)
    labels.mkdir(parents=True)
    base_x, base_y = lonlat_to_tile(*FREETOWN, ZOOM)
    tiles = [(base_x + 4 * i, base_y) for i in range(4)]
    for x, y in tiles:
        _write_chip(chips / f"OAM-{x}-{y}-{ZOOM}.tif", x, y)

    features = [
        {"type": "Feature", "properties": {"leisure": "swimming_pool"}, "geometry": _pool_polygon(*tile)}
        for tile in tiles[:3]
    ]
    collection = {"type": "FeatureCollection", "features": features}
    (labels / "labels.geojson").write_text(json.dumps(collection), encoding="utf-8")
    stac_path = root / "dataset-stac-item.json"
    stac_path.write_text(json.dumps(_build_dataset_stac_item(chips, labels, tiles), indent=2), encoding="utf-8")
    return {"chips": chips, "labels": labels, "dataset_stac_item": stac_path}


def _build_dataset_stac_item(chips_dir: Path, labels_dir: Path, tiles: list[tuple[int, int]]) -> dict[str, Any]:
    to_lonlat = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    bounds = [tile_bounds_3857(x, y, ZOOM) for x, y in tiles]
    west, south = to_lonlat.transform(min(b[0] for b in bounds), min(b[1] for b in bounds))
    east, north = to_lonlat.transform(max(b[2] for b in bounds), max(b[3] for b in bounds))
    return {
        "type": "Feature",
        "stac_version": "1.1.0",
        "stac_extensions": ["https://stac-extensions.github.io/label/v1.0.1/schema.json"],
        "id": "toy-swimming-pools",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[west, south], [east, south], [east, north], [west, north], [west, south]]],
        },
        "bbox": [west, south, east, north],
        "properties": {
            "datetime": "2026-10-01T00:00:00Z",
            "description": "Toy swimming-pool detection dataset",
            "label:type": "vector",
            "label:tasks": ["object-detection"],
            "label:classes": [{"name": "swimming_pool", "classes": ["yes"]}],
            "label:description": "Swimming pool polygons",
            "keywords": ["swimming-pool", "object-detection", "polygon"],
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


@pytest.fixture(scope="session")
def generate_toy_dataset(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    return create_toy_data(tmp_path_factory.mktemp("toy_dataset"))


@pytest.fixture()
def base_hyperparameters() -> dict[str, Any]:
    """One epoch and two chips per batch: the toy set is four chips."""
    return {"epochs": 1, "batch_size": 2}
