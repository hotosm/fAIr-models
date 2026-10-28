"""Toy OAM chips and labels for the step tests.

Small georeferenced GeoTIFFs at zoom 20 in Freetown, with GeoJSON labels. The chips are 64 px, so a model pixel
at 0.3 m (`model_gsd_m` below, for the tests only) keeps the network inputs small enough for CPU tests. The weights
of the tests are random, so these tests check the contract and the wiring, not accuracy.
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
ZOOM = 20
CHIP_PX = 64
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


def _tile_polygon(x: int, y: int) -> dict[str, Any]:
    """A 8 m square canopy patch in the middle of the tile, as a lon/lat polygon."""
    left, bottom, right, top = tile_bounds_3857(x, y, ZOOM)
    cx, cy = (left + right) / 2, (bottom + top) / 2
    to_lonlat = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    ring = [to_lonlat.transform(cx + dx, cy + dy) for dx, dy in ((-4, -4), (4, -4), (4, 4), (-4, 4), (-4, -4))]
    return {"type": "Polygon", "coordinates": [[list(point) for point in ring]]}


def _tile_point(x: int, y: int) -> dict[str, Any]:
    left, bottom, right, top = tile_bounds_3857(x, y, ZOOM)
    lon, lat = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True).transform(
        (left + right) / 2, (bottom + top) / 2
    )
    return {"type": "Point", "coordinates": [lon, lat]}


def create_toy_data(root: Path) -> dict[str, Any]:
    """Four chips in four different spatial blocks, labels for three of them: a polygon, a point and one of each.

    The fourth chip has no label, so it is a background example. Returns the chips folder, the labels folder and a
    dataset STAC item.
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
        {"type": "Feature", "properties": {"natural": "wood"}, "geometry": _tile_polygon(*tiles[0])},
        {
            "type": "Feature",
            "properties": {"natural": "tree", "diameter_crown": "4"},
            "geometry": _tile_point(*tiles[1]),
        },
        {"type": "Feature", "properties": {"natural": "tree"}, "geometry": _tile_point(*tiles[2])},
    ]
    collection = {"type": "FeatureCollection", "features": features}
    (labels / "labels.geojson").write_text(json.dumps(collection), encoding="utf-8")
    dataset_item = {"type": "Feature", "id": "toy-trees", "properties": {"title": "toy OAM chips with tree labels"}}
    return {"chips": chips, "labels": labels, "dataset_stac_item": dataset_item}


@pytest.fixture(scope="session")
def generate_toy_dataset(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    return create_toy_data(tmp_path_factory.mktemp("toy_dataset"))


@pytest.fixture()
def base_hyperparameters() -> dict[str, Any]:
    """One epoch, and a 0.3 m model pixel: the 64 px test chips stay about 126 px (the default gives about 760 px)."""
    return {"epochs": 1, "model_gsd_m": 0.3}
