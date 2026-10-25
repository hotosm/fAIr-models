import json
from pathlib import Path
from typing import Any

import mercantile
import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds

CHIP_SIZE = 256
ZOOM = 18
BLOCK_COUNT = 8
TILE_X, TILE_Y = 193232, 109610
LABEL_CYCLE = (1, 4)


def _tiles() -> list[tuple[int, int]]:
    """One tile per 2x2 block, so each tile becomes its own training window."""
    return [(TILE_X + 2 * block, TILE_Y) for block in range(BLOCK_COUNT)]


def _write_rgb(path: Path, bounds: tuple[float, float, float, float], size: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=size,
        height=size,
        count=3,
        dtype="uint8",
        crs=CRS.from_epsg(4326),
        transform=from_bounds(*bounds, size, size),
    ) as dst:
        dst.write(rng.integers(40, 220, size=(3, size, size), dtype=np.uint8))


def _center_square(x: int, y: int) -> list[list[float]]:
    west, south, east, north = mercantile.bounds(x, y, ZOOM)
    center_x, center_y = (west + east) / 2, (south + north) / 2
    half_width, half_height = (east - west) * 0.15, (north - south) * 0.15
    return [
        [center_x - half_width, center_y - half_height],
        [center_x + half_width, center_y - half_height],
        [center_x + half_width, center_y + half_height],
        [center_x - half_width, center_y + half_height],
        [center_x - half_width, center_y - half_height],
    ]


def create_toy_data(root: Path) -> dict[str, Any]:
    """Post chips, pre chips of the same tiles, and damage labels 1 and 4."""
    chips_dir, pre_chips_dir, labels_dir = root / "chips", root / "chips-pre", root / "labels"
    for directory in (chips_dir, pre_chips_dir, labels_dir):
        directory.mkdir()
    features = []
    for index, (x, y) in enumerate(_tiles()):
        bounds = tuple(mercantile.bounds(x, y, ZOOM))
        name = f"OAM-{x}-{y}-{ZOOM}.tif"
        _write_rgb(chips_dir / name, bounds, CHIP_SIZE, seed=index)
        _write_rgb(pre_chips_dir / name, bounds, CHIP_SIZE, seed=100 + index)
        features.append(
            {
                "type": "Feature",
                "properties": {"label": LABEL_CYCLE[index % len(LABEL_CYCLE)]},
                "geometry": {"type": "Polygon", "coordinates": [_center_square(x, y)]},
            }
        )
    labels_geojson = labels_dir / "labels.geojson"
    labels_geojson.write_text(json.dumps({"type": "FeatureCollection", "features": features}))

    west, south, _, _ = mercantile.bounds(*_tiles()[0], ZOOM)
    _, _, east, north = mercantile.bounds(*_tiles()[-1], ZOOM)
    bbox = [west, south, east, north]

    stac_path = root / "dataset-stac-item.json"
    stac_path.write_text(json.dumps(_build_dataset_stac_item(chips_dir, pre_chips_dir, labels_dir, bbox), indent=2))
    return {
        "chips": chips_dir,
        "pre_chips": pre_chips_dir,
        "labels": labels_dir,
        "dataset_stac_item": stac_path,
        "predict_images": {"post": str(chips_dir), "pre": str(pre_chips_dir)},
        "predict_params": {
            "buildings_geojson_uri": str(labels_geojson),
            "confidence_threshold": 0.0,
        },
    }


@pytest.fixture(scope="session")
def generate_toy_dataset(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    return create_toy_data(tmp_path_factory.mktemp("toy_dinov3_damage"))


@pytest.fixture(scope="session")
def toy_pre_chips(generate_toy_dataset: dict[str, Any]) -> Path:
    return generate_toy_dataset["pre_chips"]


@pytest.fixture(scope="session")
def predict_images(generate_toy_dataset: dict[str, Any]) -> dict[str, str]:
    return generate_toy_dataset["predict_images"]


@pytest.fixture(scope="session")
def predict_params(generate_toy_dataset: dict[str, Any]) -> dict[str, Any]:
    return generate_toy_dataset["predict_params"]


def _build_dataset_stac_item(chips_dir: Path, pre_chips_dir: Path, labels_dir: Path, bbox: list[float]) -> dict:
    west, south, east, north = bbox
    return {
        "type": "Feature",
        "stac_version": "1.1.0",
        "stac_extensions": ["https://stac-extensions.github.io/label/v1.0.1/schema.json"],
        "id": "toy-dinov3-damage",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[west, south], [east, south], [east, north], [west, north], [west, south]]],
        },
        "bbox": bbox,
        "properties": {
            "datetime": "2026-10-13T00:00:00Z",
            "description": "Toy building damage dataset",
            "label:type": "vector",
            "label:tasks": ["segmentation"],
            "label:classes": [
                {"name": "damage", "classes": ["no-damage"]},
                {"name": "damage", "classes": ["minor-damage"]},
                {"name": "damage", "classes": ["major-damage"]},
                {"name": "damage", "classes": ["destroyed"]},
            ],
            "label:description": "Building damage labels",
            "keywords": ["building"],
            "fair:user_id": "test",
            "version": "1",
            "deprecated": False,
            "license": "CC-BY-4.0",
            "providers": [{"name": "HOTOSM", "roles": ["producer"], "url": "https://www.hotosm.org"}],
        },
        "assets": {
            "chips-post": {"href": str(chips_dir), "type": "image/tiff", "roles": ["data"]},
            "chips-pre": {"href": str(pre_chips_dir), "type": "image/tiff", "roles": ["data"]},
            "labels": {"href": str(labels_dir), "type": "application/geo+json", "roles": ["labels"]},
        },
        "links": [],
    }
