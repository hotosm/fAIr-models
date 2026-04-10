from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds

CHIP_COUNT = 6
CHIP_SIZE = 256
BASE_LON, BASE_LAT, STEP = 85.5, 27.6, 0.001
_WEST, _SOUTH = BASE_LON, BASE_LAT
_EAST, _NORTH = BASE_LON + 3 * STEP, BASE_LAT + 2 * STEP
_GEOMETRY = {
    "type": "Polygon",
    "coordinates": [[[_WEST, _SOUTH], [_EAST, _SOUTH], [_EAST, _NORTH], [_WEST, _NORTH], [_WEST, _SOUTH]]],
}
_BBOX = [_WEST, _SOUTH, _EAST, _NORTH]


def create_toy_data(root: Path) -> dict[str, Path]:
    chips_dir = root / "chips"
    chips_dir.mkdir()

    images = []
    annotations = []
    for i in range(CHIP_COUNT):
        lon = BASE_LON + (i % 3) * STEP
        lat = BASE_LAT + (i // 3) * STEP
        name = f"chip_{i:03d}.tif"
        transform = from_bounds(lon, lat, lon + STEP, lat + STEP, CHIP_SIZE, CHIP_SIZE)
        with rasterio.open(
            chips_dir / name,
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
        images.append({"id": i + 1, "file_name": name, "width": CHIP_SIZE, "height": CHIP_SIZE})
        box = CHIP_SIZE // 3
        annotations.append(
            {
                "id": i + 1,
                "image_id": i + 1,
                "category_id": 0,
                "bbox": [2, 2, box, box],
                "area": box * box,
                "iscrowd": 0,
            }
        )

    labels_path = root / "labels.json"
    labels_path.write_text(
        json.dumps({"images": images, "annotations": annotations, "categories": [{"id": 0, "name": "building"}]})
    )

    stac_path = root / "dataset-stac-item.json"
    stac_path.write_text(json.dumps(_build_dataset_stac_item(chips_dir, labels_path), indent=2))
    return {"chips": chips_dir, "labels": labels_path, "dataset_stac_item": stac_path}


@pytest.fixture(scope="session")
def generate_toy_dataset(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    return create_toy_data(tmp_path_factory.mktemp("toy_detection"))


@pytest.fixture()
def base_hyperparameters(chip_size: int) -> dict[str, Any]:
    return {
        "epochs": 1,
        "batch_size": 2,
        "learning_rate": 0.001,
        "chip_size": min(chip_size, 64),
        "val_ratio": 0.3,
        "split_seed": 42,
    }


def _build_dataset_stac_item(chips_dir: Path, labels_path: Path) -> dict[str, Any]:
    return {
        "type": "Feature",
        "stac_version": "1.1.0",
        "stac_extensions": [
            "https://stac-extensions.github.io/label/v1.0.1/schema.json",
        ],
        "id": "toy-detection",
        "geometry": _GEOMETRY,
        "bbox": _BBOX,
        "properties": {
            "datetime": "2024-01-01T00:00:00Z",
            "description": "Toy detection dataset",
            "label:type": "vector",
            "label:tasks": ["object-detection"],
            "label:classes": [{"name": "building", "classes": ["building"]}],
            "label:description": "Detection labels",
            "keywords": ["building"],
            "fair:user_id": "test",
            "version": "1",
            "deprecated": False,
            "license": "CC-BY-4.0",
        },
        "assets": {
            "chips": {"href": str(chips_dir), "type": "image/tiff", "roles": ["data"]},
            "labels": {"href": str(labels_path), "type": "application/json", "roles": ["labels"]},
        },
        "links": [],
    }
