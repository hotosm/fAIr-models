"""Toy dataset fixtures for segformer_parking_lot step tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import rasterio
from pyproj import Transformer
from rasterio.crs import CRS
from rasterio.transform import from_origin

CHIP_SIZE = 32
_X0, _Y0 = 500000.0, 1000.0
_STEP = 1.0
_GEOMETRY = {
    "type": "Polygon",
    "coordinates": [[[-180, -90], [180, -90], [180, 90], [-180, 90], [-180, -90]]],
}
_BBOX = [-180, -90, 180, 90]


def create_toy_data(root: Path) -> dict[str, Path]:
    chips_dir = root / "chips"
    chips_dir.mkdir(parents=True)
    transform = from_origin(_X0, _Y0, _STEP, _STEP)

    # x=0,8,16 ensures multiple (x//4) blocks for spatial split.
    for i, tile_x in enumerate((0, 8, 16)):
        arr = np.zeros((3, CHIP_SIZE, CHIP_SIZE), dtype=np.uint8)
        arr[0, 8:20, 8:20] = 220 - i
        arr[1, 8:20, 8:20] = 150
        arr[2, 8:20, 8:20] = 100
        chip = chips_dir / f"OAM-{tile_x:04d}-0000-18.tif"
        with rasterio.open(
            chip,
            "w",
            driver="GTiff",
            width=CHIP_SIZE,
            height=CHIP_SIZE,
            count=3,
            dtype="uint8",
            crs=CRS.from_epsg(3857),
            transform=transform,
        ) as dst:
            dst.write(arr)
        (chips_dir / f"{chip.name}.aux.xml").write_text("<PAMDataset/>", encoding="utf-8")

    labels_dir = root / "labels"
    labels_dir.mkdir(parents=True)
    to_wgs84 = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    x0, y0 = _X0 + 8.0, _Y0 - 8.0
    x1, y1 = _X0 + 20.0, _Y0 - 20.0
    ring = [to_wgs84.transform(x, y) for x, y in ((x0, y0), (x0, y1), (x1, y1), (x1, y0), (x0, y0))]
    labels = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [ring]},
                "properties": {},
            }
        ],
    }
    (labels_dir / "labels.geojson").write_text(json.dumps(labels), encoding="utf-8")

    stac_path = root / "dataset-stac-item.json"
    stac_path.write_text(json.dumps(_build_dataset_stac_item(chips_dir, labels_dir), indent=2), encoding="utf-8")
    return {"chips": chips_dir, "labels": labels_dir, "dataset_stac_item": stac_path}


@pytest.fixture(scope="session")
def generate_toy_dataset(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    return create_toy_data(tmp_path_factory.mktemp("toy_segformer_parking_lot"))


def _build_dataset_stac_item(chips_dir: Path, labels_dir: Path) -> dict[str, Any]:
    return {
        "type": "Feature",
        "stac_version": "1.1.0",
        "stac_extensions": ["https://stac-extensions.github.io/label/v1.0.1/schema.json"],
        "id": "toy-segformer-parking-lot",
        "geometry": _GEOMETRY,
        "bbox": _BBOX,
        "properties": {
            "datetime": "2026-09-21T00:00:00Z",
            "description": "Toy segformer parking lot dataset",
            "label:type": "vector",
            "label:tasks": ["segmentation"],
            "label:classes": [{"name": "parking_lot", "classes": ["yes"]}],
            "label:description": "Parking-lot segmentation labels",
            "keywords": ["parking_lot"],
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
