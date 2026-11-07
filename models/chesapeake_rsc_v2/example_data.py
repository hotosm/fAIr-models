"""Small synthetic dataset for the fAIr registration and fine-tuning example."""

import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin


def create_toy_data(root: Path) -> dict[str, Path]:
    """Generate two separate spatial blocks, each with a two-chip road."""
    chips = root / "chips"
    labels = root / "labels"
    chips.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    pixel = 0.000005
    west, north = 85.52, 27.635
    features = []
    for block in (0, 4):
        for col in (block, block + 1):
            x = west + col * 64 * pixel
            rgb = np.full((3, 64, 64), 30, dtype=np.uint8)
            rgb[:, 29:35, :] = 225
            with rasterio.open(
                chips / f"OAM-{193344 + col}-110119-18.tif",
                "w",
                driver="GTiff",
                width=64,
                height=64,
                count=3,
                dtype="uint8",
                crs="EPSG:4326",
                transform=from_origin(x, north, pixel, pixel),
            ) as dst:
                dst.write(rgb)
        y = north - 32 * pixel
        features.append(
            {
                "type": "Feature",
                "properties": {"highway": "unclassified"},
                "geometry": {
                    "type": "LineString",
                    "coordinates": [
                        [west + block * 64 * pixel, y],
                        [west + (block + 2) * 64 * pixel, y],
                    ],
                },
            }
        )
    (labels / "labels.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": features}))
    east, south = west + 6 * 64 * pixel, north - 64 * pixel
    item = {
        "type": "Feature",
        "stac_version": "1.1.0",
        "id": "toy-chesapeake-rsc-v2",
        "stac_extensions": ["https://stac-extensions.github.io/label/v1.0.1/schema.json"],
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[west, south], [east, south], [east, north], [west, north], [west, south]]],
        },
        "bbox": [west, south, east, north],
        "properties": {
            "datetime": "2026-11-06T00:00:00Z",
            "description": "Synthetic roads for integration testing",
            "label:type": "vector",
            "label:properties": ["highway"],
            "label:tasks": ["segmentation"],
            "label:classes": [{"name": "road", "classes": ["yes"]}],
            "label:description": "Synthetic road centerlines",
            "keywords": ["road"],
            "fair:user_id": "test",
            "version": "1",
            "deprecated": False,
            "license": "CC0-1.0",
            "providers": [{"name": "Isaac Corley", "roles": ["producer"]}],
        },
        "assets": {
            "chips-image": {"href": str(chips.resolve()), "type": "image/tiff", "roles": ["data"]},
            "labels": {"href": str(labels.resolve()), "type": "application/geo+json", "roles": ["labels"]},
        },
        "links": [],
    }
    stac = root / "dataset-stac-item.json"
    stac.write_text(json.dumps(item, indent=2))
    return {"chips": chips, "labels": labels, "dataset_stac_item": stac}
