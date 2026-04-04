#!/usr/bin/env python3
"""Convert segmentation GeoJSON labels to COCO detection format.

For each chip, clip building polygons to chip bounds and compute
bounding boxes in pixel coordinates (COCO format).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import rasterio
from rasterio.transform import rowcol
from shapely.geometry import box, shape

log = logging.getLogger(__name__)


def _load_building_polygons(geojson_path: Path) -> list:
    with open(geojson_path, encoding="utf-8") as f:
        data = json.load(f)
    return [shape(feat["geometry"]) for feat in data["features"]]


def _geo_to_pixel_bbox(
    polygon,
    transform: rasterio.transform.Affine,
    img_width: int,
    img_height: int,
) -> tuple[float, float, float, float] | None:
    minx, miny, maxx, maxy = polygon.bounds
    row_min, col_min = rowcol(transform, minx, maxy)
    row_max, col_max = rowcol(transform, maxx, miny)

    x = max(0, int(col_min))
    y = max(0, int(row_min))
    x2 = min(img_width, int(col_max))
    y2 = min(img_height, int(row_max))
    w = x2 - x
    h = y2 - y

    if w <= 0 or h <= 0:
        return None
    return (float(x), float(y), float(w), float(h))


def convert(chips_dir: Path, geojson_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    polygons = _load_building_polygons(geojson_path)
    chips = sorted(chips_dir.glob("*.tif"))
    if not chips:
        msg = f"No .tif chips found in {chips_dir}"
        raise FileNotFoundError(msg)

    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    annotation_id = 1

    for image_id, chip in enumerate(chips, start=1):
        with rasterio.open(chip) as src:
            chip_box = box(*src.bounds)
            transform = src.transform
            width = src.width
            height = src.height

        images.append({"id": image_id, "file_name": chip.name, "width": width, "height": height})

        for poly in polygons:
            if not poly.intersects(chip_box):
                continue
            clipped = poly.intersection(chip_box)
            if clipped.is_empty:
                continue

            bbox = _geo_to_pixel_bbox(clipped, transform, width, height)
            if bbox is None:
                continue

            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 0,
                    "bbox": list(bbox),
                    "area": bbox[2] * bbox[3],
                    "iscrowd": 0,
                }
            )
            annotation_id += 1

    coco: dict[str, Any] = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 0, "name": "building"}],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(coco, f, indent=2)

    log.info(
        "Wrote %d images, %d annotations to %s",
        len(images),
        len(annotations),
        output_path,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chips-dir", type=Path, default=Path("data/sample/train/oam"))
    geojson_default = "data/sample/train/osm/osm_features_ac7e343eb1faacd2.geojson"
    parser.add_argument("--geojson", type=Path, default=Path(geojson_default))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/sample/train/detection_labels.json"),
    )
    args = parser.parse_args()
    convert(args.chips_dir, args.geojson, args.output)


if __name__ == "__main__":
    main()
