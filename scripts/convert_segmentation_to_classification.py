#!/usr/bin/env python3
"""Convert segmentation GeoJSON labels to classification CSV.

For each chip, determine if any building polygon intersects it, producing
binary (no_building / building) labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import rasterio
from shapely.geometry import box, shape

log = logging.getLogger(__name__)


def _load_building_polygons(geojson_path: Path) -> list:
    with open(geojson_path, encoding="utf-8") as f:
        data = json.load(f)
    return [shape(feat["geometry"]) for feat in data["features"]]


def _chip_has_buildings(chip_path: Path, polygons: list) -> bool:
    with rasterio.open(chip_path) as src:
        chip_box = box(*src.bounds)
    return any(poly.intersects(chip_box) for poly in polygons)


def convert(chips_dir: Path, geojson_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    polygons = _load_building_polygons(geojson_path)
    chips = sorted(chips_dir.glob("*.tif"))
    if not chips:
        msg = f"No .tif chips found in {chips_dir}"
        raise FileNotFoundError(msg)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "class_name"])
        building_count = 0
        for chip in chips:
            has_building = _chip_has_buildings(chip, polygons)
            class_name = "building" if has_building else "no_building"
            if has_building:
                building_count += 1
            writer.writerow([chip.name, class_name])

    total = len(chips)
    log.info(
        "Wrote %d labels (%d building, %d no_building) to %s",
        total,
        building_count,
        total - building_count,
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
        default=Path("data/sample/train/classification_labels.csv"),
    )
    args = parser.parse_args()
    convert(args.chips_dir, args.geojson, args.output)


if __name__ == "__main__":
    main()
