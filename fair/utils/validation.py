from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Literal

from zenml import step

log = logging.getLogger(__name__)


def _check_images_exist(chips_path: str, *, extensions: tuple[str, ...] = (".tif", ".tiff", ".png", ".jpg")) -> int:
    path = Path(chips_path)
    if not path.exists():
        msg = f"Chips directory does not exist: {chips_path}"
        raise FileNotFoundError(msg)
    count = sum(1 for f in path.iterdir() if f.suffix.lower() in extensions)
    if count == 0:
        msg = f"No image files found in {chips_path}"
        raise FileNotFoundError(msg)
    return count


def _validate_segmentation_labels(labels_path: str) -> None:
    path = Path(labels_path)
    if not path.exists():
        msg = f"Labels file does not exist: {labels_path}"
        raise FileNotFoundError(msg)
    with open(path, encoding="utf-8") as f:
        data: Any = json.load(f)
    features = data.get("features", [])
    if not features:
        msg = f"GeoJSON has no features: {labels_path}"
        raise ValueError(msg)
    for i, feat in enumerate(features):
        geom_type = feat.get("geometry", {}).get("type", "")
        if geom_type not in {"Polygon", "MultiPolygon"}:
            msg = f"Feature {i} has unsupported geometry type '{geom_type}' in {labels_path}"
            raise ValueError(msg)


def _validate_classification_labels(labels_path: str, chips_path: str) -> None:
    path = Path(labels_path)
    if not path.exists():
        msg = f"Classification labels file does not exist: {labels_path}"
        raise FileNotFoundError(msg)
    chips_dir = Path(chips_path)
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "filename" not in reader.fieldnames or "class_name" not in reader.fieldnames:
            msg = f"Classification CSV must have 'filename' and 'class_name' columns: {labels_path}"
            raise ValueError(msg)
        classes: set[str] = set()
        for row in reader:
            classes.add(row["class_name"])
            chip_file = chips_dir / row["filename"]
            if not chip_file.exists():
                msg = f"Referenced chip not found: {chip_file}"
                raise FileNotFoundError(msg)
    if len(classes) < 2:
        msg = f"Classification dataset needs at least 2 classes, found: {classes}"
        raise ValueError(msg)


def _validate_detection_labels(labels_path: str) -> None:
    path = Path(labels_path)
    if not path.exists():
        msg = f"Detection labels file does not exist: {labels_path}"
        raise FileNotFoundError(msg)
    with open(path, encoding="utf-8") as f:
        data: Any = json.load(f)
    for key in ("images", "annotations", "categories"):
        if key not in data:
            msg = f"COCO JSON missing '{key}' field: {labels_path}"
            raise ValueError(msg)
    image_ids = {img["id"] for img in data["images"]}
    for ann in data["annotations"]:
        if ann["image_id"] not in image_ids:
            msg = f"Annotation references unknown image_id {ann['image_id']}"
            raise ValueError(msg)
        bbox = ann.get("bbox", [])
        if len(bbox) != 4 or bbox[2] <= 0 or bbox[3] <= 0:
            msg = f"Invalid bbox {bbox} in annotation {ann.get('id')}"
            raise ValueError(msg)


@step
def validate_dataset(
    chips_path: str,
    labels_path: str,
    label_format: Literal["segmentation", "classification", "detection"] = "segmentation",
) -> int:
    chip_count = _check_images_exist(chips_path)
    if label_format == "segmentation":
        _validate_segmentation_labels(labels_path)
    elif label_format == "classification":
        _validate_classification_labels(labels_path, chips_path)
    elif label_format == "detection":
        _validate_detection_labels(labels_path)
    log.info("Validated %d chips with %s labels", chip_count, label_format)
    return chip_count
