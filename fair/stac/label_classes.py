"""Match a dataset's labels, declared with the STAC Label extension, to a model's `classification:classes`."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from fair.stac.constants import LABEL_CLASS_PROPERTY

OSM_TAGS_FIELD = "fair:osm_tags"


class LabelClassError(ValueError):
    """A dataset's classes do not map onto a model's classes."""


@dataclass(frozen=True)
class ModelClass:
    name: str
    value: int
    osm_tags: Mapping[str, list[str]] = field(default_factory=dict)


def model_classes(properties: Mapping[str, Any]) -> list[ModelClass]:
    """Classes of the model's first `mlm:output` that declares `classification:classes`, without nodata ones."""
    for output in properties.get("mlm:output", []):
        if classes := output.get("classification:classes"):
            return [
                ModelClass(cls["name"], int(cls["value"]), cls.get(OSM_TAGS_FIELD, {}))
                for cls in classes
                if not cls.get("nodata")
            ]
    raise LabelClassError("the model declares no classification:classes in mlm:output")


def dataset_classes(properties: Mapping[str, Any]) -> list[tuple[str, str]]:
    """(property, value) pairs the dataset's `label:properties` and `label:classes` declare."""
    names = properties.get("label:properties") or []
    pairs = [
        (cls["name"], str(value))
        for cls in properties.get("label:classes", [])
        if cls.get("name") in names
        for value in cls.get("classes", [])
    ]
    if not pairs:
        raise LabelClassError("the dataset declares no label:classes for its label:properties")
    return pairs


def class_key(name: str, value: str) -> str:
    return f"{name}={value}"


def _automatic_class(name: str, value: str, classes: list[ModelClass]) -> ModelClass | None:
    for cls in classes:
        values = cls.osm_tags.get(name)
        if values is not None and ("*" in values or value in values):
            return cls
    return next((cls for cls in classes if cls.name == value), None)


def resolve_classes(
    dataset_properties: Mapping[str, Any],
    model_properties: Mapping[str, Any],
    class_map: Mapping[str, str] | None = None,
) -> dict[tuple[str, str], ModelClass]:
    """Model class for each dataset class: `class_map` first, then the class's `fair:osm_tags` or its name."""
    classes = model_classes(model_properties)
    by_name = {cls.name: cls for cls in classes}
    pairs = dataset_classes(dataset_properties)
    known = {class_key(name, value) for name, value in pairs}
    class_map = class_map or {}
    problems = [
        f"class_map key {key!r} is not a dataset class; dataset classes: {sorted(known)}"
        for key in class_map
        if key not in known
    ] + [
        f"class_map value {target!r} is not a model class; model classes: {sorted(by_name)}"
        for target in class_map.values()
        if target not in by_name
    ]
    if problems:
        raise LabelClassError("\n".join(problems))

    resolved: dict[tuple[str, str], ModelClass] = {}
    unmatched: list[str] = []
    for name, value in pairs:
        key = class_key(name, value)
        target = by_name[class_map[key]] if key in class_map else _automatic_class(name, value, classes)
        if target is None:
            unmatched.append(key)
        else:
            resolved[(name, value)] = target
    if unmatched:
        raise LabelClassError(
            f"dataset classes {unmatched} match no model class {sorted(by_name)}; map them with class_map"
        )
    return resolved


def adapt_features(
    features: list[dict[str, Any]],
    dataset_properties: Mapping[str, Any],
    model_properties: Mapping[str, Any],
    class_map: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """The features with `label` set to their model class value."""
    resolved = resolve_classes(dataset_properties, model_properties, class_map)
    names = dataset_properties["label:properties"]
    adapted: list[dict[str, Any]] = []
    for position, feature in enumerate(features):
        properties = feature.get("properties") or {}
        target = next(
            (
                resolved[(name, str(properties[name]))]
                for name in names
                if (name, str(properties.get(name))) in resolved
            ),
            None,
        )
        if target is None:
            raise LabelClassError(f"feature {position} carries none of the dataset classes in {names}")
        adapted.append({**feature, "properties": {**properties, LABEL_CLASS_PROPERTY: target.value}})
    return adapted
