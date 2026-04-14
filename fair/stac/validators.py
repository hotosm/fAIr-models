from __future__ import annotations

import functools
import importlib.resources
import json
import logging

import pystac

from fair.schemas import register_fair_schemas

logger = logging.getLogger(__name__)


def validate_item(item: pystac.Item) -> list[str]:
    register_fair_schemas()
    errors: list[str] = []
    try:
        item.validate()
    except pystac.errors.STACValidationError as e:
        errors.append(str(e))
    errors.extend(_validate_keyword_vocabulary(item))
    return errors


_MODEL_ASSET_KEYS: tuple[str, ...] = ("checkpoint", "model")


def validate_model_asset_urls(
    item: pystac.Item,
    *,
    required_keys: tuple[str, ...] = ("checkpoint",),
    optional_keys: tuple[str, ...] = ("model",),
    timeout: float = 10.0,
) -> list[str]:

    errors: list[str] = []
    for key in required_keys:
        asset = item.assets.get(key)
        if asset is None:
            errors.append(f"Missing required asset '{key}'")
            continue
        errors.extend(_check_asset_url(key, asset.href, timeout))

    for key in optional_keys:
        asset = item.assets.get(key)
        if asset is None:
            continue
        errors.extend(_check_asset_url(key, asset.href, timeout))

    return errors


def _check_asset_url(key: str, href: str, timeout: float) -> list[str]:
    import urllib.request

    if not href.startswith(("http://", "https://")):
        return [f"Asset '{key}' href must be an http(s) URL, got: {href}"]
    try:
        req = urllib.request.Request(href, method="HEAD")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status >= 400:
                return [f"Asset '{key}' URL returned HTTP {resp.status}: {href}"]
    except Exception as exc:
        return [f"Asset '{key}' URL not accessible: {href} ({exc})"]
    return []


# Back-compat alias; prefer validate_model_asset_urls going forward
def validate_model_weight_href(item: pystac.Item, *, timeout: float = 10.0) -> list[str]:
    return validate_model_asset_urls(item, timeout=timeout)


def _validate_keyword_vocabulary(item: pystac.Item) -> list[str]:
    schema = _load_keywords_schema()
    allowed = (
        set(schema["allowed_keywords"]) | set(schema["allowed_tasks"]) | set(schema.get("allowed_geometry_types", []))
    )
    keywords = set(item.properties.get("keywords", []))
    unknown = keywords - allowed
    if unknown:
        return [f"Unknown keywords: {unknown}"]
    return []


# Alias kept for call sites that reference the old name
validate_mlm_schema = validate_item


@functools.cache
def _load_keywords_schema() -> dict:
    ref = importlib.resources.files("fair.schemas").joinpath("keywords.json")
    return json.loads(ref.read_text(encoding="utf-8"))


def validate_compatibility(
    base_model_item: pystac.Item,
    dataset_item: pystac.Item,
) -> list[str]:
    schema = _load_keywords_schema()
    allowed_keywords = set(schema["allowed_keywords"])
    allowed_tasks = set(schema["allowed_tasks"])
    task_label_mapping = schema["task_label_mapping"]

    errors: list[str] = []

    model_keywords = set(base_model_item.properties.get("keywords", []))
    dataset_keywords = set(dataset_item.properties.get("keywords", []))

    unknown_model = model_keywords - allowed_keywords - allowed_tasks - set(schema.get("allowed_geometry_types", []))
    if unknown_model:
        errors.append(f"Unknown model keywords: {unknown_model}")

    unknown_dataset = (
        dataset_keywords - allowed_keywords - allowed_tasks - set(schema.get("allowed_geometry_types", []))
    )
    if unknown_dataset:
        errors.append(f"Unknown dataset keywords: {unknown_dataset}")

    if not model_keywords & dataset_keywords:
        errors.append(f"No keywords in common: model={model_keywords}, dataset={dataset_keywords}")

    model_tasks = set(base_model_item.properties.get("mlm:tasks", []))
    label_tasks = set(dataset_item.properties.get("label:tasks", []))
    mapped_label_tasks = {task_label_mapping.get(t, t) for t in model_tasks}

    if not (mapped_label_tasks & label_tasks or model_tasks & label_tasks):
        errors.append(
            f"No task overlap: model mlm:tasks={model_tasks} "
            f"(mapped to {mapped_label_tasks}), dataset label:tasks={label_tasks}"
        )

    geom_types = set(schema.get("allowed_geometry_types", []))
    model_geom = model_keywords & geom_types
    dataset_geom = dataset_keywords & geom_types
    if model_geom and dataset_geom and not model_geom & dataset_geom:
        errors.append(f"Geometry type mismatch: model={sorted(model_geom)}, dataset={sorted(dataset_geom)}")

    return errors


def validate_pipeline_config(config: dict, *, is_training: bool = True) -> list[str]:
    errors: list[str] = []
    params = config.get("parameters", {})

    if is_training:
        for required in ("base_model_weights", "dataset_chips", "dataset_labels", "hyperparameters"):
            if required not in params:
                errors.append(f"Missing training parameter: {required}")
        hp = params.get("hyperparameters", {})
        if isinstance(hp, dict) and "epochs" not in hp:
            errors.append("hyperparameters missing required key: epochs")
    else:
        for required in ("model_uri", "input_images"):
            if required not in params:
                errors.append(f"Missing inference parameter: {required}")

    return errors


def validate_predictions_geojson(
    geojson: dict,
    base_model_item: pystac.Item | None = None,
) -> list[str]:
    errors: list[str] = []
    if geojson.get("type") != "FeatureCollection":
        errors.append(f"Expected type='FeatureCollection', got '{geojson.get('type')}'")
        return errors

    features = geojson.get("features")
    if not isinstance(features, list):
        errors.append("'features' must be a list")
        return errors

    allowed_geom = _allowed_geometry_geojson_types(base_model_item) if base_model_item else None

    for i, feat in enumerate(features):
        if feat.get("type") != "Feature":
            errors.append(f"features[{i}].type must be 'Feature'")
        if "geometry" not in feat:
            errors.append(f"features[{i}] missing 'geometry'")
        elif allowed_geom:
            geom_type = (feat["geometry"] or {}).get("type", "")
            if geom_type and geom_type not in allowed_geom:
                errors.append(
                    f"features[{i}].geometry.type '{geom_type}' "
                    f"not allowed by keyword geometry type. Expected one of: {sorted(allowed_geom)}"
                )
        if "properties" not in feat:
            errors.append(f"features[{i}] missing 'properties'")

    return errors


_KEYWORD_TO_GEOJSON_TYPES: dict[str, set[str]] = {
    "polygon": {"Polygon", "MultiPolygon"},
    "line": {"LineString", "MultiLineString"},
    "point": {"Point", "MultiPoint"},
}


def _allowed_geometry_geojson_types(item: pystac.Item) -> set[str] | None:
    schema = _load_keywords_schema()
    geom_keywords = set(schema.get("allowed_geometry_types", []))
    declared = geom_keywords & set(item.properties.get("keywords", []))
    if not declared:
        return None
    result: set[str] = set()
    for kw in declared:
        result |= _KEYWORD_TO_GEOJSON_TYPES.get(kw, set())
    return result


def validate_metrics_against_spec(
    metrics: dict,
    base_model_item: pystac.Item,
) -> list[str]:
    errors: list[str] = []
    spec = base_model_item.properties.get("fair:metrics_spec", [])
    for entry in spec:
        name = entry.get("name", "")
        if name and name not in metrics:
            errors.append(f"Missing declared metric: {name}")
    return errors


_SPEC_TYPE_MAP: dict[str, type | tuple[type, ...]] = {
    "int": int,
    "float": (int, float),
    "str": str,
    "bool": bool,
}


def validate_hyperparameters(
    hyperparameters: dict,
    base_model_item: pystac.Item,
) -> list[str]:
    errors: list[str] = []
    spec = base_model_item.properties.get("fair:hyperparameters_spec", [])
    if not spec:
        return errors

    spec_by_key = {entry["key"]: entry for entry in spec if "key" in entry}

    for key, value in hyperparameters.items():
        if key not in spec_by_key:
            continue
        entry = spec_by_key[key]

        expected_type = entry.get("type")
        if expected_type and expected_type in _SPEC_TYPE_MAP:
            allowed_types = _SPEC_TYPE_MAP[expected_type]
            if not isinstance(value, allowed_types):
                errors.append(f"Hyperparameter '{key}' expected type '{expected_type}', got {type(value).__name__}")
                continue

        allowed_values = entry.get("values")
        if allowed_values and value not in allowed_values:
            errors.append(f"Hyperparameter '{key}' value '{value}' not in allowed values: {allowed_values}")

        min_val = entry.get("min")
        if min_val is not None and isinstance(value, (int, float)) and value < min_val:
            errors.append(f"Hyperparameter '{key}' value {value} is below minimum {min_val}")

        max_val = entry.get("max")
        if max_val is not None and isinstance(value, (int, float)) and value > max_val:
            errors.append(f"Hyperparameter '{key}' value {value} is above maximum {max_val}")

    return errors
