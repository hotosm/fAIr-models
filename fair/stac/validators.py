from __future__ import annotations

import functools
import importlib.resources
import json
import logging
import math
from collections.abc import Mapping
from typing import Any

import pystac

from fair.params import inference_params
from fair.schemas import register_fair_schemas
from fair.stac.constants import BATCH_INFERENCE_ASSET, FAIR_LOCAL_MODEL_SCHEMA, TRAINING_ASSET
from fair.stac.inputs import (
    INPUT_CHIPS_ASSET_PREFIX,
    INPUT_NAME_PATTERN,
    dataset_chips_assets,
    input_chips_asset,
    input_names,
)

logger = logging.getLogger(__name__)


def validate_item(item: pystac.Item) -> list[str]:
    register_fair_schemas()
    errors: list[str] = []
    try:
        item.validate()
    except pystac.errors.STACValidationError as e:
        errors.append(str(e))
    errors.extend(_validate_keyword_vocabulary(item))
    errors.extend(_validate_hyperparameters_spec_coverage(item))
    errors.extend(_validate_prediction_variables_declared(item))
    errors.extend(_validate_input_names(item))
    return errors


def _validate_input_names(item: pystac.Item) -> list[str]:
    """Names of `mlm:input` entries and `chips-<name>` assets become asset keys and paths. A local model
    carries its base model's input names unchecked."""
    if FAIR_LOCAL_MODEL_SCHEMA in item.stac_extensions:
        return []
    asset_names = [
        key.removeprefix(INPUT_CHIPS_ASSET_PREFIX) for key in item.assets if key.startswith(INPUT_CHIPS_ASSET_PREFIX)
    ]
    names = input_names(item)
    errors = [
        f"input name {name!r} must start with a lowercase letter or digit, then use only those, '_' or '-'"
        for name in [*names, *asset_names]
        if not INPUT_NAME_PATTERN.fullmatch(name)
    ]
    if duplicates := sorted({name for name in names if names.count(name) > 1}):
        errors.append(f"input names must be unique, repeated: {', '.join(duplicates)}")
    return errors


def prediction_output(properties: dict) -> dict | None:
    """The `mlm:output` entry describing post-processed prediction features through MLM `variables`."""
    return next((output for output in properties.get("mlm:output", []) if output.get("variables")), None)


def _variable_names(output: dict) -> list[str]:
    return [variable if isinstance(variable, str) else variable["name"] for variable in output["variables"]]


def _validate_prediction_variables_declared(item: pystac.Item) -> list[str]:
    output = prediction_output(item.properties)
    if output is None:
        return []
    declared = item.properties.get("cube:variables", {})
    return [
        f"mlm:output '{output['name']}' variable '{name}' has no entry in cube:variables"
        for name in _variable_names(output)
        if name not in declared
    ]


_MODEL_ASSET_KEYS: tuple[str, ...] = ("checkpoint", "model")


def is_trainable(assets: Mapping[str, Any]) -> bool:
    """A base model without a training runtime serves predictions only and cannot be finetuned."""
    return TRAINING_ASSET in assets


def batch_runtime(assets: Mapping[str, Any]) -> Any | None:
    """The full toolchain image that runs batch predictions; the distroless mlm:inference image cannot."""
    return assets.get(TRAINING_ASSET) or assets.get(BATCH_INFERENCE_ASSET)


def required_model_asset_keys(item: pystac.Item) -> tuple[str, ...]:
    return _MODEL_ASSET_KEYS if is_trainable(item.assets) else ("model",)


def validate_model_asset_urls(
    item: pystac.Item,
    *,
    required_keys: tuple[str, ...] | None = None,
    optional_keys: tuple[str, ...] = (),
    timeout: float = 10.0,
) -> list[str]:

    errors: list[str] = []
    for key in required_keys if required_keys is not None else required_model_asset_keys(item):
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


KEYWORD_WILDCARD = "*"


def _accepts_any_keyword(schema: dict) -> bool:
    """True when `allowed_keywords` is `*`, or a list containing it, which opens the vocabulary."""
    allowed = schema["allowed_keywords"]
    if isinstance(allowed, str):
        return allowed == KEYWORD_WILDCARD
    return KEYWORD_WILDCARD in allowed


def _unknown_keywords(schema: dict, keywords: set[str]) -> set[str]:
    if _accepts_any_keyword(schema):
        return set()
    allowed = (
        set(schema["allowed_keywords"]) | set(schema["allowed_tasks"]) | set(schema.get("allowed_geometry_types", []))
    )
    return keywords - allowed


def _validate_keyword_vocabulary(item: pystac.Item) -> list[str]:
    schema = _load_keywords_schema()
    unknown = _unknown_keywords(schema, set(item.properties.get("keywords", [])))
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
    task_label_mapping = schema["task_label_mapping"]

    errors: list[str] = []

    model_keywords = set(base_model_item.properties.get("keywords", []))
    dataset_keywords = set(dataset_item.properties.get("keywords", []))

    unknown_model = _unknown_keywords(schema, model_keywords)
    if unknown_model:
        errors.append(f"Unknown model keywords: {unknown_model}")

    unknown_dataset = _unknown_keywords(schema, dataset_keywords)
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

    chips_assets = dataset_chips_assets(base_model_item, dataset_item)
    if missing := [input_chips_asset(name) for name in input_names(base_model_item) if name not in chips_assets]:
        errors.append(f"Dataset has no chips for the model's inputs: {', '.join(missing)}")

    return errors


def validate_inference_params(params: dict[str, Any], model_item: pystac.Item) -> None:
    """Raise ValueError unless every key is a declared, spec'd inference parameter with a valid value."""
    props = model_item.properties
    declared = inference_params(props.get("mlm:hyperparameters", {}))
    spec_keys = {entry["key"] for entry in props.get("fair:hyperparameters_spec", [])}
    unknown = sorted(key for key in params if key not in declared or key not in spec_keys)
    if unknown:
        raise ValueError(f"Unknown inference parameter(s) for '{model_item.id}': {', '.join(unknown)}")
    if errors := validate_hyperparameters(params, model_item):
        raise ValueError("; ".join(errors))


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
        elif base_model_item is not None:
            errors.extend(_validate_feature_variables(i, feat["properties"] or {}, base_model_item.properties))

    return errors


def _validate_feature_variables(index: int, feature_properties: dict, model_properties: dict) -> list[str]:
    output = prediction_output(model_properties)
    if output is None:
        return []
    declared = model_properties.get("cube:variables", {})
    errors: list[str] = []
    for name in _variable_names(output):
        if name not in feature_properties:
            errors.append(f"features[{index}].properties missing declared variable '{name}'")
            continue
        # validate_item guarantees every output variable is declared, so a missing entry is a real error.
        variable = declared[name]
        allowed = variable.get("values")
        value = feature_properties[name]
        if allowed is not None and value not in allowed and value != variable.get("nodata"):
            errors.append(f"features[{index}].properties.{name}={value!r} not in declared values {allowed}")
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
        if entry["key"] not in metrics:
            errors.append(f"Missing declared metric: {entry['key']}")
    return errors


_SPEC_REQUIRED_FIELDS: frozenset[str] = frozenset({"type", "default", "description"})
_HP_PREFIXES: tuple[str, ...] = ("training.", "inference.")


def _bare_hp_key(key: str) -> str:
    for prefix in _HP_PREFIXES:
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


def _validate_hyperparameters_spec_coverage(item: pystac.Item) -> list[str]:
    errors: list[str] = []
    hyperparameters: dict = item.properties.get("mlm:hyperparameters", {})
    spec: list[dict] = item.properties.get("fair:hyperparameters_spec", [])
    if not hyperparameters:
        return errors
    if not spec:
        errors.append("mlm:hyperparameters set but fair:hyperparameters_spec is missing or empty")
        return errors

    for entry in spec:
        key = entry.get("key", "<unknown>")
        missing = _SPEC_REQUIRED_FIELDS - entry.keys()
        if missing:
            errors.append(f"fair:hyperparameters_spec entry '{key}' missing required fields: {sorted(missing)}")

    spec_keys = {entry["key"] for entry in spec if "key" in entry}
    for hp_key in hyperparameters:
        bare = _bare_hp_key(hp_key)
        if bare not in spec_keys:
            errors.append(f"mlm:hyperparameters key '{hp_key}' has no matching entry in fair:hyperparameters_spec")

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
            # bool is an int subclass, so True would otherwise pass as a number.
            if not isinstance(value, allowed_types) or (isinstance(value, bool) and expected_type != "bool"):
                errors.append(f"Hyperparameter '{key}' expected type '{expected_type}', got {type(value).__name__}")
                continue

        if isinstance(value, float) and not math.isfinite(value):
            errors.append(f"Hyperparameter '{key}' value {value} is not a finite number")
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
