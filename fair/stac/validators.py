from __future__ import annotations

import functools
import importlib.resources
import json

import pystac

# TODO : extend the validation with complete set of requirements based on the prod stac , currently only handful checks
#  are in place


def validate_mlm_schema(item: pystac.Item) -> list[str]:
    errors: list[str] = []
    try:
        item.validate()
    except pystac.errors.STACValidationError as e:
        errors.append(str(e))
    return errors


@functools.cache
def _load_keywords_schema() -> dict:
    ref = importlib.resources.files("fair.schemas").joinpath("keywords.json")
    return json.loads(ref.read_text(encoding="utf-8"))


@functools.cache
def _load_base_model_requirements() -> dict:
    ref = importlib.resources.files("fair.schemas").joinpath("base_model_requirements.json")
    return json.loads(ref.read_text(encoding="utf-8"))


def _check_processing_fn(fn: object, path: str, required_fields: list[str], errors: list[str]) -> None:
    if not isinstance(fn, dict):
        errors.append(f"{path} must be an object")
        return
    for field in required_fields:
        if field not in fn:
            errors.append(f"{path} missing field: {field}")


def validate_base_model_item(item: pystac.Item) -> list[str]:
    """Validate a base-model STAC item against fAIr requirements from base_model_requirements.json."""
    reqs = _load_base_model_requirements()
    kw_schema = _load_keywords_schema()
    errors: list[str] = []

    declared = set(item.stac_extensions)
    for ext in reqs["required_extensions"]:
        if ext not in declared:
            errors.append(f"Missing extension: {ext}")

    props = item.properties
    for prop in reqs["required_properties"]:
        if prop not in props or props[prop] is None:
            errors.append(f"Missing property: {prop}")

    for prop in reqs["non_empty_list_properties"]:
        val = props.get(prop)
        if isinstance(val, list) and not val:
            errors.append(f"Property must be non-empty list: {prop}")

    allowed_kw = (
        set(kw_schema["allowed_keywords"])
        | set(kw_schema["allowed_tasks"])
        | set(kw_schema.get("allowed_geometry_types", []))
    )
    unknown_kw = set(props.get("keywords", [])) - allowed_kw
    if unknown_kw:
        errors.append(f"Unknown keywords: {unknown_kw}")

    if reqs.get("require_geometry_keyword"):
        geom_types = set(kw_schema.get("allowed_geometry_types", []))
        if not geom_types & set(props.get("keywords", [])):
            errors.append(f"keywords must include at least one geometry type: {sorted(geom_types)}")

    for prop, allowed in reqs.get("allowed_values", {}).items():
        val = props.get(prop)
        if val is None:
            continue
        items = val if isinstance(val, list) else [val]
        invalid = set(items) - set(allowed)
        if invalid:
            errors.append(f"Invalid {prop} values: {invalid}. Allowed: {allowed}")

    proc_fields = reqs["processing_function_fields"]
    for i, inp in enumerate(props.get("mlm:input") or []):
        for field in reqs["input_required_fields"]:
            if field not in inp:
                errors.append(f"mlm:input[{i}] missing: {field}")
            elif field == "pre_processing_function":
                _check_processing_fn(inp[field], f"mlm:input[{i}].{field}", proc_fields, errors)

    for i, out in enumerate(props.get("mlm:output") or []):
        for field in reqs["output_required_fields"]:
            if field not in out:
                errors.append(f"mlm:output[{i}] missing: {field}")
            elif field == "post_processing_function":
                _check_processing_fn(out[field], f"mlm:output[{i}].{field}", proc_fields, errors)

    for asset_key, required_fields in reqs["required_assets"].items():
        if asset_key not in item.assets:
            errors.append(f"Missing asset: {asset_key}")
            continue
        for field in required_fields:
            if field not in item.assets[asset_key].extra_fields:
                errors.append(f"Asset '{asset_key}' missing field: {field}")

    return errors


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
    """Validate a generated ZenML YAML config dict before running a pipeline."""
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
    """Validate that a predictions dict is a valid GeoJSON FeatureCollection."""
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
    """Check that returned metrics include all names declared in fair:metrics_spec."""
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
    """Validate hyperparameters against fair:hyperparameters_spec from a base model item."""
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
