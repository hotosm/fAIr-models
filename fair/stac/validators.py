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

    if not mapped_label_tasks & label_tasks:
        errors.append(
            f"No task overlap: model mlm:tasks={model_tasks} "
            f"(mapped to {mapped_label_tasks}), dataset label:tasks={label_tasks}"
        )

    return errors
