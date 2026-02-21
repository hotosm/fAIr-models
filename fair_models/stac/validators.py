from __future__ import annotations

import importlib.resources
import json

import pystac


def validate_mlm_schema(item: pystac.Item) -> list[str]:
    """Validate item against STAC extension schemas. Returns error messages (empty = valid)."""
    errors: list[str] = []
    try:
        item.validate()
    except pystac.errors.STACValidationError as e:
        errors.append(str(e))
    return errors


def _load_keywords_schema() -> dict:
    ref = importlib.resources.files("fair_models.schemas").joinpath("keywords.json")
    return json.loads(ref.read_text(encoding="utf-8"))


def validate_compatibility(
    base_model_item: pystac.Item,
    dataset_item: pystac.Item,
) -> list[str]:
    """Check keyword + task compatibility between base model and dataset.

    Rules:
    - At least one keyword in common
    - mlm:tasks must intersect with label:tasks (via task_label_mapping)
    - Keywords must be in allowed vocabulary
    """
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
