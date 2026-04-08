from __future__ import annotations

import importlib.resources
import json
from typing import Any

from fair.stac.constants import FAIR_BASE_MODEL_SCHEMA, FAIR_DATASET_SCHEMA, FAIR_LOCAL_MODEL_SCHEMA

_SCHEMA_FILES: dict[str, str] = {
    FAIR_BASE_MODEL_SCHEMA: "v1.0.0/base-model/schema.json",
    FAIR_DATASET_SCHEMA: "v1.0.0/dataset/schema.json",
    FAIR_LOCAL_MODEL_SCHEMA: "v1.0.0/local-model/schema.json",
}

_loaded: dict[str, dict[str, Any]] = {}


def load_fair_schemas() -> dict[str, dict[str, Any]]:
    if _loaded:
        return _loaded
    for schema_uri, path in _SCHEMA_FILES.items():
        ref = importlib.resources.files("fair.schemas").joinpath(path)
        _loaded[schema_uri] = json.loads(ref.read_text(encoding="utf-8"))
    return _loaded


def register_fair_schemas() -> None:
    from pystac.validation import JsonSchemaSTACValidator, RegisteredValidator

    validator = RegisteredValidator.get_validator()
    if not isinstance(validator, JsonSchemaSTACValidator):
        return
    validator.schema_cache.update(load_fair_schemas())
