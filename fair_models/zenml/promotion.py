"""ZenML model version promotion and STAC catalog synchronization.

Each function pairs a ZenML stage transition with the corresponding
StacCatalogManager call so the two registries stay in sync.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

import pystac
from zenml.client import Client
from zenml.enums import ModelStages

from fair_models.stac.builders import build_local_model_item
from fair_models.stac.catalog_manager import StacCatalogManager
from fair_models.stac.constants import BASE_MODELS_COLLECTION, LOCAL_MODELS_COLLECTION

log = logging.getLogger(__name__)


def _stac_item_id(model_name: str, version: int) -> str:
    return f"{model_name}-v{version}"


def _find_previous_production_item(
    catalog_manager: StacCatalogManager,
    model_name: str,
    exclude_version: int,
) -> pystac.Item | None:
    """Scan local-models for the current production item of *model_name*."""
    prefix = f"{model_name}-v"
    items = catalog_manager.list_items(LOCAL_MODELS_COLLECTION)
    for item in items:
        if not item.id.startswith(prefix):
            continue
        if item.properties.get("deprecated"):
            continue
        suffix = item.id.rsplit("-v", 1)
        if len(suffix) == 2 and suffix[1].isdigit() and int(suffix[1]) != exclude_version:
            return item
    return None


# Public API


def promote_model_version(model_name: str, version: int) -> None:
    """Set a ZenML model version stage to *production*.

    ZenML automatically archives the previous production version (if any).
    """
    client = Client()
    mv = client.get_model_version(model_name, version)
    mv.set_stage(ModelStages.PRODUCTION, force=True)
    log.info("ZenML: %s v%d -> production", model_name, version)


def publish_promoted_model(
    model_name: str,
    version: int,
    catalog_manager: StacCatalogManager,
    base_model_item_id: str,
    dataset_item_id: str,
    *,
    keywords: list[str] | None = None,
    geometry: dict[str, Any] | None = None,
) -> pystac.Item:
    """Read ZenML model version metadata, build + publish a local-model STAC item."""
    client = Client()
    mv = client.get_model_version(model_name, version)

    run_meta: dict[str, Any] = {}
    if mv.run_metadata:
        for key, val in mv.run_metadata.items():
            run_meta[key] = getattr(val, "value", val)

    hyperparams: dict[str, Any] = run_meta.get("mlm:hyperparameters", run_meta)

    model_artifact = mv.get_model_artifact("trained_model")
    model_href = model_artifact.uri if model_artifact is not None else ""

    base_model_item = catalog_manager.get_item(BASE_MODELS_COLLECTION, base_model_item_id)

    new_item_id = _stac_item_id(model_name, version)
    version_str = str(version)

    prev_item = _find_previous_production_item(catalog_manager, model_name, version)
    predecessor_id = prev_item.id if prev_item else None

    kw = keywords if keywords is not None else base_model_item.properties.get("keywords", [])

    item = build_local_model_item(
        base_model_item=base_model_item,
        item_id=new_item_id,
        dt=datetime.now(UTC),
        model_href=model_href,
        mlm_hyperparameters=hyperparams,
        keywords=kw,
        base_model_item_id=base_model_item_id,
        dataset_item_id=dataset_item_id,
        version=version_str,
        geometry=geometry,
        predecessor_version_item_id=predecessor_id,
    )

    if prev_item:
        catalog_manager.deprecate_item(LOCAL_MODELS_COLLECTION, prev_item.id)
        prev_item.add_link(pystac.Link(rel="successor-version", target=new_item_id))
        catalog_manager.publish_item(LOCAL_MODELS_COLLECTION, prev_item)
        log.info("STAC: deprecated %s, added successor-version -> %s", prev_item.id, new_item_id)

    published = catalog_manager.publish_item(LOCAL_MODELS_COLLECTION, item)
    log.info("STAC: published %s to local-models", new_item_id)
    return published


def archive_model_version(
    model_name: str,
    version: int,
    catalog_manager: StacCatalogManager,
) -> pystac.Item:
    """Archive a ZenML model version and deprecate its STAC item."""
    client = Client()
    mv = client.get_model_version(model_name, version)
    mv.set_stage(ModelStages.ARCHIVED, force=True)
    log.info("ZenML: %s v%d -> archived", model_name, version)

    item_id = _stac_item_id(model_name, version)
    item = catalog_manager.deprecate_item(LOCAL_MODELS_COLLECTION, item_id)
    log.info("STAC: deprecated %s", item_id)
    return item


def delete_model_version(
    model_name: str,
    version: int,
    catalog_manager: StacCatalogManager,
) -> None:
    """Delete a single ZenML model version and remove its STAC item."""
    client = Client()
    mv = client.get_model_version(model_name, version)
    client.delete_model_version(mv.id)
    log.info("ZenML: deleted %s v%d", model_name, version)

    item_id = _stac_item_id(model_name, version)
    catalog_manager.delete_item(LOCAL_MODELS_COLLECTION, item_id)
    log.info("STAC: removed %s", item_id)


def delete_model(
    model_name: str,
    catalog_manager: StacCatalogManager,
) -> None:
    """Delete all versions of a ZenML model and remove all STAC items."""
    client = Client()

    # STAC first so we still have IDs before ZenML deletes the model
    items = catalog_manager.list_items(LOCAL_MODELS_COLLECTION)
    for item in items:
        if item.id.startswith(f"{model_name}-v"):
            catalog_manager.delete_item(LOCAL_MODELS_COLLECTION, item.id)
            log.info("STAC: removed %s", item.id)

    client.delete_model(model_name)
    log.info("ZenML: deleted model %s", model_name)
