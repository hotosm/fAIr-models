from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Annotated, Any

import pystac
from annotated_types import Ge
from zenml.client import Client
from zenml.enums import ModelStages

from fair_models.stac.builders import build_local_model_item
from fair_models.stac.catalog_manager import StacCatalogManager
from fair_models.stac.constants import BASE_MODELS_COLLECTION, LOCAL_MODELS_COLLECTION

log = logging.getLogger(__name__)


def _stac_item_id(model_name: str, version: Annotated[int, Ge(1)]) -> str:
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


def promote_model_version(model_name: str, version: Annotated[int, Ge(1)]) -> None:
    # ZenML auto-archives previous production version
    client = Client()
    mv = client.get_model_version(model_name, version)
    mv.set_stage(ModelStages.PRODUCTION, force=True)
    log.info("ZenML: %s v%d -> production", model_name, version)


def publish_promoted_model(
    model_name: str,
    version: Annotated[int, Ge(1)],
    catalog_manager: StacCatalogManager,
    base_model_item_id: str,
    dataset_item_id: str,
    *,
    keywords: list[str] | None = None,
    geometry: dict[str, Any] | None = None,
) -> pystac.Item:
    client = Client()
    mv = client.get_model_version(model_name, version)

    # Exclude infra params, keep only tunable hparams.
    # pipeline_runs is deprecated but replacement isn't stable in ZenML 0.93.x
    _INFRA_KEYS = {"base_model_weights", "dataset_chips", "dataset_labels", "num_classes"}
    hyperparams: dict[str, Any] = {}
    runs = mv.pipeline_runs
    if runs:
        run = next(iter(runs.values()))
        step = run.steps.get("train_model")
        raw_params: dict[str, Any] | None = step.config.parameters if step else run.config.parameters
        hyperparams = {k: v for k, v in (raw_params or {}).items() if k not in _INFRA_KEYS}

    weights_art = mv.get_artifact("training_pipeline::train_model::output")
    if weights_art is None:
        msg = f"No model artifact found for {model_name} v{version}"
        raise RuntimeError(msg)
    # URI = artifact store directory; materializer handles framework-specific files within
    model_href = weights_art.uri
    artifact_version_id = str(weights_art.id)

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
        zenml_artifact_version_id=artifact_version_id,
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
    version: Annotated[int, Ge(1)],
    catalog_manager: StacCatalogManager,
) -> pystac.Item:
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
    version: Annotated[int, Ge(1)],
    catalog_manager: StacCatalogManager,
) -> None:
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
    client = Client()
    # STAC first — need IDs before ZenML deletes the model
    items = catalog_manager.list_items(LOCAL_MODELS_COLLECTION)
    for item in items:
        if item.id.startswith(f"{model_name}-v"):
            catalog_manager.delete_item(LOCAL_MODELS_COLLECTION, item.id)
            log.info("STAC: removed %s", item.id)

    client.delete_model(model_name)
    log.info("ZenML: deleted model %s", model_name)
