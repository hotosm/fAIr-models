from __future__ import annotations

import logging
from typing import Annotated, Any

import pystac
from annotated_types import Ge
from zenml.client import Client
from zenml.enums import ModelStages

from fair.stac.backend import StacBackend
from fair.stac.builders import build_local_model_item
from fair.stac.constants import BASE_MODELS_COLLECTION, DATASETS_COLLECTION, LOCAL_MODELS_COLLECTION
from fair.stac.versioning import deprecate_and_link_successor, find_previous_active_item
from fair.utils.data import s3_uri_to_http_url
from fair.zenml.metrics import read_fair_metrics, read_training_wall_time

log = logging.getLogger(__name__)


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
    catalog_manager: StacBackend,
    base_model_item_id: str,
    dataset_item_id: str,
    user_id: str,
    description: str,
    *,
    keywords: list[str] | None = None,
    geometry: dict[str, Any] | None = None,
    thumbnail_href: str | None = None,
) -> pystac.Item:
    client = Client()
    mv = client.get_model_version(model_name, version)

    # Item ID is the ZenML model version UUID: stable, unique, length-independent
    new_item_id = str(mv.id)

    if catalog_manager.item_exists(LOCAL_MODELS_COLLECTION, new_item_id):
        log.warning("%s v%d already promoted as STAC item %s; skipping", model_name, version, new_item_id)
        return catalog_manager.get_item(LOCAL_MODELS_COLLECTION, new_item_id)

    # Extract tunable hyperparams from the training run, exclude infra-level keys
    _INFRA_KEYS = {
        "base_model_weights",
        "dataset_chips",
        "dataset_labels",
        "num_classes",
        "model_name",
        "base_model_id",
        "dataset_id",
        "class_names",
    }
    hyperparams: dict[str, Any] = {}
    training_started_at: str | None = None
    training_ended_at: str | None = None
    training_duration_seconds: float | None = None
    run_links = client.list_model_version_pipeline_run_links(model_version_id=mv.id)
    if not run_links.items:
        log.warning(
            "No pipeline run links found for %s v%d; training metadata will be empty",
            model_name,
            version,
        )
    if run_links.items:
        run = run_links.items[0].pipeline_run
        step = run.steps.get("train_model")
        raw_params: dict[str, Any] | None = step.config.parameters if step else run.config.parameters
        hyperparams = (raw_params or {}).get("hyperparameters", {})
        if not hyperparams:
            hyperparams = {k: v for k, v in (raw_params or {}).items() if k not in _INFRA_KEYS}
        if step and step.start_time is not None:
            training_started_at = step.start_time.isoformat()
            if step.end_time is not None:
                training_ended_at = step.end_time.isoformat()

    raw_meta = dict(mv.run_metadata or {})
    wall_time = read_training_wall_time(raw_meta)
    if wall_time is not None:
        training_duration_seconds = wall_time
    metrics = read_fair_metrics(raw_meta)
    split_info: dict[str, Any] | None = raw_meta.get("fair/split")

    weights_art = mv.get_artifact("trained_model")
    if weights_art is None:
        msg = f"No model artifact found for {model_name} v{version}"
        raise RuntimeError(msg)
    model_href = s3_uri_to_http_url(weights_art.uri)
    artifact_version_id = str(weights_art.id)

    base_model_item = catalog_manager.get_item(BASE_MODELS_COLLECTION, base_model_item_id)

    # Geometry: prefer caller-provided, fallback to dataset item geometry
    dataset_item = catalog_manager.get_item(DATASETS_COLLECTION, dataset_item_id)
    dataset_title: str = dataset_item.properties.get("title", dataset_item_id)
    resolved_geometry = geometry
    if resolved_geometry is None:
        resolved_geometry = dataset_item.geometry
        labeled_chip_count: int | None = dataset_item.properties.get("fair:chip_count")
    else:
        labeled_chip_count = None

    kw = keywords if keywords is not None else base_model_item.properties.get("keywords", [])

    # Absolute hrefs for derived_from foreign keys
    base_model_href = catalog_manager.item_href(BASE_MODELS_COLLECTION, base_model_item_id)
    dataset_href = catalog_manager.item_href(DATASETS_COLLECTION, dataset_item_id)
    self_href = catalog_manager.item_href(LOCAL_MODELS_COLLECTION, new_item_id)

    prev_item = find_previous_active_item(
        catalog_manager,
        LOCAL_MODELS_COLLECTION,
        "mlm:name",
        model_name,
        new_item_id,
    )
    predecessor_href = catalog_manager.item_href(LOCAL_MODELS_COLLECTION, prev_item.id) if prev_item else None

    title = f"{model_name} v{version}"

    item = build_local_model_item(
        base_model_item=base_model_item,
        item_id=new_item_id,
        model_href=model_href,
        mlm_hyperparameters=hyperparams,
        keywords=kw,
        base_model_href=base_model_href,
        dataset_href=dataset_href,
        version=str(version),
        title=title,
        description=description,
        user_id=user_id,
        mlm_name=model_name,
        geometry=resolved_geometry,
        metrics=metrics,
        labeled_chip_count=labeled_chip_count,
        thumbnail_href=thumbnail_href,
        predecessor_version_href=predecessor_href,
        self_href=self_href,
        zenml_artifact_version_id=artifact_version_id,
        training_started_at=training_started_at,
        training_ended_at=training_ended_at,
        training_duration_seconds=training_duration_seconds,
        base_model_id=base_model_item_id,
        dataset_id=dataset_item_id,
        dataset_title=dataset_title,
        split_info=split_info,
    )

    if prev_item:
        deprecate_and_link_successor(catalog_manager, LOCAL_MODELS_COLLECTION, prev_item, self_href)
        log.info("STAC: deprecated %s, added successor-version -> %s", prev_item.id, new_item_id)

    published = catalog_manager.publish_item(LOCAL_MODELS_COLLECTION, item)
    log.info("STAC: published %s to local-models", new_item_id)
    return published


def archive_model_version(
    model_name: str,
    version: Annotated[int, Ge(1)],
    catalog_manager: StacBackend,
) -> pystac.Item:
    client = Client()
    mv = client.get_model_version(model_name, version)
    mv.set_stage(ModelStages.ARCHIVED, force=True)
    log.info("ZenML: %s v%d -> archived", model_name, version)

    item_id = str(mv.id)
    item = catalog_manager.deprecate_item(LOCAL_MODELS_COLLECTION, item_id)
    log.info("STAC: deprecated %s", item_id)
    return item


def delete_model_version(
    model_name: str,
    version: Annotated[int, Ge(1)],
    catalog_manager: StacBackend,
) -> None:
    client = Client()
    mv = client.get_model_version(model_name, version)
    client.delete_model_version(mv.id)
    log.info("ZenML: deleted %s v%d", model_name, version)

    item_id = str(mv.id)
    catalog_manager.delete_item(LOCAL_MODELS_COLLECTION, item_id)
    log.info("STAC: removed %s", item_id)


def delete_model(
    model_name: str,
    catalog_manager: StacBackend,
) -> None:
    client = Client()
    # STAC first — need IDs before ZenML deletes the model
    items = catalog_manager.list_items(LOCAL_MODELS_COLLECTION)
    for item in items:
        if item.properties.get("mlm:name") == model_name:
            catalog_manager.delete_item(LOCAL_MODELS_COLLECTION, item.id)
            log.info("STAC: removed %s", item.id)

    client.delete_model(model_name)
    log.info("ZenML: deleted model %s", model_name)
