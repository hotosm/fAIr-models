from __future__ import annotations

import os
from datetime import UTC, datetime

import pystac
from pystac import CatalogType

from fair_models.stac.constants import (
    BASE_MODELS_COLLECTION,
    DATASET_EXTENSIONS,
    DATASETS_COLLECTION,
    LOCAL_MODELS_COLLECTION,
    MODEL_EXTENSIONS,
)


def create_base_models_collection() -> pystac.Collection:
    """base-models: model blueprints contributed via PR."""
    return pystac.Collection(
        id=BASE_MODELS_COLLECTION,
        description="Model blueprints contributed via PR. Each item is a complete model card.",
        extent=pystac.Extent(
            spatial=pystac.SpatialExtent(bboxes=[[-180, -90, 180, 90]]),
            temporal=pystac.TemporalExtent(intervals=[[datetime(2026, 1, 1, tzinfo=UTC), None]]),
        ),
        license="various",
        stac_extensions=MODEL_EXTENSIONS,
    )


def create_local_models_collection() -> pystac.Collection:
    """local-models: finetuned models, only promoted versions."""
    return pystac.Collection(
        id=LOCAL_MODELS_COLLECTION,
        description="Finetuned models produced by ZenML pipelines. Only promoted versions appear here.",
        extent=pystac.Extent(
            spatial=pystac.SpatialExtent(bboxes=[[-180, -90, 180, 90]]),
            temporal=pystac.TemporalExtent(intervals=[[datetime(2026, 1, 1, tzinfo=UTC), None]]),
        ),
        license="various",
        stac_extensions=MODEL_EXTENSIONS,
    )


def create_datasets_collection() -> pystac.Collection:
    """datasets: training data registered via fAIr UI/backend."""
    return pystac.Collection(
        id=DATASETS_COLLECTION,
        description="Training data registered via fAIr UI/backend.",
        extent=pystac.Extent(
            spatial=pystac.SpatialExtent(bboxes=[[-180, -90, 180, 90]]),
            temporal=pystac.TemporalExtent(intervals=[[datetime(2026, 1, 1, tzinfo=UTC), None]]),
        ),
        license="various",
        stac_extensions=DATASET_EXTENSIONS,
    )


def initialize_catalog(catalog_path: str) -> pystac.Catalog:
    """Create catalog.json + 3 empty collections. Saves to disk.

    returns existing catalog if already present.
    """
    if os.path.exists(catalog_path):
        return pystac.Catalog.from_file(catalog_path)

    catalog = pystac.Catalog(
        id="fair-models",
        description="fAIr model registry and dataset catalog",
    )

    catalog.add_child(create_base_models_collection())
    catalog.add_child(create_local_models_collection())
    catalog.add_child(create_datasets_collection())

    catalog_dir = os.path.dirname(catalog_path) or "."
    catalog.normalize_hrefs(catalog_dir)
    catalog.save(catalog_type=CatalogType.SELF_CONTAINED)

    return catalog
