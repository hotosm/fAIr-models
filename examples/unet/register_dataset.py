"""Register downloaded Banepa dataset as a STAC item in the local catalog."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from fair_models.stac.builders import build_dataset_item
from fair_models.stac.catalog_manager import StacCatalogManager
from fair_models.stac.constants import DATASETS_COLLECTION

CATALOG_PATH = "stac_catalog/catalog.json"
ITEM_ID = "buildings-banepa"
CHIPS_HREF = "data/banepa_test/oam"
_OSM_DIR = Path("data/banepa_test/osm")
KEYWORDS = ["building", "semantic-segmentation", "polygon"]

_candidates = sorted(_OSM_DIR.glob("*.geojson"))
if not _candidates:
    raise FileNotFoundError(f"No .geojson in {_OSM_DIR}. Run download.py first.")
if len(_candidates) > 1:
    raise RuntimeError(f"Multiple .geojson files in {_OSM_DIR}: {_candidates}. Expected exactly one.")
LABELS_HREF = str(_candidates[0])

catalog = StacCatalogManager(CATALOG_PATH)

item = build_dataset_item(
    item_id=ITEM_ID,
    dt=datetime.now(UTC),
    label_type="vector",
    label_tasks=["segmentation"],
    label_classes=[{"name": "building", "classes": ["building"]}],
    keywords=KEYWORDS,
    chips_href=CHIPS_HREF,
    labels_href=LABELS_HREF,
)

published_item = catalog.publish_item(DATASETS_COLLECTION, item)
print(f"Registered dataset: {published_item.id}, version: {published_item.properties.get('version')}")
print(f"  chips:  {CHIPS_HREF}")
print(f"  labels: {LABELS_HREF}")
