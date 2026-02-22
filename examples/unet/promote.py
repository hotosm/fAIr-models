"""Promote a trained model to production and publish its STAC item."""

from __future__ import annotations

from fair_models.stac.catalog_manager import StacCatalogManager
from fair_models.zenml.promotion import promote_model_version, publish_promoted_model

CATALOG_PATH = "stac_catalog/catalog.json"
MODEL_NAME = "example-unet-finetuned-banepa"
VERSION = 9
BASE_MODEL_ID = "example-unet"
DATASET_ID = "buildings-banepa"

promote_model_version(MODEL_NAME, VERSION)
print(f"ZenML: {MODEL_NAME} v{VERSION} -> production")

catalog = StacCatalogManager(CATALOG_PATH)
item = publish_promoted_model(
    model_name=MODEL_NAME,
    version=VERSION,
    catalog_manager=catalog,
    base_model_item_id=BASE_MODEL_ID,
    dataset_item_id=DATASET_ID,
)
print(f"STAC:  published {item.id}")
print(f"  model href:     {item.assets['model'].href}")
print(f"  hyperparameters: {item.properties.get('mlm:hyperparameters')}")
print(f"  derived_from:   {[lnk.target for lnk in item.get_links('derived_from')]}")
