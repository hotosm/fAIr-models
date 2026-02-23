"""Register example-unet base model into the local catalog."""

from __future__ import annotations

import pystac

from fair.stac.catalog_manager import StacCatalogManager
from fair.stac.constants import BASE_MODELS_COLLECTION
from fair.stac.validators import validate_mlm_schema

CATALOG_PATH = "stac_catalog/catalog.json"
STAC_ITEM_JSON = "models/example_unet/stac-item.json"

item = pystac.Item.from_file(STAC_ITEM_JSON)
errors = validate_mlm_schema(item)
if errors:
    msg = "MLM schema validation failed:\n" + "\n".join(errors)
    raise ValueError(msg)

catalog = StacCatalogManager(CATALOG_PATH)
published_item = catalog.publish_item(BASE_MODELS_COLLECTION, item)
print(f"Registered base model: {published_item.id}, version: {published_item.properties.get('version')}")
print(f"  framework:     {published_item.properties.get('mlm:framework')}")
print(f"  architecture:  {published_item.properties.get('mlm:architecture')}")
print(f"  weights href:  {published_item.assets['model'].href}")
