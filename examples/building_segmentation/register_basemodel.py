"""Register example-unet base model from stac-item.json into the local catalog."""

from __future__ import annotations

import pystac

from fair_models.stac.catalog_manager import StacCatalogManager
from fair_models.stac.constants import BASE_MODELS_COLLECTION
from fair_models.stac.validators import validate_mlm_schema

CATALOG_PATH = "stac_catalog/catalog.json"
STAC_ITEM_JSON = "models/example_unet/stac-item.json"

item = pystac.Item.from_file(STAC_ITEM_JSON)
errors = validate_mlm_schema(item)
if errors:
    msg = "MLM schema validation failed:\n" + "\n".join(errors)
    raise ValueError(msg)

catalog = StacCatalogManager(CATALOG_PATH)
catalog.publish_item(BASE_MODELS_COLLECTION, item)
print(f"Registered base model: {item.id}")
print(f"  framework:     {item.properties.get('mlm:framework')}")
print(f"  architecture:  {item.properties.get('mlm:architecture')}")
print(f"  weights href:  {item.assets['model'].href}")
