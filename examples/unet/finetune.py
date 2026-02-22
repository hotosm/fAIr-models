"""Finetune example-unet on the Banepa buildings dataset.

Reads base model + dataset from catalog, validates compatibility,
generates a ZenML config, and triggers the training pipeline.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from fair_models.stac.catalog_manager import StacCatalogManager
from fair_models.stac.constants import BASE_MODELS_COLLECTION, DATASETS_COLLECTION
from fair_models.stac.validators import validate_compatibility
from fair_models.zenml.config import generate_training_config
from models.example_unet.pipeline import training_pipeline

CATALOG_PATH = "stac_catalog/catalog.json"
BASE_MODEL_ID = "example-unet"
DATASET_ID = "buildings-banepa"
MODEL_NAME = "example-unet-finetuned-banepa"
CONFIG_PATH = Path("examples/building_segmentation/config/generated_train.yaml")
OVERRIDES = {"epochs": 1}  # 1 epoch for quick validation; remove to use default 15

catalog = StacCatalogManager(CATALOG_PATH)
base_model_item = catalog.get_item(BASE_MODELS_COLLECTION, BASE_MODEL_ID)
dataset_item = catalog.get_item(DATASETS_COLLECTION, DATASET_ID)

errors = validate_compatibility(base_model_item, dataset_item)
if errors:
    msg = "Compatibility check failed:\n" + "\n".join(errors)
    raise ValueError(msg)

config = generate_training_config(base_model_item, dataset_item, MODEL_NAME, OVERRIDES)
CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
CONFIG_PATH.write_text(yaml.dump(config, sort_keys=False))
print(f"Config written to {CONFIG_PATH}")

run = training_pipeline.with_options(config_path=str(CONFIG_PATH))()
assert run is not None
print(f"Pipeline run: {run.id}  status: {run.status}")
