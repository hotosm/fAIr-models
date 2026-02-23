"""Run inference using the promoted local model."""

from __future__ import annotations

from pathlib import Path

import yaml

from fair_models.stac.catalog_manager import StacCatalogManager
from fair_models.stac.constants import LOCAL_MODELS_COLLECTION
from fair_models.zenml.config import generate_inference_config
from models.example_unet.pipeline import inference_pipeline

CATALOG_PATH = "stac_catalog/catalog.json"
LOCAL_MODEL_ID = "example-unet-finetuned-banepa-v5"
INPUT_IMAGES = "data/sample/predict/oam"
CONFIG_PATH = "examples/unet/config/generated_inference.yaml"

catalog = StacCatalogManager(CATALOG_PATH)
model_item = catalog.get_item(LOCAL_MODELS_COLLECTION, LOCAL_MODEL_ID)

config = generate_inference_config(model_item, INPUT_IMAGES)

Path(CONFIG_PATH).parent.mkdir(parents=True, exist_ok=True)
Path(CONFIG_PATH).write_text(yaml.dump(config, sort_keys=False))
print(f"Config written to {CONFIG_PATH}")

run = inference_pipeline.with_options(config_path=CONFIG_PATH)()
assert run is not None
print(f"Pipeline run: {run.id}  status: {run.status}")
