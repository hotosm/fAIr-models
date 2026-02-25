"""CI workflow for fAIr-models UNet example.

Commands: init, register, finetune, promote, predict, all, clean
Run from project root: python examples/unet/run.py <command>
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pystac
import yaml
from zenml.client import Client

from fair.stac.builders import build_dataset_item
from fair.stac.catalog_manager import StacCatalogManager
from fair.stac.collections import initialize_catalog
from fair.stac.constants import BASE_MODELS_COLLECTION, DATASETS_COLLECTION, LOCAL_MODELS_COLLECTION
from fair.stac.validators import validate_compatibility, validate_mlm_schema
from fair.zenml.config import generate_inference_config, generate_training_config
from fair.zenml.promotion import promote_model_version, publish_promoted_model
from models.example_unet.pipeline import inference_pipeline, training_pipeline

CATALOG_PATH = "stac_catalog/catalog.json"
BASE_MODEL_ID = "example-unet"
DATASET_ID = "buildings-banepa"
MODEL_NAME = "example-unet-finetuned-banepa"
STAC_ITEM = "models/example_unet/stac-item.json"
TRAIN_OAM = "data/sample/train/oam"
TRAIN_OSM = "data/sample/train/osm"
PREDICT_OAM = "data/sample/predict/oam"
CONFIG_DIR = Path("examples/unet/config")


def init() -> None:
    subprocess.run(["zenml", "init"], check=True, capture_output=True)
    Path("artifacts").mkdir(exist_ok=True)
    initialize_catalog(CATALOG_PATH)
    print("init: ok")


def register() -> None:
    cat = StacCatalogManager(CATALOG_PATH)

    base = pystac.Item.from_file(STAC_ITEM)
    if errs := validate_mlm_schema(base):
        sys.exit(f"MLM invalid: {errs}")
    pub = cat.publish_item(BASE_MODELS_COLLECTION, base)
    print(f"register: base-model {pub.id} v{pub.properties['version']}")

    labels = sorted(Path(TRAIN_OSM).glob("*.geojson"))
    if not labels:
        sys.exit(f"No .geojson in {TRAIN_OSM}")
    ds = build_dataset_item(
        item_id=DATASET_ID,
        dt=datetime.now(UTC),
        label_type="vector",
        label_tasks=["segmentation"],
        label_classes=[{"name": "building", "classes": ["building"]}],
        keywords=["building", "semantic-segmentation", "polygon"],
        chips_href=TRAIN_OAM,
        labels_href=str(labels[0]),
    )
    pub = cat.publish_item(DATASETS_COLLECTION, ds)
    print(f"register: dataset {pub.id} v{pub.properties['version']}")


def finetune() -> None:
    cat = StacCatalogManager(CATALOG_PATH)
    base = cat.get_item(BASE_MODELS_COLLECTION, BASE_MODEL_ID)
    ds = cat.get_item(DATASETS_COLLECTION, DATASET_ID)
    if errs := validate_compatibility(base, ds):
        sys.exit(f"Incompatible: {errs}")

    cfg_data = generate_training_config(base, ds, MODEL_NAME, {"epochs": 1})
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    cfg = CONFIG_DIR / "generated_train.yaml"
    cfg.write_text(yaml.dump(cfg_data, sort_keys=False))

    run = training_pipeline.with_options(config_path=str(cfg))()
    assert run is not None
    print(f"finetune: {run.id} ({run.status})")


def promote() -> None:
    client = Client()
    versions = client.list_model_versions(model=MODEL_NAME, sort_by="desc:created")
    if not versions:
        sys.exit(f"No versions for {MODEL_NAME}")
    latest = versions[0].number

    promote_model_version(MODEL_NAME, latest)

    cat = StacCatalogManager(CATALOG_PATH)
    item = publish_promoted_model(
        model_name=MODEL_NAME,
        version=latest,
        catalog_manager=cat,
        base_model_item_id=BASE_MODEL_ID,
        dataset_item_id=DATASET_ID,
    )
    print(f"promote: {item.id}")


def predict() -> None:
    cat = StacCatalogManager(CATALOG_PATH)
    items = cat.list_items(LOCAL_MODELS_COLLECTION)
    active = [i for i in items if not i.properties.get("deprecated")]
    if not active:
        sys.exit("No active local model")

    cfg_data = generate_inference_config(active[-1], PREDICT_OAM)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    cfg = CONFIG_DIR / "generated_inference.yaml"
    cfg.write_text(yaml.dump(cfg_data, sort_keys=False))

    run = inference_pipeline.with_options(config_path=str(cfg), enable_cache=False)()
    assert run is not None
    print(f"predict: {run.id} ({run.status})")


def all_steps() -> None:
    for step in (init, register, finetune, promote, predict):
        step()


def clean() -> None:
    for d in ("stac_catalog", "artifacts"):
        if (p := Path(d)).exists():
            shutil.rmtree(p)
            print(f"clean: {d}")
    for f in CONFIG_DIR.glob("generated_*.yaml"):
        f.unlink()
    preds = Path("data/sample/predict/predictions")
    if preds.exists():
        shutil.rmtree(preds)
    print("clean: ok")


COMMANDS: dict[str, Callable[[], None]] = {
    "init": init,
    "register": register,
    "finetune": finetune,
    "promote": promote,
    "predict": predict,
    "all": all_steps,
    "clean": clean,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="run.py", description="fAIr-models UNet CI workflow")
    parser.add_argument("command", choices=COMMANDS)
    COMMANDS[parser.parse_args().command]()
