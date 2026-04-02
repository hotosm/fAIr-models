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
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pystac
import yaml
from zenml.client import Client
from zenml.enums import StackComponentType

from fair.stac.backend import StacBackend
from fair.stac.builders import build_dataset_item
from fair.stac.catalog_manager import StacCatalogManager
from fair.stac.collections import initialize_catalog
from fair.stac.constants import BASE_MODELS_COLLECTION, DATASETS_COLLECTION, LOCAL_MODELS_COLLECTION
from fair.stac.validators import validate_compatibility, validate_mlm_schema
from fair.stac.versioning import deprecate_and_link_successor, find_previous_active_item
from fair.utils.data import count_chips, create_dataset_archive, upload_item_assets
from fair.zenml.config import generate_inference_config, generate_training_config
from fair.zenml.promotion import promote_model_version, publish_promoted_model

CATALOG_PATH = "stac_catalog/catalog.json"
BASE_MODEL_ID = "example-unet"
DATASET_TITLE = "buildings-banepa"
DATASET_DESCRIPTION = "OpenAerialMap chips with OSM building footprints for Banepa, Nepal."
MODEL_NAME = "example-unet-finetuned-banepa"
STAC_ITEM = "models/example_unet/stac-item.json"
TRAIN_OAM = "data/sample/train/oam"
TRAIN_OSM = "data/sample/train/osm"
PREDICT_OAM = "data/sample/predict/oam"
CONFIG_DIR = Path("examples/unet/config")

SOURCE_IMAGERY_HREF = "https://tiles.openaerialmap.org/62d85d11d8499800053796c1/0/62d85d11d8499800053796c2/{z}/{x}/{y}"
DATASET_LICENSE = "CC-BY-4.0"
DATASET_PROVIDERS = [
    {"name": "HOTOSM", "roles": ["producer", "licensor"], "url": "https://www.hotosm.org"},
]
DATASET_LABEL_DESCRIPTION = "Building footprints manually labeled from OpenAerialMap imagery"
DATASET_LABEL_METHODS = ["manual"]


@dataclass
class RunConfig:
    stac_api_url: str | None = None
    dsn: str | None = None
    data_prefix: str | None = None
    user_id: str = "anonymous"


def _get_backend(cfg: RunConfig) -> StacBackend:
    if cfg.stac_api_url:
        from fair.stac.pgstac_backend import PgStacBackend

        if not cfg.dsn:
            msg = "--dsn is required when --stac-api-url is set"
            raise ValueError(msg)
        return PgStacBackend(dsn=cfg.dsn, stac_api_url=cfg.stac_api_url)
    return StacCatalogManager(CATALOG_PATH)


def _is_remote(href: str) -> bool:
    return "://" in href


def _item_href(backend: StacBackend, collection_id: str, item: pystac.Item) -> str:
    if hasattr(backend, "stac_api_url"):
        return f"{backend.stac_api_url}/collections/{collection_id}/items/{item.id}"
    return f"../{collection_id}/{item.id}/{item.id}.json"


def init(cfg: RunConfig) -> None:
    subprocess.run(["zenml", "init"], check=True, capture_output=True)
    Path("artifacts").mkdir(exist_ok=True)
    if not cfg.stac_api_url:
        initialize_catalog(CATALOG_PATH)
    print("init: ok")


def register(cfg: RunConfig) -> None:
    cat = _get_backend(cfg)

    base = pystac.Item.from_file(STAC_ITEM)
    if errs := validate_mlm_schema(base):
        sys.exit(f"MLM invalid: {errs}")
    pub = cat.publish_item(BASE_MODELS_COLLECTION, base)
    print(f"register: base-model {pub.id} v{pub.properties['version']}")

    local_labels = sorted(Path(TRAIN_OSM).glob("*.geojson"))
    if not local_labels:
        sys.exit(f"No .geojson in {TRAIN_OSM}")

    from fair.stac.builders import _geometry_and_bbox_from_geojson

    geometry, bbox = _geometry_and_bbox_from_geojson(str(local_labels[0]))

    chips_href = TRAIN_OAM
    labels_href = str(local_labels[0])

    chip_count = count_chips(chips_href)

    prev = find_previous_active_item(cat, DATASETS_COLLECTION, "title", DATASET_TITLE)
    version = str(int(prev.properties["version"]) + 1) if prev else "1"
    predecessor_href = _item_href(cat, DATASETS_COLLECTION, prev) if prev else None

    archive_path = Path("artifacts") / f"{DATASET_TITLE}-v{version}.zip"
    labels_dir = str(Path(labels_href).parent)
    create_dataset_archive(
        chips_dir=chips_href,
        labels_dir=labels_dir,
        output_path=str(archive_path),
    )

    ds = build_dataset_item(
        dt=datetime.now(UTC),
        label_type="vector",
        label_tasks=["segmentation"],
        label_classes=[{"name": "building", "classes": ["building"]}],
        keywords=["building", "semantic-segmentation", "polygon"],
        chips_href=chips_href,
        labels_href=labels_href,
        title=DATASET_TITLE,
        description=DATASET_DESCRIPTION,
        user_id=cfg.user_id,
        chip_count=chip_count if chip_count > 0 else None,
        geometry=geometry,
        bbox=bbox,
        version=version,
        license_id=DATASET_LICENSE,
        providers=DATASET_PROVIDERS,
        label_description=DATASET_LABEL_DESCRIPTION,
        label_methods=DATASET_LABEL_METHODS,
        source_imagery_href=SOURCE_IMAGERY_HREF,
        predecessor_version_href=predecessor_href,
        download_href=str(archive_path),
    )

    if cfg.data_prefix:
        upload_item_assets(ds, cfg.data_prefix, DATASETS_COLLECTION)

    if prev:
        self_href = _item_href(cat, DATASETS_COLLECTION, ds)
        deprecate_and_link_successor(cat, DATASETS_COLLECTION, prev, self_href)

    pub = cat.publish_item(DATASETS_COLLECTION, ds)
    print(f"register: dataset {pub.id} v{pub.properties['version']}")


def finetune(cfg: RunConfig) -> None:
    cat = _get_backend(cfg)
    base = cat.get_item(BASE_MODELS_COLLECTION, BASE_MODEL_ID)
    try:
        ds = cat.get_item(DATASETS_COLLECTION, DATASET_TITLE)
    except KeyError:
        sys.exit(f"Dataset '{DATASET_TITLE}' not found - run register first")
    if errs := validate_compatibility(base, ds):
        sys.exit(f"Incompatible: {errs}")

    trackers = Client().active_stack_model.components.get(
        StackComponentType.EXPERIMENT_TRACKER,
        [],
    )
    tracker_name = trackers[0].name if trackers else None
    cfg_data = generate_training_config(
        base,
        ds,
        MODEL_NAME,
        {"epochs": 1},
        experiment_tracker=tracker_name,
    )
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    train_cfg = CONFIG_DIR / "generated_train.yaml"
    train_cfg.write_text(yaml.dump(cfg_data, sort_keys=False))

    from models.example_unet.pipeline import training_pipeline

    run = training_pipeline.with_options(config_path=str(train_cfg))()
    assert run is not None
    print(f"finetune: {run.id} ({run.status})")


def promote(cfg: RunConfig) -> None:
    client = Client()
    versions = client.list_model_versions(model=MODEL_NAME, sort_by="desc:created")
    if not versions:
        sys.exit(f"No versions for {MODEL_NAME}")
    latest = versions[0].number

    promote_model_version(MODEL_NAME, latest)

    cat = _get_backend(cfg)
    item = publish_promoted_model(
        model_name=MODEL_NAME,
        version=latest,
        catalog_manager=cat,
        base_model_item_id=BASE_MODEL_ID,
        dataset_item_id=DATASET_TITLE,
        user_id=cfg.user_id,
        description=f"UNet finetuned on {DATASET_TITLE}",
    )
    print(f"promote: {item.id}")


def predict(cfg: RunConfig) -> None:
    cat = _get_backend(cfg)
    items = cat.list_items(LOCAL_MODELS_COLLECTION)
    active = [i for i in items if not i.properties.get("deprecated")]
    if not active:
        sys.exit("No active local model")

    input_images = f"{cfg.data_prefix}/predict/oam" if cfg.data_prefix else PREDICT_OAM
    if cfg.data_prefix:
        from fair.utils.data import _upload_local_directory

        _upload_local_directory(Path(PREDICT_OAM), input_images)
    cfg_data = generate_inference_config(
        active[-1],
        input_images,
    )
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    inf_cfg = CONFIG_DIR / "generated_inference.yaml"
    inf_cfg.write_text(yaml.dump(cfg_data, sort_keys=False))

    from models.example_unet.pipeline import inference_pipeline

    run = inference_pipeline.with_options(config_path=str(inf_cfg), enable_cache=False)()
    assert run is not None
    print(f"predict: {run.id} ({run.status})")


def all_steps(cfg: RunConfig) -> None:
    for step in (init, register, finetune, promote, predict):
        step(cfg)


def clean(cfg: RunConfig) -> None:
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


COMMANDS: dict[str, Callable[[RunConfig], None]] = {
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
    parser.add_argument("--stac-api-url", help="STAC API URL (enables PgStacBackend)")
    parser.add_argument("--dsn", help="Postgres DSN for pgstac writes (default: PG env vars)")
    parser.add_argument("--data-prefix", help="S3 prefix for dataset assets (e.g. s3://bucket/data/sample)")
    parser.add_argument("--user-id", default="anonymous", help="OSM user ID stored as fair:user_id")
    args = parser.parse_args()
    run_config = RunConfig(
        stac_api_url=args.stac_api_url,
        dsn=args.dsn,
        data_prefix=args.data_prefix,
        user_id=args.user_id,
    )
    COMMANDS[args.command](run_config)
