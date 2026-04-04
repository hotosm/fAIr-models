from __future__ import annotations

import argparse
import importlib
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import pystac
import yaml
from zenml.client import Client
from zenml.enums import StackComponentType

from fair.stac.builders import build_dataset_item, geometry_and_bbox_from_geojson
from fair.stac.catalog_manager import StacCatalogManager
from fair.stac.collections import initialize_catalog
from fair.stac.constants import (
    BASE_MODELS_COLLECTION,
    DATASETS_COLLECTION,
    LOCAL_MODELS_COLLECTION,
)
from fair.stac.validators import validate_compatibility, validate_mlm_schema
from fair.stac.versioning import archive_previous_version, find_previous_active_item
from fair.utils.data import count_chips, create_dataset_archive, upload_item_assets
from fair.zenml.config import generate_inference_config, generate_training_config
from fair.zenml.promotion import promote_model_version, publish_promoted_model

if TYPE_CHECKING:
    from fair.stac.pgstac_backend import PgStacBackend

CATALOG_PATH = "stac_catalog/catalog.json"

_DEFAULT_PROVIDERS = [
    {
        "name": "HOTOSM",
        "roles": ["producer", "licensor"],
        "url": "https://www.hotosm.org",
    },
]


@dataclass
class DatasetConfig:
    title: str
    description: str
    label_type: Literal["vector", "raster"]
    label_tasks: list[str]
    label_classes: list[dict[str, Any]]
    keywords: list[str]
    train_chips_path: str
    train_labels_path: str
    predict_images_path: str
    geojson_dir: str = "data/sample/train/osm"
    labels_pattern: str | None = None
    source_imagery_href: str | None = None
    license_id: str = "CC-BY-4.0"
    providers: list[dict[str, Any]] = field(default_factory=lambda: list(_DEFAULT_PROVIDERS))
    label_description: str = ""
    label_methods: list[str] = field(default_factory=lambda: ["manual"])


@dataclass
class RunConfig:
    stac_api_url: str | None = None
    dsn: str | None = None
    data_prefix: str | None = None
    user_id: str = "anonymous"


class FairWorkflowRunner:
    def __init__(
        self,
        *,
        base_model_id: str,
        model_name: str,
        stac_item_path: str,
        pipeline_module: str,
        config_dir: str,
        dataset: DatasetConfig,
        finetune_overrides: dict[str, Any] | None = None,
        promote_description: str = "",
    ) -> None:
        self.base_model_id = base_model_id
        self.model_name = model_name
        self.stac_item_path = stac_item_path
        self.pipeline_module = pipeline_module
        self.config_dir = Path(config_dir)
        self.dataset = dataset
        self.finetune_overrides = finetune_overrides
        self.promote_description = promote_description or f"{model_name} finetuned on {dataset.title}"

    def _get_backend(self, cfg: RunConfig) -> StacCatalogManager | PgStacBackend:
        if cfg.stac_api_url:
            from fair.stac.pgstac_backend import PgStacBackend

            if not cfg.dsn:
                msg = "--dsn is required when --stac-api-url is set"
                raise ValueError(msg)
            return PgStacBackend(dsn=cfg.dsn, stac_api_url=cfg.stac_api_url)
        return StacCatalogManager(CATALOG_PATH)

    def _resolve_labels(self) -> tuple[str, str]:
        ds = self.dataset
        if ds.labels_pattern:
            label_dir = Path(ds.train_labels_path)
            matches = sorted(label_dir.glob(ds.labels_pattern))
            if not matches:
                sys.exit(f"No files matching '{ds.labels_pattern}' in {label_dir}")
            return str(matches[0]), str(matches[0].parent)
        labels_path = Path(ds.train_labels_path)
        if not labels_path.exists():
            sys.exit(f"Labels not found at {ds.train_labels_path}")
        return str(labels_path), str(labels_path.parent)

    def init(self, cfg: RunConfig) -> None:
        subprocess.run(["zenml", "init"], check=True, capture_output=True)
        Path("artifacts").mkdir(exist_ok=True)
        if not cfg.stac_api_url:
            initialize_catalog(CATALOG_PATH)
        print("init: ok")

    def register(self, cfg: RunConfig) -> None:
        cat = self._get_backend(cfg)

        base = pystac.Item.from_file(self.stac_item_path)
        if errs := validate_mlm_schema(base):
            sys.exit(f"MLM invalid: {errs}")
        pub = cat.publish_item(BASE_MODELS_COLLECTION, base)
        print(f"register: base-model {pub.id} v{pub.properties['version']}")

        ds = self.dataset
        labels_href, labels_dir = self._resolve_labels()

        geojson_files = sorted(Path(ds.geojson_dir).glob("*.geojson"))
        if not geojson_files:
            sys.exit(f"No .geojson files in {ds.geojson_dir}")
        geometry, bbox = geometry_and_bbox_from_geojson(str(geojson_files[0]))

        chip_count = count_chips(ds.train_chips_path)
        prev = find_previous_active_item(cat, DATASETS_COLLECTION, "title", ds.title)
        version = str(int(prev.properties["version"]) + 1) if prev else "1"
        predecessor_href = cat.item_href(DATASETS_COLLECTION, prev.id) if prev else None

        archive_path = Path("artifacts") / f"{ds.title}-v{version}.zip"
        create_dataset_archive(
            chips_dir=ds.train_chips_path,
            labels_dir=labels_dir,
            output_path=str(archive_path),
        )

        dataset_item = build_dataset_item(
            dt=datetime.now(UTC),
            label_type=ds.label_type,
            label_tasks=ds.label_tasks,
            label_classes=ds.label_classes,
            keywords=ds.keywords,
            chips_href=ds.train_chips_path,
            labels_href=labels_href,
            title=ds.title,
            description=ds.description,
            user_id=cfg.user_id,
            chip_count=chip_count if chip_count > 0 else None,
            geometry=geometry,
            bbox=bbox,
            version=version,
            license_id=ds.license_id,
            providers=ds.providers,
            label_description=ds.label_description,
            label_methods=ds.label_methods,
            source_imagery_href=ds.source_imagery_href,
            predecessor_version_href=predecessor_href,
            download_href=str(archive_path),
        )

        if cfg.data_prefix:
            upload_item_assets(dataset_item, cfg.data_prefix, DATASETS_COLLECTION)

        if prev:
            successor_href = cat.item_href(DATASETS_COLLECTION, dataset_item.id)
            archive_previous_version(cat, DATASETS_COLLECTION, prev, successor_href)

        pub = cat.publish_item(DATASETS_COLLECTION, dataset_item)
        print(f"register: dataset {pub.id} v{pub.properties['version']}")

    def finetune(self, cfg: RunConfig) -> None:
        cat = self._get_backend(cfg)
        base = cat.get_item(BASE_MODELS_COLLECTION, self.base_model_id)
        try:
            ds = cat.get_item(DATASETS_COLLECTION, self.dataset.title)
        except KeyError:
            sys.exit(f"Dataset '{self.dataset.title}' not found. Run register first.")

        if errs := validate_compatibility(base, ds):
            sys.exit(f"Incompatible: {errs}")

        trackers = Client().active_stack_model.components.get(StackComponentType.EXPERIMENT_TRACKER, [])
        tracker_name = trackers[0].name if trackers else None
        cfg_data = generate_training_config(
            base,
            ds,
            self.model_name,
            self.finetune_overrides,
            experiment_tracker=tracker_name,
        )
        self.config_dir.mkdir(parents=True, exist_ok=True)
        train_cfg = self.config_dir / "generated_train.yaml"
        train_cfg.write_text(yaml.dump(cfg_data, sort_keys=False))

        mod = importlib.import_module(self.pipeline_module)
        run = mod.training_pipeline.with_options(config_path=str(train_cfg))()
        if run is None:
            raise RuntimeError("Training pipeline returned no run")
        print(f"finetune: {run.id} ({run.status})")

    def promote(self, cfg: RunConfig) -> None:
        client = Client()
        versions = client.list_model_versions(model=self.model_name, sort_by="desc:created")
        if not versions:
            sys.exit(f"No versions for {self.model_name}")
        latest = versions[0].number

        promote_model_version(self.model_name, latest)

        cat = self._get_backend(cfg)
        item = publish_promoted_model(
            model_name=self.model_name,
            version=latest,
            catalog_manager=cat,
            base_model_item_id=self.base_model_id,
            dataset_item_id=self.dataset.title,
            user_id=cfg.user_id,
            description=self.promote_description,
        )
        print(f"promote: {item.id}")

    def predict(self, cfg: RunConfig) -> None:
        cat = self._get_backend(cfg)
        items = cat.list_items(LOCAL_MODELS_COLLECTION)
        active = [i for i in items if not i.properties.get("deprecated")]
        if not active:
            sys.exit("No active local model")

        input_images = f"{cfg.data_prefix}/predict/oam" if cfg.data_prefix else self.dataset.predict_images_path
        if cfg.data_prefix:
            from fair.utils.data import _upload_local_directory

            _upload_local_directory(Path(self.dataset.predict_images_path), input_images)

        cfg_data = generate_inference_config(active[-1], input_images)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        inf_cfg = self.config_dir / "generated_inference.yaml"
        inf_cfg.write_text(yaml.dump(cfg_data, sort_keys=False))

        mod = importlib.import_module(self.pipeline_module)
        run = mod.inference_pipeline.with_options(config_path=str(inf_cfg), enable_cache=False)()
        if run is None:
            raise RuntimeError("Inference pipeline returned no run")
        print(f"predict: {run.id} ({run.status})")

    def verify(self, cfg: RunConfig) -> None:
        cat = self._get_backend(cfg)
        collections = [
            BASE_MODELS_COLLECTION,
            DATASETS_COLLECTION,
            LOCAL_MODELS_COLLECTION,
        ]
        total = 0
        for coll_id in collections:
            items = cat.list_items(coll_id)
            for item in items:
                cat.get_item(coll_id, item.id)
                total += 1
        print(f"verify: ok ({total} items across {len(collections)} collections)")

    def all_steps(self, cfg: RunConfig) -> None:
        for step_fn in (
            self.init,
            self.register,
            self.finetune,
            self.promote,
            self.predict,
            self.verify,
        ):
            step_fn(cfg)

    def clean(self, cfg: RunConfig) -> None:
        for d in ("stac_catalog", "artifacts"):
            if (p := Path(d)).exists():
                shutil.rmtree(p)
                print(f"clean: {d}")
        for f in self.config_dir.glob("generated_*.yaml"):
            f.unlink()
        preds = Path("data/sample/predict/predictions")
        if preds.exists():
            shutil.rmtree(preds)
        print("clean: ok")

    def run(self) -> None:
        from fair.utils import install_s3_cleanup_handler

        install_s3_cleanup_handler()

        commands: dict[str, Callable[[RunConfig], None]] = {
            "init": self.init,
            "register": self.register,
            "finetune": self.finetune,
            "promote": self.promote,
            "predict": self.predict,
            "verify": self.verify,
            "all": self.all_steps,
            "clean": self.clean,
        }

        parser = argparse.ArgumentParser(
            prog="run.py",
            description=f"fAIr-models {self.model_name} CI workflow",
        )
        parser.add_argument("command", choices=commands)
        parser.add_argument("--stac-api-url", help="STAC API URL (enables PgStacBackend)")
        parser.add_argument("--dsn", help="Postgres DSN for pgstac writes")
        parser.add_argument("--data-prefix", help="S3 prefix for dataset assets")
        parser.add_argument("--user-id", default="anonymous", help="OSM user ID")
        args = parser.parse_args()

        run_config = RunConfig(
            stac_api_url=args.stac_api_url,
            dsn=args.dsn,
            data_prefix=args.data_prefix,
            user_id=args.user_id,
        )
        commands[args.command](run_config)
