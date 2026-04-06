from __future__ import annotations

import importlib
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pystac
import yaml
from zenml.client import Client
from zenml.enums import StackComponentType

from fair.stac.builders import build_dataset_item, geometry_and_bbox_from_geojson
from fair.stac.catalog_manager import StacCatalogManager
from fair.stac.collections import initialize_catalog
from fair.stac.constants import BASE_MODELS_COLLECTION, DATASETS_COLLECTION, LOCAL_MODELS_COLLECTION
from fair.stac.validators import validate_compatibility, validate_mlm_schema
from fair.stac.versioning import archive_previous_version, find_previous_active_item
from fair.utils.data import count_chips, create_dataset_archive
from fair.zenml.config import generate_inference_config, generate_training_config
from fair.zenml.promotion import promote_model_version, publish_promoted_model
from fair.zenml.runner import DatasetConfig

if TYPE_CHECKING:
    from fair.stac.pgstac_backend import PgStacBackend

_DEFAULT_CATALOG_PATH = "stac_catalog/catalog.json"


class FairClientError(Exception):
    pass


class FairClient:
    def __init__(
        self,
        *,
        zenml_store_url: str | None = None,
        stac_api_url: str | None = None,
        dsn: str | None = None,
        catalog_path: str = _DEFAULT_CATALOG_PATH,
        user_id: str = "anonymous",
        config_dir: str = "config",
    ) -> None:
        self._zenml_store_url = zenml_store_url
        self._stac_api_url = stac_api_url
        self._dsn = dsn
        self._catalog_path = catalog_path
        self.user_id = user_id
        self._config_dir = Path(config_dir)

    def _get_backend(self) -> StacCatalogManager | PgStacBackend:
        if self._stac_api_url:
            from fair.stac.pgstac_backend import PgStacBackend

            if not self._dsn:
                raise FairClientError("dsn is required when stac_api_url is set")
            return PgStacBackend(dsn=self._dsn, stac_api_url=self._stac_api_url)
        return StacCatalogManager(self._catalog_path)

    def _resolve_labels(self, dataset: DatasetConfig) -> tuple[str, str]:
        if dataset.labels_pattern:
            label_dir = Path(dataset.train_labels_path)
            matches = sorted(label_dir.glob(dataset.labels_pattern))
            if not matches:
                raise FairClientError(f"No files matching '{dataset.labels_pattern}' in {label_dir}")
            return str(matches[0]), str(matches[0].parent)
        labels_path = Path(dataset.train_labels_path)
        if not labels_path.exists():
            raise FairClientError(f"Labels not found at {dataset.train_labels_path}")
        return str(labels_path), str(labels_path.parent)

    def _pipeline_module_from_item(self, item: pystac.Item) -> str:
        src = item.assets.get("source-code")
        if src is None:
            raise FairClientError(f"Item '{item.id}' has no source-code asset")
        entrypoint: str = src.extra_fields.get("mlm:entrypoint", "")
        module = entrypoint.split(":")[0] if ":" in entrypoint else entrypoint
        if not module:
            raise FairClientError(f"Cannot determine pipeline module from entrypoint '{entrypoint}'")
        return module

    def setup(self) -> None:
        subprocess.run(["zenml", "init"], check=True, capture_output=True)
        Path("artifacts").mkdir(exist_ok=True)
        if not self._stac_api_url:
            initialize_catalog(self._catalog_path)
        print("setup: ok")

    def register_base_model(self, stac_item_path: str) -> str:
        cat = self._get_backend()
        item = pystac.Item.from_file(stac_item_path)
        if errs := validate_mlm_schema(item):
            raise FairClientError(f"MLM schema invalid: {errs}")
        published = cat.publish_item(BASE_MODELS_COLLECTION, item)
        print(f"register: base-model {published.id} v{published.properties['version']}")
        return published.id

    def register_dataset(
        self,
        dataset: DatasetConfig,
        *,
        geojson_dir: str | None = None,
    ) -> str:
        cat = self._get_backend()
        labels_path_str, labels_dir_str = self._resolve_labels(dataset)

        gj_dir = geojson_dir or dataset.geojson_dir
        geojson_files = sorted(Path(gj_dir).glob("*.geojson"))
        if not geojson_files:
            raise FairClientError(f"No .geojson boundary files in {gj_dir}")
        geometry, bbox = geometry_and_bbox_from_geojson(str(geojson_files[0]))

        chip_count = count_chips(dataset.train_chips_path)
        prev = find_previous_active_item(cat, DATASETS_COLLECTION, "title", dataset.title)
        version = str(int(prev.properties["version"]) + 1) if prev else "1"
        predecessor_href = cat.item_href(DATASETS_COLLECTION, prev.id) if prev else None

        Path("artifacts").mkdir(exist_ok=True)
        archive_path = Path("artifacts") / f"{dataset.title}-v{version}.zip"
        create_dataset_archive(
            chips_dir=dataset.train_chips_path,
            labels_dir=labels_dir_str,
            output_path=str(archive_path),
        )

        dataset_item = build_dataset_item(
            dt=datetime.now(UTC),
            label_type=dataset.label_type,
            label_tasks=dataset.label_tasks,
            label_classes=dataset.label_classes,
            keywords=dataset.keywords,
            chips_href=dataset.train_chips_path,
            labels_href=labels_path_str,
            title=dataset.title,
            description=dataset.description,
            user_id=self.user_id,
            chip_count=chip_count if chip_count > 0 else None,
            geometry=geometry,
            bbox=bbox,
            version=version,
            license_id=dataset.license_id,
            providers=dataset.providers,
            label_description=dataset.label_description,
            label_methods=dataset.label_methods,
            source_imagery_href=dataset.source_imagery_href,
            predecessor_version_href=predecessor_href,
            download_href=str(archive_path),
        )

        if prev:
            successor_href = cat.item_href(DATASETS_COLLECTION, dataset_item.id)
            archive_previous_version(cat, DATASETS_COLLECTION, prev, successor_href)

        published = cat.publish_item(DATASETS_COLLECTION, dataset_item)
        print(f"register: dataset {published.id} v{published.properties['version']}")
        return published.id

    def finetune(
        self,
        *,
        base_model_id: str,
        dataset_id: str,
        model_name: str,
        overrides: dict[str, Any] | None = None,
    ) -> str:
        cat = self._get_backend()
        base = cat.get_item(BASE_MODELS_COLLECTION, base_model_id)
        try:
            ds = cat.get_item(DATASETS_COLLECTION, dataset_id)
        except KeyError as exc:
            raise FairClientError(f"Dataset '{dataset_id}' not found. Run register_dataset first.") from exc

        if errs := validate_compatibility(base, ds):
            raise FairClientError(f"Incompatible: {errs}")

        pipeline_module = self._pipeline_module_from_item(base)

        trackers = Client().active_stack_model.components.get(StackComponentType.EXPERIMENT_TRACKER, [])
        tracker_name = trackers[0].name if trackers else None

        cfg_data = generate_training_config(base, ds, model_name, overrides, experiment_tracker=tracker_name)
        self._config_dir.mkdir(parents=True, exist_ok=True)
        train_cfg = self._config_dir / f"train_{model_name}.yaml"
        train_cfg.write_text(yaml.dump(cfg_data, sort_keys=False))

        mod = importlib.import_module(pipeline_module)
        run = mod.training_pipeline.with_options(config_path=str(train_cfg))()
        if run is None:
            raise RuntimeError("Training pipeline returned no run")
        print(f"finetune: {run.id} ({run.status})")
        return model_name

    def promote(
        self,
        finetuned_model_id: str,
        *,
        base_model_id: str | None = None,
        dataset_id: str | None = None,
        description: str = "",
    ) -> str:
        model_name = finetuned_model_id

        client = Client()
        versions = client.list_model_versions(model=model_name, sort_by="desc:created")
        if not versions:
            raise FairClientError(f"No versions for '{model_name}'. Run finetune first.")
        latest = versions[0]
        version_number = latest.number

        resolved_base_id = base_model_id
        resolved_dataset_id = dataset_id
        if resolved_base_id is None or resolved_dataset_id is None:
            run_links = client.list_model_version_pipeline_run_links(model_version_id=latest.id)
            if run_links.items:
                run = run_links.items[0].pipeline_run
                step = run.steps.get("train_model")
                params: dict[str, Any] = (step.config.parameters if step else run.config.parameters) or {}
                if resolved_base_id is None:
                    resolved_base_id = params.get("base_model_id")
                if resolved_dataset_id is None:
                    resolved_dataset_id = params.get("dataset_id")

        if not resolved_base_id:
            raise FairClientError("Cannot resolve base_model_id. Pass it explicitly to promote().")
        if not resolved_dataset_id:
            raise FairClientError("Cannot resolve dataset_id. Pass it explicitly to promote().")

        promote_model_version(model_name, version_number)

        cat = self._get_backend()
        item = publish_promoted_model(
            model_name=model_name,
            version=version_number,
            catalog_manager=cat,
            base_model_item_id=resolved_base_id,
            dataset_item_id=resolved_dataset_id,
            user_id=self.user_id,
            description=description or f"{model_name} finetuned on {resolved_dataset_id}",
        )
        print(f"promote: {item.id}")
        return item.id

    def predict(self, local_model_id: str, image_path: str) -> None:
        cat = self._get_backend()
        try:
            model_item = cat.get_item(LOCAL_MODELS_COLLECTION, local_model_id)
        except KeyError as exc:
            raise FairClientError(f"Local model '{local_model_id}' not found. Run promote() first.") from exc

        pipeline_module = self._pipeline_module_from_item(model_item)

        cfg_data = generate_inference_config(model_item, image_path)
        self._config_dir.mkdir(parents=True, exist_ok=True)
        inf_cfg = self._config_dir / f"inference_{local_model_id[:8]}.yaml"
        inf_cfg.write_text(yaml.dump(cfg_data, sort_keys=False))

        mod = importlib.import_module(pipeline_module)
        run = mod.inference_pipeline.with_options(config_path=str(inf_cfg), enable_cache=False)()
        if run is None:
            raise RuntimeError("Inference pipeline returned no run")
        print(f"predict: {run.id} ({run.status})")
