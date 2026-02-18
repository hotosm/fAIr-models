"""Minimal STAC-based Model Registry for ZenML."""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import pystac
from zenml.model_registries.base_model_registry import (
    BaseModelRegistry,
    RegisteredModel,
    RegistryModelVersion,
)


class STACModelRegistry(BaseModelRegistry):
    """STAC-based model registry using pystac."""

    def __init__(self, catalog_path: str = "./stac_catalog", **kwargs):
        super().__init__(**kwargs)
        self.catalog_path = Path(catalog_path)
        self.catalog_path.mkdir(exist_ok=True)
        self.catalog = self._load_catalog()

    def _load_catalog(self) -> pystac.Catalog:
        """Load or create STAC catalog."""
        catalog_file = self.catalog_path / "catalog.json"
        if catalog_file.exists():
            return cast(pystac.Catalog, pystac.Catalog.from_file(str(catalog_file)))
        catalog = pystac.Catalog(id="fair-models", description="fAIr Model Registry")
        catalog.normalize_and_save(
            str(self.catalog_path), catalog_type=pystac.CatalogType.SELF_CONTAINED
        )
        return catalog

    def _get_collection(self, name: str) -> Optional[pystac.Collection]:
        """Get model collection."""
        child = self.catalog.get_child(name)
        return cast(Optional[pystac.Collection], child) if child else None

    def _get_items(self, name: str) -> List[pystac.Item]:
        """Get all versions (items) for a model."""
        collection = self._get_collection(name)
        return list(collection.get_items()) if collection else []

    def register_model(
        self,
        name: str,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> RegisteredModel:
        """Register a new model."""
        if self._get_collection(name):
            raise ValueError(f"Model {name} already exists")

        collection = pystac.Collection(
            id=name,
            description=description or f"Model: {name}",
            extent=pystac.Extent(
                spatial=pystac.SpatialExtent([[-180.0, -90.0, 180.0, 90.0]]),
                temporal=pystac.TemporalExtent([[datetime.now(), None]]),
            ),
        )
        if metadata:
            collection.extra_fields["metadata"] = metadata

        self.catalog.add_child(collection)
        self.catalog.save(catalog_type=pystac.CatalogType.SELF_CONTAINED)
        return RegisteredModel(
            name=name, description=description, metadata=metadata or {}
        )

    def delete_model(self, name: str) -> None:
        if self._get_collection(name) is None:
            raise KeyError(f"Model {name} not found")

        self.catalog.remove_child(name)
        self.catalog.save(catalog_type=pystac.CatalogType.SELF_CONTAINED)

    def update_model(
        self,
        name: str,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        remove_metadata: Optional[List[str]] = None,
    ) -> RegisteredModel:

        collection = self._get_collection(name)
        if collection is None:
            raise KeyError(f"Model {name} not found")

        if description:
            collection.description = description

        current_metadata: Dict[str, Any] = collection.extra_fields.get("metadata", {})

        if metadata:
            current_metadata.update(metadata)

        if remove_metadata:
            for key in remove_metadata:
                current_metadata.pop(key, None)

        collection.extra_fields["metadata"] = current_metadata
        collection.save_object()

        return RegisteredModel(
            name=name, description=description, metadata=current_metadata
        )

    def get_model(self, name: str) -> RegisteredModel:
        collection = self._get_collection(name)
        if collection is None:
            raise KeyError(f"Model {name} not found")

        metadata: Dict[str, Any] = collection.extra_fields.get("metadata", {})
        return RegisteredModel(
            name=name, description=collection.description, metadata=metadata
        )

    def list_models(
        self,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[RegisteredModel]:

        if name:
            coll = self._get_collection(name)
            collections: List[pystac.Collection] = [coll] if coll else []
        else:
            collections = cast(
                List[pystac.Collection], list(self.catalog.get_children())
            )

        models: List[RegisteredModel] = []

        for col in collections:
            if col is None:
                continue

            col_metadata: Dict[str, Any] = col.extra_fields.get("metadata", {})

            if metadata and not all(
                col_metadata.get(k) == v for k, v in metadata.items()
            ):
                continue

            models.append(
                RegisteredModel(
                    name=col.id, description=col.description, metadata=col_metadata
                )
            )

        return models

    def register_model_version(
        self,
        name: str,
        version: Optional[str] = None,
        model_source_uri: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Any] = None,
        **kwargs: Any,
    ) -> RegistryModelVersion:
        """Register a model version."""
        collection = self._get_collection(name)
        if not collection:
            raise KeyError(f"Model {name} not registered")

        version_num = int(version) if version else len(self._get_items(name)) + 1
        model_path = model_source_uri or str(
            self.catalog_path / f"{name}_v{version_num}.pkl"
        )

        item = pystac.Item(
            id=f"{name}_v{version_num}",
            geometry=None,
            bbox=None,
            datetime=datetime.now(),
            properties={"version": version_num, "description": description or ""},
        )
        item.add_asset("model", pystac.Asset(href=model_path, roles=["data"]))
        collection.add_item(item)
        self.catalog.save(catalog_type=pystac.CatalogType.SELF_CONTAINED)

        return RegistryModelVersion(
            registered_model=RegisteredModel(name=name),
            version=str(version_num),
            created_at=datetime.now(),
            model_source_uri=model_path,
            model_format="pkl",
            metadata=metadata,
        )

    def get_model_version(self, name: str, version: str) -> RegistryModelVersion:
        items: List[pystac.Item] = self._get_items(name)
        item: Optional[pystac.Item] = next(
            (
                i
                for i in items
                if cast(int, i.properties.get("version")) == int(version)
            ),
            None,
        )

        if item is None:
            raise KeyError(f"Model version {name} v{version} not found")

        assets = item.get_assets()
        model_asset: Optional[pystac.Asset] = assets.get("model") if assets else None
        model_uri: str = cast(str, model_asset.href) if model_asset else ""

        return RegistryModelVersion(
            registered_model=RegisteredModel(name=name),
            version=str(item.properties.get("version", "")),
            created_at=datetime.fromisoformat(str(item.datetime)),
            model_source_uri=model_uri,
            model_format="pkl",
            metadata=None,
        )

    def delete_model_version(self, name: str, version: str) -> None:
        """Deletes a model version from the STAC catalog."""
        items: List[pystac.Item] = self._get_items(name)
        item: Optional[pystac.Item] = next(
            (
                i
                for i in items
                if cast(int, i.properties.get("version")) == int(version)
            ),
            None,
        )

        if item is None:
            raise KeyError(f"Model version {name} v{version} not found")

        collection: pystac.Collection = cast(
            pystac.Collection, self.catalog.get_child(name)
        )
        collection.remove_item(item.id)
        self.catalog.save(catalog_type=pystac.CatalogType.SELF_CONTAINED)

    def update_model_version(
        self,
        name: str,
        version: str,
        description: Optional[str] = None,
        metadata: Optional[Any] = None,
        remove_metadata: Optional[List[str]] = None,
        stage: Optional[Any] = None,
    ) -> RegistryModelVersion:
        """Updates a model version in the STAC catalog."""
        item = self.get_model_version(name, version)

        items: List[pystac.Item] = self._get_items(name)
        stac_item: Optional[pystac.Item] = next(
            (
                i
                for i in items
                if cast(int, i.properties.get("version")) == int(version)
            ),
            None,
        )

        if stac_item is None:
            raise KeyError(f"Model version {name} v{version} not found")

        if description:
            stac_item.properties["description"] = description
        if metadata:
            stac_item.properties["metadata"] = metadata
        if remove_metadata:
            for key in remove_metadata:
                stac_item.properties.pop(key, None)
        if stage:
            stac_item.properties["stage"] = str(stage)

        self.catalog.save(catalog_type=pystac.CatalogType.SELF_CONTAINED)
        return item

    def list_model_versions(
        self,
        name: Optional[str] = None,
        model_source_uri: Optional[str] = None,
        metadata: Optional[Any] = None,
        stage: Optional[Any] = None,
        count: Optional[int] = None,
        created_after: Optional[datetime] = None,
        created_before: Optional[datetime] = None,
        order_by_date: Optional[str] = None,
        **kwargs: Any,
    ) -> Optional[List[RegistryModelVersion]]:
        """Lists all model versions for a registered model."""
        if name is None:
            return None

        items: List[pystac.Item] = self._get_items(name)
        versions: List[RegistryModelVersion] = []

        for item in items:
            assets = item.get_assets()
            model_asset: Optional[pystac.Asset] = (
                assets.get("model") if assets else None
            )
            model_uri: str = cast(str, model_asset.href) if model_asset else ""
            item_datetime = datetime.fromisoformat(str(item.datetime))

            if model_source_uri and model_uri != model_source_uri:
                continue
            if created_after and item_datetime < created_after:
                continue
            if created_before and item_datetime > created_before:
                continue

            versions.append(
                RegistryModelVersion(
                    registered_model=RegisteredModel(name=name),
                    version=str(item.properties.get("version", "")),
                    created_at=item_datetime,
                    model_source_uri=model_uri,
                    model_format="pkl",
                    metadata=item.properties.get("metadata"),
                )
            )

        if order_by_date:
            reverse = order_by_date.lower() == "desc"
            versions.sort(key=lambda v: v.created_at, reverse=reverse)

        if count:
            versions = versions[:count]

        return versions

    def load_model_version(self, name: str, version: str, **kwargs: Any) -> Any:
        """Loads a model version from the STAC catalog."""
        import pickle

        item = self.get_model_version(name, version)
        model_uri = item.model_source_uri

        if model_uri is None:
            raise ValueError(f"Model URI not found for {name} v{version}")

        with open(model_uri, "rb") as f:
            return pickle.load(f)

    def get_model_uri_artifact_store(self, model_version: RegistryModelVersion) -> str:
        """Gets the artifact store URI for a model version."""
        if model_version.model_source_uri is None:
            raise ValueError("Model source URI is not set")
        return model_version.model_source_uri
