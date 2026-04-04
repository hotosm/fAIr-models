from __future__ import annotations

import logging
import os

import pystac
from pystac import CatalogType
from upath import UPath

from fair.stac.versioning import ensure_version_links

log = logging.getLogger(__name__)


class StacCatalogManager:
    """CRUD on a pystac Catalog persisted as local JSON."""

    def __init__(self, catalog_path: str):
        self.catalog = pystac.Catalog.from_file(catalog_path)

    def _get_collection(self, collection_id: str) -> pystac.Collection:
        child = self.catalog.get_child(collection_id)
        if child is None or not isinstance(child, pystac.Collection):
            msg = f"Collection '{collection_id}' not found in catalog"
            raise KeyError(msg)
        return child

    def _save(self) -> None:
        # Manual self_hrefs: normalize_hrefs() breaks cross-reference links
        # (derived_from, version) that use item IDs as targets.
        root = os.path.dirname(self.catalog.self_href)
        for child in self.catalog.get_children():
            child_dir = os.path.join(root, child.id)
            child.set_self_href(os.path.join(child_dir, "collection.json"))
            for item in child.get_items():
                item.set_self_href(os.path.join(child_dir, item.id, f"{item.id}.json"))
        self.catalog.save(catalog_type=CatalogType.SELF_CONTAINED)

    def publish_item(self, collection_id: str, item: pystac.Item) -> pystac.Item:
        collection = self._get_collection(collection_id)
        existing = collection.get_item(item.id)

        if existing is not None:
            old_version = int(existing.properties.get("version", "1"))
            item.properties["version"] = str(old_version + 1)
            collection.remove_item(item.id)
        else:
            item.properties.setdefault("version", "1")

        self._make_asset_hrefs_absolute(item)
        self._ensure_version_links(collection_id, item)
        collection.add_item(item)
        self._save()
        log.info("Published %s/%s v%s", collection_id, item.id, item.properties.get("version"))
        return item

    def _ensure_version_links(self, collection_id: str, item: pystac.Item) -> None:
        ensure_version_links(item, self.item_href(collection_id, item.id))

    def get_item(self, collection_id: str, item_id: str) -> pystac.Item:
        collection = self._get_collection(collection_id)
        item = collection.get_item(item_id)
        if item is None:
            msg = f"Item '{item_id}' not found in collection '{collection_id}'"
            raise KeyError(msg)
        # SELF_CONTAINED save makes asset hrefs relative to the item JSON;
        # restore absolute paths so consumers get usable file-system paths.
        item.make_asset_hrefs_absolute()
        return item

    def item_exists(self, collection_id: str, item_id: str) -> bool:
        collection = self._get_collection(collection_id)
        return collection.get_item(item_id) is not None

    def list_items(self, collection_id: str, *, limit: int | None = None) -> list[pystac.Item]:
        items = list(self._get_collection(collection_id).get_items())
        return items[:limit] if limit is not None else items

    def deprecate_item(self, collection_id: str, item_id: str) -> pystac.Item:
        item = self.get_item(collection_id, item_id)
        item.properties["deprecated"] = True
        self._save()
        log.info("Deprecated %s/%s", collection_id, item_id)
        return item

    def delete_item(self, collection_id: str, item_id: str) -> None:
        collection = self._get_collection(collection_id)
        if collection.get_item(item_id) is None:
            msg = f"Item '{item_id}' not found in collection '{collection_id}'"
            raise KeyError(msg)
        collection.remove_item(item_id)
        self._save()
        log.info("Deleted %s/%s", collection_id, item_id)

    @staticmethod
    def _make_asset_hrefs_absolute(item: pystac.Item) -> None:
        # Resolve local relative hrefs to absolute before save so PySTAC's
        # SELF_CONTAINED roundtrip restores the correct absolute paths on read.
        # Remote hrefs (s3://, https://, etc.) are left untouched.
        for asset in item.assets.values():
            path = UPath(asset.href)
            if not path.protocol and not path.is_absolute():
                asset.href = str(path.resolve())

    def item_href(self, collection_id: str, item_id: str) -> str:
        return f"../../{collection_id}/{item_id}/{item_id}.json"
