from __future__ import annotations

import os

import pystac
from pystac import CatalogType


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

        collection.add_item(item)
        self._save()
        return item

    def get_item(self, collection_id: str, item_id: str) -> pystac.Item:
        collection = self._get_collection(collection_id)
        item = collection.get_item(item_id)
        if item is None:
            msg = f"Item '{item_id}' not found in collection '{collection_id}'"
            raise KeyError(msg)
        return item

    def list_items(self, collection_id: str) -> list[pystac.Item]:
        return list(self._get_collection(collection_id).get_items())

    def deprecate_item(self, collection_id: str, item_id: str) -> pystac.Item:
        item = self.get_item(collection_id, item_id)
        item.properties["deprecated"] = True
        self._save()
        return item

    def delete_item(self, collection_id: str, item_id: str) -> None:
        collection = self._get_collection(collection_id)
        if collection.get_item(item_id) is None:
            msg = f"Item '{item_id}' not found in collection '{collection_id}'"
            raise KeyError(msg)
        collection.remove_item(item_id)
        self._save()
