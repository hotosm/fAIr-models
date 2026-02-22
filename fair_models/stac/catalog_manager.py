from __future__ import annotations

import os

import pystac
from pystac import CatalogType


class StacCatalogManager:
    """CRUD operations on a pystac Catalog (local JSON files)."""

    def __init__(self, catalog_path: str):
        self.catalog = pystac.Catalog.from_file(catalog_path)

    def _get_collection(self, collection_id: str) -> pystac.Collection:
        child = self.catalog.get_child(collection_id)
        if child is None or not isinstance(child, pystac.Collection):
            msg = f"Collection '{collection_id}' not found in catalog"
            raise KeyError(msg)
        return child

    def _save(self) -> None:
        """Set self_hrefs manually and persist as SELF_CONTAINED JSON.

        Avoids normalize_hrefs() which chokes on cross-reference links
        (derived_from, version links) that use item IDs as targets.
        """
        root = os.path.dirname(self.catalog.self_href)
        for child in self.catalog.get_children():
            child_dir = os.path.join(root, child.id)
            child.set_self_href(os.path.join(child_dir, "collection.json"))
            for item in child.get_items():
                item.set_self_href(os.path.join(child_dir, item.id, f"{item.id}.json"))
        self.catalog.save(catalog_type=CatalogType.SELF_CONTAINED)

    def publish_item(self, collection_id: str, item: pystac.Item) -> pystac.Item:
        """Add or update item in collection. Saves catalog. Returns item.

        If item ID already exists: replaces it and bumps the Version Extension
        'version' property. If new: inserts with version '1'.
        """
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
        """Retrieve item by ID. Raises KeyError if not found."""
        collection = self._get_collection(collection_id)
        item = collection.get_item(item_id)
        if item is None:
            msg = f"Item '{item_id}' not found in collection '{collection_id}'"
            raise KeyError(msg)
        return item

    def list_items(self, collection_id: str) -> list[pystac.Item]:
        """List all items in a collection."""
        return list(self._get_collection(collection_id).get_items())

    def deprecate_item(self, collection_id: str, item_id: str) -> pystac.Item:
        """Set deprecated=True on item. Returns updated item."""
        item = self.get_item(collection_id, item_id)
        item.properties["deprecated"] = True
        self._save()
        return item

    def delete_item(self, collection_id: str, item_id: str) -> None:
        """Remove item from collection. Saves catalog."""
        collection = self._get_collection(collection_id)
        if collection.get_item(item_id) is None:
            msg = f"Item '{item_id}' not found in collection '{collection_id}'"
            raise KeyError(msg)
        collection.remove_item(item_id)
        self._save()
