from __future__ import annotations

import pystac
from pystac import CatalogType
from pystac.catalog import Catalog
from pystac.collection import Collection
from pystac.item import Item


class StacCatalogManager:
    """CRUD operations on a pystac Catalog (local JSON files)."""

    def __init__(self, catalog_path: str):
        self.catalog: Catalog = pystac.Catalog.from_file(catalog_path)

    def _get_collection(self, collection_id: str) -> pystac.Collection:
        collection: Catalog | None = self.catalog.get_child(collection_id)
        if collection is None or not isinstance(collection, pystac.Collection):
            msg = f"Collection '{collection_id}' not found in catalog"
            raise KeyError(msg)
        return collection

    def _save(self) -> None:
        self.catalog.normalize_hrefs(self.catalog.self_href.rsplit("/", 1)[0])
        self.catalog.save(catalog_type=CatalogType.SELF_CONTAINED)

    def publish_item(self, collection_id: str, item: pystac.Item) -> pystac.Item:
        """Add or update item in collection. Saves catalog. Returns item.

        If item ID already exists: replaces it and bumps the Version Extension
        'version' property. If new: inserts with version '1'.
        """
        collection: Collection = self._get_collection(collection_id)
        existing: Item | None = collection.get_item(item.id)

        if existing is not None:
            # Bump version on the incoming item
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
        collection: Collection = self._get_collection(collection_id)
        item: Item | None = collection.get_item(item_id)
        if item is None:
            msg = f"Item '{item_id}' not found in collection '{collection_id}'"
            raise KeyError(msg)
        return item

    def list_items(self, collection_id: str) -> list[pystac.Item]:
        """List all items in a collection."""
        collection: Collection = self._get_collection(collection_id)
        return list(collection.get_items())

    def deprecate_item(self, collection_id: str, item_id: str) -> pystac.Item:
        """Set deprecated=True on item. Returns updated item."""
        item: Item = self.get_item(collection_id, item_id)
        item.extra_fields["deprecated"] = True
        self._save()
        return item

    def delete_item(self, collection_id: str, item_id: str) -> None:
        """Remove item from collection. Saves catalog."""
        collection: Collection = self._get_collection(collection_id)
        item: Item | None = collection.get_item(item_id)
        if item is None:
            msg = f"Item '{item_id}' not found in collection '{collection_id}'"
            raise KeyError(msg)
        collection.remove_item(item_id)
        self._save()
