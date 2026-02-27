"""Protocol conformance: both backends satisfy StacBackend."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import pystac

from fair.stac.catalog_manager import StacCatalogManager


@runtime_checkable
class _CheckableBackend(Protocol):
    """Mirror of StacBackend, but runtime_checkable for tests."""

    def publish_item(self, collection_id: str, item: pystac.Item) -> pystac.Item: ...

    def get_item(self, collection_id: str, item_id: str) -> pystac.Item: ...

    def list_items(self, collection_id: str) -> list[pystac.Item]: ...

    def deprecate_item(self, collection_id: str, item_id: str) -> pystac.Item: ...

    def delete_item(self, collection_id: str, item_id: str) -> None: ...


class _Dummy:
    """Minimal concrete class implementing all 5 Protocol methods."""

    def publish_item(self, collection_id: str, item: pystac.Item) -> pystac.Item:
        return item

    def get_item(self, collection_id: str, item_id: str) -> pystac.Item:
        return pystac.Item(item_id, geometry=None, bbox=None, datetime=None, properties={})

    def list_items(self, collection_id: str) -> list[pystac.Item]:
        return []

    def deprecate_item(self, collection_id: str, item_id: str) -> pystac.Item:
        return pystac.Item(item_id, geometry=None, bbox=None, datetime=None, properties={"deprecated": True})

    def delete_item(self, collection_id: str, item_id: str) -> None:
        return None


def test_dummy_satisfies_protocol():
    assert isinstance(_Dummy(), _CheckableBackend)


def test_catalog_manager_satisfies_protocol():
    """StacCatalogManager structurally matches StacBackend."""
    required = {"publish_item", "get_item", "list_items", "deprecate_item", "delete_item"}
    assert required.issubset(set(dir(StacCatalogManager)))


def test_missing_method_fails_protocol():
    class _Incomplete:
        def publish_item(self, collection_id: str, item: pystac.Item) -> pystac.Item:
            return item

    assert not isinstance(_Incomplete(), _CheckableBackend)
