from __future__ import annotations

from datetime import UTC, datetime

import pystac
import pytest

from fair_models.stac.catalog_manager import StacCatalogManager
from fair_models.stac.collections import initialize_catalog


@pytest.fixture()
def catalog_manager(tmp_path):
    catalog_path = str(tmp_path / "catalog.json")
    initialize_catalog(catalog_path)
    return StacCatalogManager(catalog_path)


def _make_item(item_id: str = "test-item") -> pystac.Item:
    return pystac.Item(
        id=item_id,
        geometry={"type": "Point", "coordinates": [0, 0]},
        bbox=[-1, -1, 1, 1],
        datetime=datetime(2024, 6, 1, tzinfo=UTC),
        properties={"keywords": ["building"]},
    )


class TestPublishAndGet:
    def test_publish_new_item_and_retrieve(self, catalog_manager):
        item = _make_item("model-a")
        result = catalog_manager.publish_item("base-models", item)

        assert result.properties["version"] == "1"

        retrieved = catalog_manager.get_item("base-models", "model-a")
        assert retrieved.id == "model-a"
        assert retrieved.properties["version"] == "1"

    def test_get_missing_item_raises(self, catalog_manager):
        with pytest.raises(KeyError, match="not found"):
            catalog_manager.get_item("base-models", "nonexistent")


class TestListItems:
    def test_list_items_returns_all(self, catalog_manager):
        catalog_manager.publish_item("datasets", _make_item("ds-1"))
        catalog_manager.publish_item("datasets", _make_item("ds-2"))

        items = catalog_manager.list_items("datasets")
        assert len(items) == 2
        ids = {i.id for i in items}
        assert ids == {"ds-1", "ds-2"}

    def test_list_empty_collection(self, catalog_manager):
        items = catalog_manager.list_items("base-models")
        assert items == []


class TestDeprecateItem:
    def test_deprecate_sets_flag(self, catalog_manager):
        catalog_manager.publish_item("local-models", _make_item("lm-1"))

        result = catalog_manager.deprecate_item("local-models", "lm-1")
        assert result.extra_fields.get("deprecated") is True

        # Persisted after reload
        fresh = StacCatalogManager(catalog_manager.catalog.self_href)
        reloaded = fresh.get_item("local-models", "lm-1")
        assert reloaded.extra_fields.get("deprecated") is True


class TestDeleteItem:
    def test_delete_removes_item(self, catalog_manager):
        catalog_manager.publish_item("base-models", _make_item("to-delete"))
        catalog_manager.delete_item("base-models", "to-delete")

        with pytest.raises(KeyError, match="not found"):
            catalog_manager.get_item("base-models", "to-delete")

    def test_delete_missing_raises(self, catalog_manager):
        with pytest.raises(KeyError, match="not found"):
            catalog_manager.delete_item("base-models", "nonexistent")


class TestPublishExistingBumpsVersion:
    def test_version_increments_on_republish(self, catalog_manager):
        item_v1 = _make_item("same-id")
        catalog_manager.publish_item("base-models", item_v1)

        item_v2 = _make_item("same-id")
        item_v2.properties["keywords"] = ["road"]
        result = catalog_manager.publish_item("base-models", item_v2)

        assert result.properties["version"] == "2"
        assert result.properties["keywords"] == ["road"]

        retrieved = catalog_manager.get_item("base-models", "same-id")
        assert retrieved.properties["version"] == "2"
