from __future__ import annotations

from datetime import UTC, datetime

import pystac
import pytest

from fair_models.stac.catalog_manager import StacCatalogManager
from fair_models.stac.collections import initialize_catalog


@pytest.fixture()
def cm(tmp_path):
    path = str(tmp_path / "catalog.json")
    initialize_catalog(path)
    return StacCatalogManager(path)


def _item(item_id: str = "test") -> pystac.Item:
    return pystac.Item(
        id=item_id,
        geometry={"type": "Point", "coordinates": [0, 0]},
        bbox=[-1, -1, 1, 1],
        datetime=datetime(2024, 6, 1, tzinfo=UTC),
        properties={"keywords": ["building"]},
    )


def test_publish_get_list(cm):
    cm.publish_item("base-models", _item("a"))
    cm.publish_item("base-models", _item("b"))

    assert cm.get_item("base-models", "a").properties["version"] == "1"
    assert len(cm.list_items("base-models")) == 2


def test_get_missing_raises(cm):
    with pytest.raises(KeyError, match="not found"):
        cm.get_item("base-models", "nope")


def test_version_bump_on_republish(cm):
    cm.publish_item("base-models", _item("x"))
    result = cm.publish_item("base-models", _item("x"))
    assert result.properties["version"] == "2"


def test_deprecate_persists(cm):
    cm.publish_item("local-models", _item("lm"))
    cm.deprecate_item("local-models", "lm")

    # Reload from disk to verify persistence
    fresh = StacCatalogManager(cm.catalog.self_href)
    assert fresh.get_item("local-models", "lm").extra_fields["deprecated"] is True


def test_delete(cm):
    cm.publish_item("base-models", _item("del"))
    cm.delete_item("base-models", "del")
    with pytest.raises(KeyError):
        cm.get_item("base-models", "del")


def test_delete_missing_raises(cm):
    with pytest.raises(KeyError, match="not found"):
        cm.delete_item("base-models", "nope")


def test_initialize_catalog_idempotent(tmp_path):
    path = str(tmp_path / "catalog.json")
    cat1 = initialize_catalog(path)
    cat2 = initialize_catalog(path)
    assert cat2.id == cat1.id
    assert len(list(cat2.get_children())) == 3


def test_invalid_collection_raises(cm):
    with pytest.raises(KeyError, match="not found"):
        cm._get_collection("nonexistent")
