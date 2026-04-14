from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

import pystac
import pytest

from fair.stac.catalog_manager import StacCatalogManager
from fair.stac.collections import initialize_catalog


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


def test_list_items_with_limit(cm):
    cm.publish_item("base-models", _item("a"))
    cm.publish_item("base-models", _item("b"))
    cm.publish_item("base-models", _item("c"))
    assert len(cm.list_items("base-models", limit=2)) == 2
    assert len(cm.list_items("base-models", limit=None)) == 3


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
    assert fresh.get_item("local-models", "lm").properties["deprecated"] is True


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


def test_item_href_resolves_from_item_dir(cm):
    """Relative hrefs from item_href must resolve to the correct item JSON."""
    item = _item("abc")
    cm.publish_item("base-models", item)

    href = cm.item_href("base-models", "abc")
    catalog_root = Path(cm.catalog.self_href).parent
    item_dir = catalog_root / "base-models" / "abc"
    resolved = (item_dir / href).resolve()

    expected = (catalog_root / "base-models" / "abc" / "abc.json").resolve()
    assert resolved == expected
    assert resolved.exists()


def test_cross_collection_href_resolves(cm):
    """Relative href from one collection's item must resolve to another collection's item."""
    cm.publish_item("base-models", _item("model-x"))
    cm.publish_item("datasets", _item("data-y"))

    href_to_dataset = cm.item_href("datasets", "data-y")
    catalog_root = Path(cm.catalog.self_href).parent
    model_item_dir = catalog_root / "base-models" / "model-x"
    resolved = (model_item_dir / href_to_dataset).resolve()

    expected = (catalog_root / "datasets" / "data-y" / "data-y.json").resolve()
    assert resolved == expected
    assert resolved.exists()


def test_asset_hrefs_survive_save_roundtrip(cm):
    """Asset hrefs must resolve to the correct absolute path after publish -> get_item."""
    chips_href = "data/sample/train/oam"
    labels_href = "data/sample/train/osm/labels.geojson"
    expected_chips = os.path.abspath(chips_href)
    expected_labels = os.path.abspath(labels_href)

    item = pystac.Item(
        id="ds-roundtrip",
        geometry={"type": "Point", "coordinates": [0, 0]},
        bbox=[-1, -1, 1, 1],
        datetime=datetime(2024, 6, 1, tzinfo=UTC),
        properties={"keywords": ["building"]},
    )
    item.add_asset("chips", pystac.Asset(href=chips_href, media_type="image/tiff", roles=["data"]))
    item.add_asset("labels", pystac.Asset(href=labels_href, media_type="application/geo+json", roles=["labels"]))

    cm.publish_item("datasets", item)
    retrieved = cm.get_item("datasets", "ds-roundtrip")

    assert retrieved.assets["chips"].href == expected_chips
    assert retrieved.assets["labels"].href == expected_labels
