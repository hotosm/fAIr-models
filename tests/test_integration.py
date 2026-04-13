"""Integration tests for dataset version chain management.

Verifies the full lifecycle: publish v1, archive it, publish v2,
then verify the link chain integrity and archived item state.
"""

from __future__ import annotations

import pystac
import pytest

from fair.stac.builders import build_dataset_item
from fair.stac.catalog_manager import StacCatalogManager
from fair.stac.collections import initialize_catalog
from fair.stac.constants import DATASETS_COLLECTION
from fair.stac.versioning import archive_previous_version, find_previous_active_item

_GEOM = {
    "type": "Polygon",
    "coordinates": [[[85.51, 27.63], [85.53, 27.63], [85.53, 27.64], [85.51, 27.64], [85.51, 27.63]]],
}
_BBOX = [85.51, 27.63, 85.53, 27.64]


def _make_dataset(title: str, version: str = "1") -> pystac.Item:
    return build_dataset_item(
        label_type="vector",
        label_tasks=["segmentation"],
        label_classes=[],
        keywords=["building"],
        chips_href="chips/",
        labels_href="labels.geojson",
        title=title,
        description="Test dataset",
        user_id="test",
        providers=[{"name": "test", "roles": ["producer"]}],
        geometry=_GEOM,
        bbox=_BBOX,
        version=version,
    )


@pytest.fixture()
def cm(tmp_path) -> StacCatalogManager:
    path = str(tmp_path / "catalog.json")
    initialize_catalog(path)
    return StacCatalogManager(path)


class TestVersionChainIntegrity:
    def test_archive_creates_versioned_copy(self, cm: StacCatalogManager) -> None:
        v1 = _make_dataset("my-dataset", version="1")
        cm.publish_item(DATASETS_COLLECTION, v1)

        old = cm.get_item(DATASETS_COLLECTION, v1.id)
        successor_href = cm.item_href(DATASETS_COLLECTION, "my-dataset-v2")

        archived = archive_previous_version(cm, DATASETS_COLLECTION, old, successor_href)

        assert archived.id == f"{v1.id}-v1"
        assert archived.properties["deprecated"] is True

    def test_archived_item_has_successor_link(self, cm: StacCatalogManager) -> None:
        v1 = _make_dataset("my-dataset", version="1")
        cm.publish_item(DATASETS_COLLECTION, v1)

        old = cm.get_item(DATASETS_COLLECTION, v1.id)
        successor_href = cm.item_href(DATASETS_COLLECTION, "my-dataset-v2")
        archived = archive_previous_version(cm, DATASETS_COLLECTION, old, successor_href)

        successor_links = [lnk for lnk in archived.links if lnk.rel == "successor-version"]
        assert len(successor_links) == 1
        assert successor_links[0].get_href() == successor_href

    def test_archived_item_has_no_latest_link(self, cm: StacCatalogManager) -> None:
        v1 = _make_dataset("my-dataset", version="1")
        cm.publish_item(DATASETS_COLLECTION, v1)

        old = cm.get_item(DATASETS_COLLECTION, v1.id)
        successor_href = cm.item_href(DATASETS_COLLECTION, "my-dataset-v2")
        archived = archive_previous_version(cm, DATASETS_COLLECTION, old, successor_href)

        latest_links = [lnk for lnk in archived.links if lnk.rel == "latest-version"]
        assert len(latest_links) == 0

    def test_full_v1_v2_v3_chain(self, cm: StacCatalogManager) -> None:
        title = "chain-test"

        v1 = _make_dataset(title, version="1")
        cm.publish_item(DATASETS_COLLECTION, v1)

        v1_item = cm.get_item(DATASETS_COLLECTION, v1.id)
        v2 = _make_dataset(title, version="2")
        v2_href = cm.item_href(DATASETS_COLLECTION, v2.id)
        archive_previous_version(cm, DATASETS_COLLECTION, v1_item, v2_href)
        cm.publish_item(DATASETS_COLLECTION, v2)

        v2_item = cm.get_item(DATASETS_COLLECTION, v2.id)
        v3 = _make_dataset(title, version="3")
        v3_href = cm.item_href(DATASETS_COLLECTION, v3.id)
        archive_previous_version(cm, DATASETS_COLLECTION, v2_item, v3_href)
        cm.publish_item(DATASETS_COLLECTION, v3)

        archived_v1 = cm.get_item(DATASETS_COLLECTION, f"{v1.id}-v1")
        assert archived_v1.properties["deprecated"] is True
        v1_successors = [lnk for lnk in archived_v1.links if lnk.rel == "successor-version"]
        assert len(v1_successors) == 1

        archived_v2 = cm.get_item(DATASETS_COLLECTION, f"{v2.id}-v2")
        assert archived_v2.properties["deprecated"] is True
        v2_successors = [lnk for lnk in archived_v2.links if lnk.rel == "successor-version"]
        assert len(v2_successors) == 1

        active = find_previous_active_item(cm, DATASETS_COLLECTION, "title", title)
        assert active is not None
        assert active.properties["version"] == "3"

    def test_find_previous_skips_archived_items(self, cm: StacCatalogManager) -> None:
        title = "skip-test"

        v1 = _make_dataset(title, version="1")
        cm.publish_item(DATASETS_COLLECTION, v1)

        v1_item = cm.get_item(DATASETS_COLLECTION, v1.id)
        v2 = _make_dataset(title, version="2")
        v2_href = cm.item_href(DATASETS_COLLECTION, v2.id)
        archive_previous_version(cm, DATASETS_COLLECTION, v1_item, v2_href)
        cm.publish_item(DATASETS_COLLECTION, v2)

        active = find_previous_active_item(cm, DATASETS_COLLECTION, "title", title)
        assert active is not None
        assert active.properties["version"] == "2"
        assert active.properties.get("deprecated") is not True
