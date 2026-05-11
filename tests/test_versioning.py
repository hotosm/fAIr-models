from __future__ import annotations

import pystac
import pytest

from fair.stac.builders import build_dataset_item
from fair.stac.catalog_manager import StacCatalogManager
from fair.stac.collections import initialize_catalog
from fair.stac.constants import DATASETS_COLLECTION
from fair.stac.versioning import (
    add_version_links,
    deprecate_and_link_successor,
    find_previous_active_item,
    normalize_version_link_hrefs,
)

_GEOM = {
    "type": "Polygon",
    "coordinates": [[[85.51, 27.63], [85.53, 27.63], [85.53, 27.64], [85.51, 27.64], [85.51, 27.63]]],
}
_BBOX = [85.51, 27.63, 85.53, 27.64]


def _make_dataset(item_id: str, version: str = "1", deprecated: bool = False) -> pystac.Item:
    return build_dataset_item(
        item_id=item_id,
        label_type="vector",
        label_tasks=["segmentation"],
        label_classes=[],
        keywords=["building"],
        chips_href="chips/",
        labels_href="labels.geojson",
        title="Test",
        description="Test dataset.",
        user_id="u",
        providers=[{"name": "u", "roles": ["producer"]}],
        geometry=_GEOM,
        bbox=_BBOX,
        version=version,
        deprecated=deprecated,
    )


@pytest.fixture()
def cm(tmp_path) -> StacCatalogManager:
    path = str(tmp_path / "catalog.json")
    initialize_catalog(path)
    return StacCatalogManager(path)


class TestFindPreviousActiveItem:
    def test_finds_active_item(self, cm: StacCatalogManager) -> None:
        cm.publish_item(DATASETS_COLLECTION, _make_dataset("ds-1"))
        found = find_previous_active_item(cm, DATASETS_COLLECTION, "title", "Test")
        assert found is not None
        assert found.id == "ds-1"

    def test_skips_deprecated(self, cm: StacCatalogManager) -> None:
        cm.publish_item(DATASETS_COLLECTION, _make_dataset("ds-1", deprecated=True))
        found = find_previous_active_item(cm, DATASETS_COLLECTION, "title", "Test")
        assert found is None

    def test_excludes_by_id(self, cm: StacCatalogManager) -> None:
        cm.publish_item(DATASETS_COLLECTION, _make_dataset("ds-1"))
        found = find_previous_active_item(cm, DATASETS_COLLECTION, "title", "Test", exclude_item_id="ds-1")
        assert found is None

    def test_no_match(self, cm: StacCatalogManager) -> None:
        cm.publish_item(DATASETS_COLLECTION, _make_dataset("ds-1"))
        found = find_previous_active_item(cm, DATASETS_COLLECTION, "title", "NonExistent")
        assert found is None


class TestAddVersionLinks:
    def _bare_item(self, item_id: str) -> pystac.Item:
        from datetime import UTC, datetime

        return pystac.Item(item_id, _GEOM, _BBOX, datetime(2024, 1, 1, tzinfo=UTC), {})

    def test_adds_self_and_latest(self) -> None:
        item = self._bare_item("ds-1")
        add_version_links(item, "https://api/collections/datasets/items/ds-1", None)
        self_links = [lnk for lnk in item.links if lnk.rel == "self"]
        latest = [lnk for lnk in item.links if lnk.rel == "latest-version"]
        assert len(self_links) == 1
        assert len(latest) == 1

    def test_adds_predecessor(self) -> None:
        item = self._bare_item("ds-2")
        add_version_links(item, "https://api/ds-2", "https://api/ds-1")
        pred = [lnk for lnk in item.links if lnk.rel == "predecessor-version"]
        assert len(pred) == 1
        assert pred[0].get_href() == "https://api/ds-1"

    def test_no_self_href_skips_self_and_latest(self) -> None:
        item = self._bare_item("ds-1")
        add_version_links(item, None, None)
        latest = [lnk for lnk in item.links if lnk.rel == "latest-version"]
        self_links = [lnk for lnk in item.links if lnk.rel == "self"]
        assert len(latest) == 0
        assert len(self_links) == 0


class TestNormalizeVersionLinkHrefs:
    def _bare_item(self, item_id: str) -> pystac.Item:
        from datetime import UTC, datetime

        return pystac.Item(item_id, _GEOM, _BBOX, datetime(2024, 1, 1, tzinfo=UTC), {})

    def test_rewrites_relative_path_to_absolute_api_url(self) -> None:
        item = self._bare_item("ds-2")
        item.add_link(pystac.Link(rel="predecessor-version", target="../../datasets/ds-1/ds-1.json"))

        def factory(coll: str, item_id: str) -> str:
            return f"https://api/collections/{coll}/items/{item_id}"

        normalize_version_link_hrefs(item, factory, "fallback")
        rewritten = next(lnk for lnk in item.links if lnk.rel == "predecessor-version")
        assert rewritten.target == "https://api/collections/datasets/items/ds-1"

    def test_falls_back_to_default_collection_when_path_too_short(self) -> None:
        item = self._bare_item("ds-2")
        item.add_link(pystac.Link(rel="predecessor-version", target="ds-1.json"))

        normalize_version_link_hrefs(
            item,
            lambda coll, item_id: f"https://api/{coll}/{item_id}",
            "fallback-coll",
        )
        rewritten = next(lnk for lnk in item.links if lnk.rel == "predecessor-version")
        assert rewritten.target == "https://api/fallback-coll/ds-1"

    def test_skips_links_without_json_segment(self) -> None:
        item = self._bare_item("ds-2")
        item.add_link(pystac.Link(rel="predecessor-version", target="../../datasets/ds-1/"))

        normalize_version_link_hrefs(item, lambda _coll, _item_id: "ignored", "fallback")
        link = next(lnk for lnk in item.links if lnk.rel == "predecessor-version")
        assert link.target == "../../datasets/ds-1/"

    def test_skips_absolute_urls_and_unrelated_rels(self) -> None:
        item = self._bare_item("ds-2")
        item.add_link(pystac.Link(rel="predecessor-version", target="https://api/already-absolute/x.json"))
        item.add_link(pystac.Link(rel="related", target="../../datasets/ds-2/ds-2.json"))

        normalize_version_link_hrefs(item, lambda _coll, _item_id: "REWRITTEN", "fallback")
        absolute = next(lnk for lnk in item.links if lnk.rel == "predecessor-version")
        related = next(lnk for lnk in item.links if lnk.rel == "related")
        assert absolute.target == "https://api/already-absolute/x.json"
        assert related.target == "../../datasets/ds-2/ds-2.json"


class TestDeprecateAndLinkSuccessor:
    def test_deprecates_and_adds_links(self, cm: StacCatalogManager) -> None:
        cm.publish_item(DATASETS_COLLECTION, _make_dataset("ds-1"))
        old = cm.get_item(DATASETS_COLLECTION, "ds-1")
        updated = deprecate_and_link_successor(cm, DATASETS_COLLECTION, old, "https://api/ds-2")
        assert updated.properties["deprecated"] is True
        successor = [lnk for lnk in updated.links if lnk.rel == "successor-version"]
        latest = [lnk for lnk in updated.links if lnk.rel == "latest-version"]
        assert len(successor) == 1
        assert len(latest) == 0
        assert successor[0].get_href() == "https://api/ds-2"
