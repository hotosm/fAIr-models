from __future__ import annotations

import logging
from typing import Any

import pystac

from fair.stac.backend import StacBackend

log = logging.getLogger(__name__)


def find_previous_active_item(
    backend: StacBackend,
    collection_id: str,
    match_field: str,
    match_value: Any,
    exclude_item_id: str | None = None,
) -> pystac.Item | None:
    for item in backend.list_items(collection_id):
        if item.properties.get(match_field) != match_value:
            continue
        if item.properties.get("deprecated"):
            continue
        if exclude_item_id and item.id == exclude_item_id:
            continue
        return item
    return None


def ensure_version_links(item: pystac.Item, self_href: str) -> None:
    if not any(lnk.rel == "self" for lnk in item.links):
        item.add_link(pystac.Link(rel="self", target=self_href, media_type="application/geo+json"))
    if not item.properties.get("deprecated") and not any(lnk.rel == "latest-version" for lnk in item.links):
        item.add_link(pystac.Link(rel="latest-version", target=self_href))


def add_version_links(
    item: pystac.Item,
    self_href: str | None,
    predecessor_href: str | None,
) -> None:
    if self_href:
        item.add_link(pystac.Link(rel="self", target=self_href, media_type="application/geo+json"))
        item.add_link(pystac.Link(rel="latest-version", target=self_href))
    if predecessor_href:
        item.add_link(pystac.Link(rel="predecessor-version", target=predecessor_href))


def deprecate_and_link_successor(
    backend: StacBackend,
    collection_id: str,
    old_item: pystac.Item,
    new_item_href: str,
) -> pystac.Item:
    old_item = backend.deprecate_item(collection_id, old_item.id)
    old_item.links = [lnk for lnk in old_item.links if lnk.rel != "latest-version"]
    old_item.add_link(pystac.Link(rel="successor-version", target=new_item_href))
    return backend.publish_item(collection_id, old_item)


def archive_previous_version(
    backend: StacBackend,
    collection_id: str,
    old_item: pystac.Item,
    new_item_href: str,
) -> pystac.Item:
    """Copy the old item to a version-suffixed ID and deprecate it.

    Preserves the old version as a discoverable, deprecated STAC item
    per the STAC Version Extension v1.2.0 spec.
    """
    old_version = old_item.properties.get("version", "1")
    archived_id = f"{old_item.id}-v{old_version}"

    archived = old_item.clone()
    archived.id = archived_id
    archived.properties["deprecated"] = True
    archived.links = [lnk for lnk in archived.links if lnk.rel not in {"latest-version", "self"}]
    archived.add_link(pystac.Link(rel="successor-version", target=new_item_href))

    archived_href = backend.item_href(collection_id, archived_id)
    archived.add_link(pystac.Link(rel="self", target=archived_href, media_type="application/geo+json"))

    published = backend.publish_item(collection_id, archived)
    log.info("Archived %s/%s as deprecated", collection_id, archived_id)
    return published
