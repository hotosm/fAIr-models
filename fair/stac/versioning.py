from __future__ import annotations

from typing import Any

import pystac

from fair.stac.backend import StacBackend


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
