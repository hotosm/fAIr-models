from __future__ import annotations

from typing import Protocol

import pystac


class StacBackend(Protocol):
    """Structural interface for STAC catalog operations.

    StacCatalogManager (local JSON) and PgStacBackend (pgstac) both
    conform without explicit inheritance.
    """

    def publish_item(self, collection_id: str, item: pystac.Item) -> pystac.Item: ...

    def get_item(self, collection_id: str, item_id: str) -> pystac.Item: ...

    def list_items(self, collection_id: str) -> list[pystac.Item]: ...

    def deprecate_item(self, collection_id: str, item_id: str) -> pystac.Item: ...

    def delete_item(self, collection_id: str, item_id: str) -> None: ...
