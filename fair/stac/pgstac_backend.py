from __future__ import annotations

import pystac
from pypgstac.db import PgstacDB  # optional dep (k8s group)
from pypgstac.load import Loader, Methods  # optional dep (k8s group)
from pystac_client import Client as StacClient  # optional dep (k8s group)

from fair.stac.collections import (
    create_base_models_collection,
    create_datasets_collection,
    create_local_models_collection,
)


class PgStacBackend:
    """STAC backend using pypgstac Loader and pystac-client.

    Writes use pypgstac's Loader (bulk upsert via COPY + pgstac SQL).
    Reads use pystac-client against the STAC API.
    Delete uses pgstac's delete_item() SQL function directly since
    Loader has no delete API.
    """

    def __init__(self, dsn: str, stac_api_url: str) -> None:
        self._dsn = dsn
        self._stac_api_url = stac_api_url
        self._bootstrap_collections()

    def _get_db(self) -> PgstacDB:
        return PgstacDB(dsn=self._dsn)

    def _bootstrap_collections(self) -> None:
        """Ensure the 3 fAIr collections exist in pgstac (idempotent upsert)."""
        collections = [
            create_base_models_collection(),
            create_local_models_collection(),
            create_datasets_collection(),
        ]
        with self._get_db() as db:
            loader = Loader(db)
            loader.load_collections(
                iter(c.to_dict() for c in collections),
                insert_mode=Methods.upsert,
            )

    def publish_item(self, collection_id: str, item: pystac.Item) -> pystac.Item:
        item_dict = item.to_dict()
        item_dict["collection"] = collection_id
        with self._get_db() as db:
            loader = Loader(db)
            loader.load_items(iter([item_dict]), insert_mode=Methods.upsert)
        return item

    def get_item(self, collection_id: str, item_id: str) -> pystac.Item:
        client = StacClient.open(self._stac_api_url)
        collection = client.get_collection(collection_id)
        item = collection.get_item(item_id)
        if item is None:
            msg = f"Item '{item_id}' not found in collection '{collection_id}'"
            raise KeyError(msg)
        return item

    def list_items(self, collection_id: str) -> list[pystac.Item]:
        client = StacClient.open(self._stac_api_url)
        return list(client.search(collections=[collection_id]).items())

    def deprecate_item(self, collection_id: str, item_id: str) -> pystac.Item:
        item = self.get_item(collection_id, item_id)
        item.properties["deprecated"] = True
        return self.publish_item(collection_id, item)

    def delete_item(self, collection_id: str, item_id: str) -> None:
        # Loader has no delete API; call pgstac's delete_item() directly
        with self._get_db() as db:
            db.query_one("SELECT delete_item(%s, %s)", [item_id, collection_id])
