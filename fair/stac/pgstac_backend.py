from __future__ import annotations

import logging
from datetime import UTC, datetime

import httpx
import pystac
from pypgstac.db import PgstacDB
from pypgstac.load import Loader, Methods

from fair.stac.collections import (
    create_base_models_collection,
    create_datasets_collection,
    create_local_models_collection,
)
from fair.stac.versioning import ensure_version_links, normalize_version_link_hrefs

log = logging.getLogger(__name__)


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
        self._http = httpx.Client(timeout=30)
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
        item.properties.setdefault("version", "1")
        item.properties["updated"] = datetime.now(UTC).isoformat()
        ensure_version_links(item, self.item_href(collection_id, item.id))
        normalize_version_link_hrefs(item, self.item_href, collection_id)
        item_dict = item.to_dict(transform_hrefs=False)
        item_dict["collection"] = collection_id
        with self._get_db() as db:
            loader = Loader(db)
            loader.load_items(iter([item_dict]), insert_mode=Methods.upsert)
        log.info("Published %s/%s v%s", collection_id, item.id, item.properties.get("version"))
        return item

    def get_item(self, collection_id: str, item_id: str) -> pystac.Item:
        url = f"{self._stac_api_url}/collections/{collection_id}/items/{item_id}"
        resp = self._http.get(url)
        if resp.status_code == 404:
            msg = f"Item '{item_id}' not found in collection '{collection_id}'"
            raise KeyError(msg)
        resp.raise_for_status()
        return pystac.Item.from_dict(resp.json())

    def item_exists(self, collection_id: str, item_id: str) -> bool:
        url = f"{self._stac_api_url}/collections/{collection_id}/items/{item_id}"
        resp = self._http.get(url)
        return resp.status_code == 200

    def list_items(self, collection_id: str, *, limit: int | None = None) -> list[pystac.Item]:
        url = f"{self._stac_api_url}/search"
        payload: dict[str, object] = {"collections": [collection_id]}
        if limit is not None:
            payload["limit"] = limit
        resp = self._http.post(url, json=payload)
        resp.raise_for_status()
        features = resp.json().get("features", [])
        return [pystac.Item.from_dict(f) for f in features]

    def patch_item(self, collection_id: str, item_id: str, patch: dict) -> pystac.Item:
        # pgstac's loader has no native PATCH; do a get + merge + upsert
        item = self.get_item(collection_id, item_id)
        for key, value in patch.items():
            if key == "properties" and isinstance(value, dict):
                item.properties.update(value)
            else:
                setattr(item, key, value)
        return self.publish_item(collection_id, item)

    def deprecate_item(self, collection_id: str, item_id: str) -> pystac.Item:
        item = self.get_item(collection_id, item_id)
        item.properties["deprecated"] = True
        return self.publish_item(collection_id, item)

    def delete_item(self, collection_id: str, item_id: str) -> None:
        # Loader has no delete API; call pgstac's delete_item() directly
        with self._get_db() as db:
            db.query_one("SELECT delete_item(%s, %s)", [item_id, collection_id])
        log.info("Deleted %s/%s", collection_id, item_id)

    def item_href(self, collection_id: str, item_id: str) -> str:
        return f"{self._stac_api_url}/collections/{collection_id}/items/{item_id}"
