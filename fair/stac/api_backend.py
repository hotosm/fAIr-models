from __future__ import annotations

import logging
from datetime import UTC, datetime

import httpx
import pystac

from fair.stac.collections import (
    create_base_models_collection,
    create_datasets_collection,
    create_local_models_collection,
)
from fair.stac.versioning import ensure_version_links, normalize_version_link_hrefs

log = logging.getLogger(__name__)


class StacApiBackend:
    """STAC backend that talks only to a STAC API URL via the Transactions extension.

    Unlike PgStacBackend this does not need a Postgres DSN. Production fAIr deployments
    expose a hosted STAC API and never give the backend direct DB access.
    """

    def __init__(
        self,
        stac_api_url: str,
        *,
        api_key: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self._stac_api_url = stac_api_url.rstrip("/")
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._http = httpx.Client(timeout=timeout, headers=headers)
        self._bootstrap_collections()

    def _bootstrap_collections(self) -> None:
        for collection in (
            create_base_models_collection(),
            create_local_models_collection(),
            create_datasets_collection(),
        ):
            self._upsert_collection(collection)

    def _upsert_collection(self, collection: pystac.Collection) -> None:
        body = collection.to_dict()
        existing = self._http.get(f"{self._stac_api_url}/collections/{collection.id}")
        if existing.status_code == 200:
            resp = self._http.put(f"{self._stac_api_url}/collections/{collection.id}", json=body)
        elif existing.status_code == 404:
            resp = self._http.post(f"{self._stac_api_url}/collections", json=body)
        else:
            existing.raise_for_status()
            return
        if resp.status_code not in (200, 201, 204):
            resp.raise_for_status()

    def publish_item(self, collection_id: str, item: pystac.Item) -> pystac.Item:
        item.properties.setdefault("version", "1")
        item.properties["updated"] = datetime.now(UTC).isoformat()
        ensure_version_links(item, self.item_href(collection_id, item.id))
        normalize_version_link_hrefs(item, self.item_href, collection_id)

        body = item.to_dict(transform_hrefs=False)
        body["collection"] = collection_id

        if self.item_exists(collection_id, item.id):
            url = f"{self._stac_api_url}/collections/{collection_id}/items/{item.id}"
            resp = self._http.put(url, json=body)
        else:
            url = f"{self._stac_api_url}/collections/{collection_id}/items"
            resp = self._http.post(url, json=body)
        resp.raise_for_status()
        log.info("Published %s/%s v%s", collection_id, item.id, item.properties.get("version"))
        return item

    def get_item(self, collection_id: str, item_id: str) -> pystac.Item:
        url = f"{self._stac_api_url}/collections/{collection_id}/items/{item_id}"
        resp = self._http.get(url)
        if resp.status_code == 404:
            raise KeyError(f"Item '{item_id}' not found in collection '{collection_id}'")
        resp.raise_for_status()
        return pystac.Item.from_dict(resp.json())

    def item_exists(self, collection_id: str, item_id: str) -> bool:
        url = f"{self._stac_api_url}/collections/{collection_id}/items/{item_id}"
        resp = self._http.get(url)
        if resp.status_code == 200:
            return True
        if resp.status_code == 404:
            return False
        resp.raise_for_status()
        return False

    def list_items(self, collection_id: str, *, limit: int | None = None) -> list[pystac.Item]:
        url = f"{self._stac_api_url}/search"
        payload: dict[str, object] = {"collections": [collection_id]}
        if limit is not None:
            payload["limit"] = limit
        resp = self._http.post(url, json=payload)
        resp.raise_for_status()
        features = resp.json().get("features", [])
        return [pystac.Item.from_dict(f) for f in features]

    def deprecate_item(self, collection_id: str, item_id: str) -> pystac.Item:
        item = self.get_item(collection_id, item_id)
        item.properties["deprecated"] = True
        return self.publish_item(collection_id, item)

    def delete_item(self, collection_id: str, item_id: str) -> None:
        url = f"{self._stac_api_url}/collections/{collection_id}/items/{item_id}"
        resp = self._http.delete(url)
        if resp.status_code in (200, 204, 404):
            return
        resp.raise_for_status()

    def item_href(self, collection_id: str, item_id: str) -> str:
        return f"{self._stac_api_url}/collections/{collection_id}/items/{item_id}"
