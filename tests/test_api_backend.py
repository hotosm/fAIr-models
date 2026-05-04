from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import httpx
import pystac
import pytest

from fair.stac.api_backend import StacApiBackend


def _routes_to(handler):
    transport = httpx.MockTransport(handler)
    return transport


def _build_dummy_item(item_id: str = "ds-1") -> pystac.Item:
    return pystac.Item(
        id=item_id,
        geometry={
            "type": "Polygon",
            "coordinates": [[[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]]],
        },
        bbox=[-1, -1, 1, 1],
        datetime=datetime.now(UTC),
        properties={"version": "1"},
    )


def _make_backend(handler) -> StacApiBackend:
    backend = StacApiBackend.__new__(StacApiBackend)
    backend._stac_api_url = "https://stac.example/api"
    backend._http = httpx.Client(transport=_routes_to(handler))
    return backend


def test_publish_item_posts_when_missing() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path.endswith("/items/ds-1"):
            return httpx.Response(404)
        if request.method == "POST" and request.url.path.endswith("/collections/datasets/items"):
            seen["body"] = request.content
            return httpx.Response(201, json={})
        return httpx.Response(500)

    backend = _make_backend(handler)
    item = _build_dummy_item()
    published = backend.publish_item("datasets", item)

    assert published.id == "ds-1"
    assert b'"id":"ds-1"' in seen["body"]
    assert b'"collection":"datasets"' in seen["body"]


def test_publish_item_puts_when_present() -> None:
    method_seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        method_seen.append(request.method)
        if request.method == "GET" and request.url.path.endswith("/items/ds-1"):
            return httpx.Response(200, json={"id": "ds-1"})
        if request.method == "PUT" and request.url.path.endswith("/items/ds-1"):
            return httpx.Response(200, json={})
        return httpx.Response(500)

    backend = _make_backend(handler)
    backend.publish_item("datasets", _build_dummy_item())
    assert "PUT" in method_seen


def test_get_item_404_raises_keyerror() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    backend = _make_backend(handler)
    with pytest.raises(KeyError, match="not found"):
        backend.get_item("datasets", "missing")


def test_item_exists_returns_true_for_200() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"id": "ds-1"})

    backend = _make_backend(handler)
    assert backend.item_exists("datasets", "ds-1") is True


def test_item_exists_returns_false_for_404() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    backend = _make_backend(handler)
    assert backend.item_exists("datasets", "ds-1") is False


def test_list_items_returns_features() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path.endswith("/search")
        return httpx.Response(
            200,
            json={
                "features": [
                    _build_dummy_item("ds-1").to_dict(),
                    _build_dummy_item("ds-2").to_dict(),
                ]
            },
        )

    backend = _make_backend(handler)
    items = backend.list_items("datasets", limit=10)
    assert [i.id for i in items] == ["ds-1", "ds-2"]


def test_delete_item_tolerates_404() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    backend = _make_backend(handler)
    backend.delete_item("datasets", "missing")


def test_item_href_format() -> None:
    backend = _make_backend(lambda _: httpx.Response(200, json={}))
    href = backend.item_href("datasets", "ds-1")
    assert href == "https://stac.example/api/collections/datasets/items/ds-1"


def test_api_key_attached_as_bearer() -> None:
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization", "")
        return httpx.Response(404)

    backend = StacApiBackend.__new__(StacApiBackend)
    backend._stac_api_url = "https://stac.example/api"
    backend._http = httpx.Client(
        transport=_routes_to(handler),
        headers={"Authorization": "Bearer secret-token-123"},
    )
    backend.item_exists("datasets", "ds-1")
    assert seen["auth"] == "Bearer secret-token-123"


def test_init_bootstraps_collections_via_post_and_put() -> None:
    seen: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, request.url.path))
        if request.method == "GET" and "/collections/" in request.url.path:
            collection_id = request.url.path.rsplit("/", 1)[-1]
            if collection_id == "datasets":
                return httpx.Response(200, json={"id": collection_id})
            return httpx.Response(404)
        if request.method == "POST" and request.url.path.endswith("/collections"):
            return httpx.Response(201, json={})
        if request.method == "PUT" and "/collections/" in request.url.path:
            return httpx.Response(200, json={})
        return httpx.Response(500)

    transport = httpx.MockTransport(handler)
    backend = StacApiBackend.__new__(StacApiBackend)
    backend._stac_api_url = "https://stac.example/api"
    backend._http = httpx.Client(transport=transport)
    backend._bootstrap_collections()

    methods = {(method, path.split("/")[-1]) for method, path in seen}
    assert ("PUT", "datasets") in methods
    assert ("POST", "collections") in methods


def test_init_attaches_bearer_token_via_constructor(monkeypatch: pytest.MonkeyPatch) -> None:
    auth_seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        auth_seen.setdefault("auth", request.headers.get("authorization", ""))
        if request.method == "GET":
            return httpx.Response(200, json={"id": "x"})
        return httpx.Response(200, json={})

    real_client = httpx.Client
    transport = httpx.MockTransport(handler)

    def fake_client(*, timeout: float, headers: dict[str, str]) -> httpx.Client:
        return real_client(timeout=timeout, headers=headers, transport=transport)

    monkeypatch.setattr("fair.stac.api_backend.httpx.Client", fake_client)
    StacApiBackend("https://stac.example/api/", api_key="tkn-xyz")

    assert auth_seen["auth"] == "Bearer tkn-xyz"


def test_upsert_collection_propagates_unexpected_get_status() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    backend = _make_backend(handler)
    collection = pystac.Collection(
        id="datasets",
        description="d",
        extent=pystac.Extent(
            pystac.SpatialExtent([[-1, -1, 1, 1]]),
            pystac.TemporalExtent([[datetime.now(UTC), None]]),
        ),
    )
    with pytest.raises(httpx.HTTPStatusError):
        backend._upsert_collection(collection)


def test_item_exists_raises_on_unexpected_status() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    backend = _make_backend(handler)
    with pytest.raises(httpx.HTTPStatusError):
        backend.item_exists("datasets", "ds-1")


def test_delete_item_raises_on_unexpected_status() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    backend = _make_backend(handler)
    with pytest.raises(httpx.HTTPStatusError):
        backend.delete_item("datasets", "ds-1")


def test_deprecate_item_sets_flag_and_republishes() -> None:
    deprecated = [False]
    calls: list[tuple[str, str]] = []
    item_payload = _build_dummy_item().to_dict()

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path))
        if request.method == "GET" and request.url.path.endswith("/items/ds-1"):
            return httpx.Response(200, json=item_payload)
        if request.method == "PUT" and request.url.path.endswith("/items/ds-1"):
            body = request.content.decode()
            deprecated[0] = '"deprecated":true' in body
            return httpx.Response(200, json={})
        return httpx.Response(500)

    backend = _make_backend(handler)
    backend.deprecate_item("datasets", "ds-1")
    assert deprecated[0] is True
