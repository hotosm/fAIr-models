from datetime import UTC, datetime

import httpx
import pystac
import pytest

from fair.infra.registry import RegistryError, _parse, pin_image_digests, resolve_digest

DIGEST = "sha256:" + "a" * 64


def _registry(requests: list[httpx.Request]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.host == "ghcr.io" and request.url.path == "/token":
            assert request.url.params["scope"] == "repository:hotosm/fair-models/demo:pull"
            return httpx.Response(200, json={"token": "t0k"})
        if request.headers.get("authorization") != "Bearer t0k":
            return httpx.Response(
                401,
                headers={
                    "www-authenticate": 'Bearer realm="https://ghcr.io/token",service="ghcr.io",'
                    'scope="repository:hotosm/fair-models/demo:pull"'
                },
            )
        assert request.url.path == "/v2/hotosm/fair-models/demo/manifests/v1-inference"
        return httpx.Response(200, headers={"docker-content-digest": DIGEST})

    return httpx.MockTransport(handler)


def test_resolve_digest_follows_bearer_challenge() -> None:
    requests: list[httpx.Request] = []
    ref = "ghcr.io/hotosm/fair-models/demo:v1-inference"
    assert resolve_digest(ref, transport=_registry(requests)) == f"ghcr.io/hotosm/fair-models/demo@{DIGEST}"
    assert [r.url.path for r in requests] == [
        "/v2/hotosm/fair-models/demo/manifests/v1-inference",
        "/token",
        "/v2/hotosm/fair-models/demo/manifests/v1-inference",
    ]


def test_resolve_digest_passthrough_and_errors() -> None:
    pinned = f"ghcr.io/hotosm/demo@{DIGEST}"
    assert resolve_digest(pinned) == pinned
    missing = httpx.MockTransport(lambda _: httpx.Response(404))
    with pytest.raises(RegistryError):
        resolve_digest("ghcr.io/hotosm/demo:gone", transport=missing)


def test_parse_defaults() -> None:
    assert _parse("python:3.12") == ("python", "registry-1.docker.io", "library/python", "3.12")
    assert _parse("docker.io/org/app") == ("docker.io/org/app", "registry-1.docker.io", "org/app", "latest")
    assert _parse("localhost:5000/app:dev") == ("localhost:5000/app", "localhost:5000", "app", "dev")


def test_pin_image_digests_rewrites_only_image_refs() -> None:
    item = pystac.Item(id="demo", geometry=None, bbox=None, datetime=datetime.now(UTC), properties={})
    item.add_asset("mlm:inference", pystac.Asset(href="ghcr.io/hotosm/fair-models/demo:v1-inference"))
    item.add_asset("mlm:training", pystac.Asset(href="https://example.com/not-an-image"))
    pin_image_digests(item, transport=_registry([]))
    assert item.assets["mlm:inference"].href == f"ghcr.io/hotosm/fair-models/demo@{DIGEST}"
    assert item.assets["mlm:training"].href == "https://example.com/not-an-image"
