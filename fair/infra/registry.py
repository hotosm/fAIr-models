"""Resolve OCI image tags to immutable digests via the registry v2 API (anonymous pull)."""

import re

import httpx
import pystac

_ACCEPT = ", ".join(
    (
        "application/vnd.oci.image.index.v1+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.docker.distribution.manifest.v2+json",
    )
)
IMAGE_ASSET_KEYS = ("mlm:training", "mlm:inference")


class RegistryError(Exception):
    """An image reference could not be resolved to a digest."""


def _parse(ref: str) -> tuple[str, str, str, str]:
    """Split `[registry/]repo[:tag]` into (name, registry host, repo path, tag)."""
    last = ref.rsplit("/", 1)[-1]
    name, tag = ref.rsplit(":", 1) if ":" in last else (ref, "latest")
    first, _, rest = name.partition("/")
    if rest and ("." in first or ":" in first or first == "localhost"):
        host, repo = first, rest
    else:
        host, repo = "registry-1.docker.io", name if "/" in name else f"library/{name}"
    if host == "docker.io":
        host = "registry-1.docker.io"
    return name, host, repo, tag


def resolve_digest(ref: str, *, transport: httpx.BaseTransport | None = None, timeout: float = 30.0) -> str:
    """Return `ref` pinned as `name@sha256:...`. Digest references pass through unchanged."""
    if "@sha256:" in ref:
        return ref
    name, host, repo, tag = _parse(ref)
    url = f"https://{host}/v2/{repo}/manifests/{tag}"
    headers = {"Accept": _ACCEPT}
    try:
        with httpx.Client(transport=transport, timeout=timeout, follow_redirects=True) as http:
            resp = http.head(url, headers=headers)
            if resp.status_code == 401:
                challenge = dict(re.findall(r'(\w+)="([^"]*)"', resp.headers.get("www-authenticate", "")))
                if "realm" not in challenge:
                    raise RegistryError(f"{ref}: registry sent no bearer challenge")
                token_resp = http.get(
                    challenge["realm"],
                    params={
                        "service": challenge.get("service", host),
                        "scope": challenge.get("scope", f"repository:{repo}:pull"),
                    },
                )
                token_resp.raise_for_status()
                body = token_resp.json()
                headers["Authorization"] = f"Bearer {body.get('token') or body.get('access_token')}"
                resp = http.head(url, headers=headers)
            resp.raise_for_status()
    except httpx.HTTPError as exc:
        raise RegistryError(f"{ref}: {exc}") from exc
    digest = resp.headers.get("docker-content-digest")
    if not digest:
        raise RegistryError(f"{ref}: registry returned no Docker-Content-Digest")
    return f"{name}@{digest}"


def pin_image_digests(item: pystac.Item, *, transport: httpx.BaseTransport | None = None) -> None:
    """Pin runtime image assets to digests. Leave URL assets unchanged."""
    for key in IMAGE_ASSET_KEYS:
        asset = item.assets.get(key)
        if asset is not None and "://" not in asset.href:
            asset.href = resolve_digest(asset.href, transport=transport)
