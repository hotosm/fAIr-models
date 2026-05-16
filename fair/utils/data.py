"""S3 data helpers for model pipelines.

Uses universal-pathlib (UPath) over fsspec for unified local/S3 file access.
fsspec/s3fs reads AWS_ENDPOINT_URL natively for MinIO compatibility.

Caching: fsspec supports URL-chaining (simplecache::s3://, filecache::s3://,
blockcache::s3://) — model developers opt in as needed.
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from upath import UPath

_PARALLEL_WORKERS = int(os.environ.get("FAIR_PARALLEL_IO_WORKERS", "8"))

if TYPE_CHECKING:
    import pystac

logger = logging.getLogger(__name__)


_DEFAULT_CACHE = Path(os.environ.get("FAIR_CACHE_DIR", Path(tempfile.gettempdir()) / "fair-data"))


def _is_remote(href: str) -> bool:
    return "://" in href


def s3_uri_to_http_url(s3_uri: str) -> str:
    parsed = urlparse(s3_uri)
    if parsed.scheme != "s3":
        return s3_uri
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    endpoint = os.environ.get("FAIR_S3_PUBLIC_URL", os.environ.get("AWS_ENDPOINT_URL", "")).rstrip("/")
    if endpoint:
        return f"{endpoint}/{bucket}/{key}"
    region = os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))
    return f"https://{bucket}.s3.{region}.amazonaws.com/{key}"


def http_url_to_s3_uri(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return url
    for env_var in ("FAIR_S3_PUBLIC_URL", "AWS_ENDPOINT_URL"):
        endpoint = os.environ.get(env_var, "").rstrip("/")
        if not endpoint:
            continue
        endpoint_parsed = urlparse(endpoint)
        if parsed.hostname == endpoint_parsed.hostname:
            path_parts = parsed.path.lstrip("/").split("/", 1)
            if len(path_parts) == 2:
                return f"s3://{path_parts[0]}/{path_parts[1]}"
    s3_match = re.match(r"^https?://(.+?)\.s3\.(.+?)\.amazonaws\.com/(.+)$", url)
    if s3_match:
        return f"s3://{s3_match.group(1)}/{s3_match.group(3)}"
    return url


def list_files(href: str, pattern: str = "*") -> list[str]:
    """List files under href matching glob pattern.

    Args:
        href: Local path or s3://bucket/prefix.
        pattern: Glob pattern (e.g. "OAM-*.tif").
    """
    p = UPath(href)
    return sorted(str(f) for f in p.glob(pattern) if not f.is_dir())


def count_chips(chips_href: str) -> int:
    """Count image files in a chips directory (local or S3).

    Counts files matching common raster extensions. Useful for
    setting fair:chip_count on STAC dataset items.
    """
    total = 0
    for ext in ("*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg"):
        total += len(list_files(chips_href, ext))
    return total


def resolve_path(href: str, local_dir: Path | None = None) -> Path:
    """Download a single remote file to local cache. Local paths pass through.

    Args:
        href: Local path or s3://bucket/key URI.
        local_dir: Download target directory. Defaults to /tmp/fair-data.
    """
    if not _is_remote(href):
        return Path(href)

    # Derive cache path from URI without instantiating UPath (avoids remote access)
    rel = urlparse(href).path.lstrip("/")
    dest = (local_dir or _DEFAULT_CACHE) / rel

    if dest.exists():
        logger.debug("Cache hit: %s", dest)
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s -> %s", href, dest)
    dest.write_bytes(UPath(href).read_bytes())
    return dest


def resolve_directory(href: str, pattern: str = "*", local_dir: Path | None = None) -> Path:
    """Download all files under a remote prefix to local cache. Local paths pass through.

    Args:
        href: Local path or s3://bucket/prefix.
        pattern: Glob pattern to filter files (e.g. "OAM-*.tif").
        local_dir: Download target root. Defaults to /tmp/fair-data.
    """
    if not _is_remote(href):
        return Path(href)

    uris = list_files(href, pattern)
    if not uris:
        msg = f"No files matching '{pattern}' found at {href}"
        raise FileNotFoundError(msg)

    cache = local_dir or _DEFAULT_CACHE

    with ThreadPoolExecutor(max_workers=_PARALLEL_WORKERS) as pool:
        locals_ = list(pool.map(lambda u: resolve_path(u, local_dir=cache), uris))
    return locals_[0].parent


def create_dataset_archive(
    chips_dir: str,
    labels_dir: str,
    output_path: str,
) -> str:
    """Zip chips and labels directories into a single archive.

    Args:
        chips_dir: Path (local or s3://) to the chips directory.
        labels_dir: Path (local or s3://) to the labels directory.
        output_path: Local path for the output .zip file.

    Returns:
        The output_path after the archive is written.
    """
    chips = Path(chips_dir) if not _is_remote(chips_dir) else resolve_directory(chips_dir)
    labels = Path(labels_dir) if not _is_remote(labels_dir) else resolve_directory(labels_dir)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(chips.rglob("*")):
            if f.is_file():
                zf.write(f, Path("chips") / f.relative_to(chips))
        for f in sorted(labels.rglob("*")):
            if f.is_file():
                zf.write(f, Path("labels") / f.relative_to(labels))

    logger.info("Created dataset archive: %s", out)
    return str(out)


def upload_storage_options() -> dict:
    """fsspec/s3fs storage options applied to upload writes.

    FAIR_S3_UPLOAD_ACL: canned S3 ACL (e.g. "public-read"). When set, passed
    via s3_additional_kwargs so every put/open call on the filesystem
    inherits it. Unset means no ACL header is sent (bucket default applies).

    Pass to UPath at construction time: UPath(uri, **upload_storage_options()).
    Children created via `/` inherit storage_options.
    """
    acl = os.environ.get("FAIR_S3_UPLOAD_ACL", "").strip()
    if not acl:
        return {}
    return {"s3_additional_kwargs": {"ACL": acl}}


def mirror(src: str | Path, dest: str | Path) -> None:
    """Copy a file or directory tree from src to dest.

    src and dest may each be local paths or remote URIs (s3://, https://, etc).
    If src is a directory, dest is treated as a directory prefix and the tree
    is copied recursively. If src is a file, dest is treated as the target file path.

    Upload options from upload_storage_options() (e.g. ACL) are applied at the
    destination. Remote-to-remote copies stream through the local process.
    """
    src_path = UPath(str(src))
    dest_str = str(dest)

    if src_path.is_dir():
        dest_path = UPath(dest_str, **upload_storage_options())
        files = [f for f in sorted(src_path.rglob("*")) if f.is_file()]

        def _copy(f: UPath) -> None:
            target = dest_path / f.relative_to(src_path)
            target.write_bytes(f.read_bytes())
            logger.info("Mirrored %s -> %s", f, target)

        with ThreadPoolExecutor(max_workers=_PARALLEL_WORKERS) as pool:
            list(pool.map(_copy, files))
        return

    if src_path.is_file():
        dest_path = UPath(dest_str, **upload_storage_options())
        dest_path.write_bytes(src_path.read_bytes())
        logger.info("Mirrored %s -> %s", src_path, dest_path)
        return

    raise FileNotFoundError(f"mirror source not found: {src}")


def upload_item_assets(
    item: pystac.Item,
    data_prefix: str,
    collection_id: str,
) -> pystac.Item:
    """Upload local asset files to S3 and rewrite hrefs in-place.

    Deterministic path: {data_prefix}/{collection_id}/{item.id}/{asset_key}/...
    Files are uploaded; directories are uploaded recursively.
    Remote hrefs are left untouched.

    Returns the item with rewritten hrefs.
    """
    for key, asset in item.assets.items():
        if _is_remote(asset.href):
            continue

        remote_base = f"{data_prefix}/{collection_id}/{item.id}/{key}"
        local = Path(asset.href)

        if local.is_dir():
            mirror(local, remote_base)
            asset.href = s3_uri_to_http_url(remote_base)
        elif local.is_file():
            remote_path = f"{remote_base}/{local.name}"
            mirror(local, remote_path)
            asset.href = s3_uri_to_http_url(remote_path)
        else:
            logger.warning("Asset '%s' href not found locally: %s", key, asset.href)

    return item


def upload_local_directory(local_dir: Path, remote_prefix: str) -> None:
    """Recursively upload a local directory to a remote prefix. Thin wrapper over mirror()."""
    mirror(local_dir, remote_prefix)
