"""S3 data helpers for model pipelines.

Uses universal-pathlib (UPath) over fsspec for unified local/S3 file access.
fsspec/s3fs reads AWS_ENDPOINT_URL natively for MinIO compatibility.

Caching: fsspec supports URL-chaining (simplecache::s3://, filecache::s3://,
blockcache::s3://) — model developers opt in as needed.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from urllib.parse import urlparse

from upath import UPath

logger = logging.getLogger(__name__)


_DEFAULT_CACHE = Path(os.environ.get("FAIR_CACHE_DIR", Path(tempfile.gettempdir()) / "fair-data"))


def _is_remote(href: str) -> bool:
    return "://" in href


def list_files(href: str, pattern: str = "*") -> list[str]:
    """List files under href matching glob pattern.

    Args:
        href: Local path or s3://bucket/prefix.
        pattern: Glob pattern (e.g. "OAM-*.tif").
    """
    p = UPath(href)
    return sorted(str(f) for f in p.glob(pattern) if not f.is_dir())


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
    dest_dir: Path | None = None

    for uri in uris:
        local = resolve_path(uri, local_dir=cache)
        if dest_dir is None:
            dest_dir = local.parent

    assert dest_dir is not None  # guaranteed by non-empty uris
    return dest_dir
