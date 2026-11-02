from __future__ import annotations

import contextlib
import functools
import http.server
import io
import threading
import time
import zipfile
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import aiohttp
import fsspec
import pystac
import pytest

from fair.utils.data import (
    _is_remote,
    create_dataset_archive,
    list_files,
    resolve_directory,
    resolve_path,
    upload_item_assets,
    upload_storage_options,
)

_NOW = datetime(2024, 1, 1, tzinfo=UTC)


class TestIsRemote:
    def test_s3(self) -> None:
        assert _is_remote("s3://bucket/key") is True

    def test_https(self) -> None:
        assert _is_remote("https://example.com/file") is True

    def test_local(self) -> None:
        assert _is_remote("/data/sample/train") is False

    def test_relative(self) -> None:
        assert _is_remote("data/file.tif") is False


class TestListFiles:
    @patch("fair.utils.data.UPath")
    def test_glob_s3(self, mock_upath_cls: MagicMock) -> None:
        mock_path = MagicMock()
        mock_upath_cls.return_value = mock_path
        f1, f2 = MagicMock(), MagicMock()
        f1.__str__ = lambda _: "s3://bucket/oam/OAM-1-2-3.tif"
        f1.is_dir.return_value = False
        f2.__str__ = lambda _: "s3://bucket/oam/OAM-4-5-6.tif"
        f2.is_dir.return_value = False
        mock_path.glob.return_value = [f2, f1]

        result = list_files("s3://bucket/oam", "OAM-*.tif")
        assert result == ["s3://bucket/oam/OAM-1-2-3.tif", "s3://bucket/oam/OAM-4-5-6.tif"]
        mock_path.glob.assert_called_once_with("OAM-*.tif")

    def test_local_glob(self, tmp_path: Path) -> None:
        (tmp_path / "a.tif").write_bytes(b"x")
        (tmp_path / "b.tif").write_bytes(b"y")
        (tmp_path / "c.txt").write_bytes(b"z")
        result = list_files(str(tmp_path), "*.tif")
        assert len(result) == 2
        assert all(r.endswith(".tif") for r in result)


class TestResolvePath:
    def test_local_passthrough(self) -> None:
        assert resolve_path("/data/sample/train/oam") == Path("/data/sample/train/oam")

    def test_cached_file(self, tmp_path: Path) -> None:
        cache = tmp_path / "cache"
        target = cache / "data" / "file.tif"
        target.parent.mkdir(parents=True)
        target.write_bytes(b"fake")
        result = resolve_path("s3://bucket/data/file.tif", local_dir=cache)
        assert result == target

    @patch("fair.utils.data.UPath")
    def test_downloads_from_s3(self, mock_upath_cls: MagicMock, tmp_path: Path) -> None:
        mock_remote = MagicMock()
        mock_upath_cls.return_value = mock_remote
        mock_remote.path = "/data/file.tif"
        mock_remote.protocol = "s3"
        mock_remote.open.return_value = io.BytesIO(b"raster-data")

        result = resolve_path("s3://bucket/data/file.tif", local_dir=tmp_path)
        assert result == tmp_path / "data" / "file.tif"
        assert result.read_bytes() == b"raster-data"
        mock_remote.open.assert_called_once_with("rb")


class TestResolveDirectory:
    def test_local_passthrough(self) -> None:
        assert resolve_directory("/data/sample/train/oam") == Path("/data/sample/train/oam")

    @patch("fair.utils.data.UPath")
    def test_downloads_prefix(self, mock_upath_cls: MagicMock, tmp_path: Path) -> None:
        listing = MagicMock()
        f1, f2 = MagicMock(), MagicMock()
        f1.__str__ = lambda _: "s3://bucket/train/oam/OAM-1-2-3.tif"
        f1.is_dir.return_value = False
        f1.protocol = "s3"
        f1.open.return_value = io.BytesIO(b"tile1")
        f2.__str__ = lambda _: "s3://bucket/train/oam/OAM-4-5-6.tif"
        f2.is_dir.return_value = False
        f2.protocol = "s3"
        f2.open.return_value = io.BytesIO(b"tile2")
        listing.glob.return_value = [f1, f2]

        mocks = {
            "s3://bucket/train/oam": listing,
            "s3://bucket/train/oam/OAM-1-2-3.tif": f1,
            "s3://bucket/train/oam/OAM-4-5-6.tif": f2,
        }
        mock_upath_cls.side_effect = lambda href: mocks[href]

        result = resolve_directory("s3://bucket/train/oam", pattern="OAM-*.tif", local_dir=tmp_path)
        assert result == tmp_path / "train" / "oam"
        assert (tmp_path / "train" / "oam" / "OAM-1-2-3.tif").read_bytes() == b"tile1"
        assert (tmp_path / "train" / "oam" / "OAM-4-5-6.tif").read_bytes() == b"tile2"

    @patch("fair.utils.data.UPath")
    def test_skips_cached(self, mock_upath_cls: MagicMock, tmp_path: Path) -> None:
        listing = MagicMock()
        f1 = MagicMock()
        f1.__str__ = lambda _: "s3://bucket/d/f.tif"
        f1.is_dir.return_value = False
        listing.glob.return_value = [f1]

        cached = tmp_path / "d" / "f.tif"
        cached.parent.mkdir(parents=True)
        cached.write_bytes(b"cached")

        mock_upath_cls.side_effect = lambda _href: listing
        resolve_directory("s3://bucket/d", local_dir=tmp_path)
        f1.read_bytes.assert_not_called()

    @patch("fair.utils.data.UPath")
    def test_raises_on_empty_prefix(self, mock_upath_cls: MagicMock, tmp_path: Path) -> None:
        mock_remote = MagicMock()
        mock_remote.glob.return_value = []
        mock_upath_cls.return_value = mock_remote

        with pytest.raises(FileNotFoundError, match="No files matching"):
            resolve_directory("s3://bucket/empty-prefix", local_dir=tmp_path)


class TestCreateDatasetArchive:
    def test_creates_zip_with_chips_and_labels(self, tmp_path: Path) -> None:
        chips = tmp_path / "chips"
        chips.mkdir()
        (chips / "a.tif").write_bytes(b"chip-a")
        (chips / "b.tif").write_bytes(b"chip-b")

        labels = tmp_path / "labels"
        labels.mkdir()
        (labels / "a.geojson").write_text('{"type":"Feature"}')

        out = tmp_path / "archive.zip"
        result = create_dataset_archive({"image": str(chips)}, str(labels), str(out))
        assert result == str(out)
        assert out.exists()

        with zipfile.ZipFile(out) as zf:
            names = sorted(zf.namelist())
            assert "chips-image/a.tif" in names
            assert "chips-image/b.tif" in names
            assert "labels/a.geojson" in names
            assert len(names) == 3

    def test_puts_each_input_under_its_own_folder(self, tmp_path: Path) -> None:
        for folder, name in (("post", "a.tif"), ("pre", "a.tif"), ("labels", "a.geojson")):
            (tmp_path / folder).mkdir()
            (tmp_path / folder / name).write_bytes(b"x")

        out = tmp_path / "archive.zip"
        create_dataset_archive(
            {"post": str(tmp_path / "post"), "pre": str(tmp_path / "pre")}, str(tmp_path / "labels"), str(out)
        )

        with zipfile.ZipFile(out) as zf:
            assert sorted(zf.namelist()) == ["chips-post/a.tif", "chips-pre/a.tif", "labels/a.geojson"]


class TestUploadItemAssets:
    def _make_item(self, tmp_path: Path) -> pystac.Item:
        chips_dir = tmp_path / "chips"
        chips_dir.mkdir()
        (chips_dir / "a.tif").write_bytes(b"chip-a")
        (chips_dir / "b.tif").write_bytes(b"chip-b")

        label_file = tmp_path / "labels" / "a.geojson"
        label_file.parent.mkdir()
        label_file.write_text('{"type":"Feature"}')

        archive = tmp_path / "archive.zip"
        archive.write_bytes(b"fake-zip")

        item = pystac.Item(
            id="test-dataset",
            geometry={"type": "Point", "coordinates": [0, 0]},
            bbox=[0, 0, 0, 0],
            datetime=_NOW,
            properties={},
        )
        item.add_asset("chips", pystac.Asset(href=str(chips_dir), roles=["data"]))
        item.add_asset("labels", pystac.Asset(href=str(label_file), roles=["labels"]))
        item.add_asset("download", pystac.Asset(href=str(archive), roles=["data", "archive"]))
        return item

    @patch("fair.utils.data.UPath")
    def test_uploads_and_rewrites_hrefs(self, mock_upath_cls: MagicMock, tmp_path: Path) -> None:
        item = self._make_item(tmp_path)
        mock_dest = MagicMock()
        mock_upath_cls.return_value = mock_dest

        upload_item_assets(item, "s3://bucket/data", "datasets")

        assert (
            item.assets["chips"].href == "https://bucket.s3.us-east-1.amazonaws.com/data/datasets/test-dataset/v1/chips"
        )
        assert (
            item.assets["labels"].href
            == "https://bucket.s3.us-east-1.amazonaws.com/data/datasets/test-dataset/v1/labels/a.geojson"
        )
        assert (
            item.assets["download"].href
            == "https://bucket.s3.us-east-1.amazonaws.com/data/datasets/test-dataset/v1/download/archive.zip"
        )

    def test_skips_remote_hrefs(self) -> None:
        item = pystac.Item(
            id="remote-item",
            geometry={"type": "Point", "coordinates": [0, 0]},
            bbox=[0, 0, 0, 0],
            datetime=_NOW,
            properties={},
        )
        item.add_asset("chips", pystac.Asset(href="s3://bucket/already-remote", roles=["data"]))

        upload_item_assets(item, "s3://bucket/data", "datasets")
        assert item.assets["chips"].href == "s3://bucket/already-remote"

    def test_skips_runtime_image_refs(self, caplog: pytest.LogCaptureFixture) -> None:
        item = pystac.Item(
            id="image-item",
            geometry={"type": "Point", "coordinates": [0, 0]},
            bbox=[0, 0, 0, 0],
            datetime=_NOW,
            properties={},
        )
        pinned = "ghcr.io/hotosm/fair-models/demo@sha256:" + "a" * 64
        item.add_asset("mlm:training", pystac.Asset(href=pinned))
        item.add_asset("mlm:inference", pystac.Asset(href=pinned))

        upload_item_assets(item, "s3://bucket/data", "base-models")
        assert item.assets["mlm:training"].href == pinned
        assert item.assets["mlm:inference"].href == pinned
        assert "not found locally" not in caplog.text

    def test_returns_item(self, tmp_path: Path) -> None:
        item = self._make_item(tmp_path)
        with patch("fair.utils.data.UPath"):
            result = upload_item_assets(item, "s3://bucket/data", "datasets")
        assert result is item


class TestUploadStorageOptions:
    def test_unset_returns_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("FAIR_S3_UPLOAD_ACL", raising=False)
        assert upload_storage_options() == {}

    def test_blank_returns_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FAIR_S3_UPLOAD_ACL", "   ")
        assert upload_storage_options() == {}

    def test_set_returns_acl(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FAIR_S3_UPLOAD_ACL", "public-read")
        assert upload_storage_options() == {"s3_additional_kwargs": {"ACL": "public-read"}}

    @patch("fair.utils.data.UPath")
    def test_upload_item_assets_passes_acl_kwarg(
        self,
        mock_upath_cls: MagicMock,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("FAIR_S3_UPLOAD_ACL", "public-read")
        f = tmp_path / "thing.bin"
        f.write_bytes(b"x")
        item = pystac.Item(
            id="i",
            geometry={"type": "Point", "coordinates": [0, 0]},
            bbox=[0, 0, 0, 0],
            datetime=_NOW,
            properties={},
        )
        item.add_asset("a", pystac.Asset(href=str(f), roles=["data"]))

        upload_item_assets(item, "s3://bucket/data", "datasets")

        assert mock_upath_cls.call_args.kwargs == {"s3_additional_kwargs": {"ACL": "public-read"}}


WEIGHTS = bytes(range(256)) * 40_000


class _NoRangeHandler(http.server.SimpleHTTPRequestHandler):
    """The stdlib server ignores Range; saying so makes fsspec stream the body."""

    def end_headers(self) -> None:
        self.send_header("Accept-Ranges", "none")
        super().end_headers()


class _TruncatingHandler(http.server.BaseHTTPRequestHandler):
    """Declares the full payload, sends half of it, then closes the connection."""

    def do_HEAD(self) -> None:
        self._send_headers()

    def do_GET(self) -> None:
        self._send_headers()
        self.wfile.write(WEIGHTS[: len(WEIGHTS) // 2])
        self.close_connection = True

    def _send_headers(self) -> None:
        self.send_response(200)
        self.send_header("Content-Length", str(len(WEIGHTS)))
        self.send_header("Accept-Ranges", "none")
        self.end_headers()


@contextlib.contextmanager
def _serve(handler: Callable[..., http.server.BaseHTTPRequestHandler]) -> Iterator[str]:
    """Base URL of a local HTTP server running `handler` for the duration of the block."""
    with http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler) as server:
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            yield f"http://127.0.0.1:{server.server_address[1]}"
        finally:
            server.shutdown()


@pytest.fixture
def served_weights(tmp_path: Path) -> Path:
    served = tmp_path / "served"
    served.mkdir()
    (served / "weights.ckpt").write_bytes(WEIGHTS)
    return served


def test_mirror_copies_a_file_served_over_http_byte_for_byte(tmp_path: Path, served_weights: Path) -> None:
    from fair.utils.data import mirror

    (tmp_path / "copy").mkdir()
    with _serve(functools.partial(_NoRangeHandler, directory=str(served_weights))) as base_url:
        mirror(f"{base_url}/weights.ckpt", tmp_path / "copy" / "weights.ckpt")

    assert (tmp_path / "copy" / "weights.ckpt").read_bytes() == WEIGHTS


def test_resolve_path_streams_an_http_file_into_the_cache(tmp_path: Path, served_weights: Path) -> None:
    handler = functools.partial(_NoRangeHandler, directory=str(served_weights))
    with _serve(handler) as base_url:
        result = resolve_path(f"{base_url}/weights.ckpt", local_dir=tmp_path / "cache")

    assert result.read_bytes() == WEIGHTS


def test_resolve_path_cache_entry_holds_the_whole_file_once_it_appears(
    tmp_path: Path, served_weights: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Smaller than a write buffer, so unflushed bytes would still be buffered at the rename.
    (served_weights / "small.bin").write_bytes(WEIGHTS[:1000])
    seen_at_rename: list[bytes] = []
    rename = Path.replace

    def rename_and_read(self: Path, target: Path) -> Path:
        renamed = rename(self, target)
        seen_at_rename.append(Path(target).read_bytes())
        return renamed

    monkeypatch.setattr(Path, "replace", rename_and_read)
    with _serve(functools.partial(_NoRangeHandler, directory=str(served_weights))) as base_url:
        resolve_path(f"{base_url}/small.bin", local_dir=tmp_path / "cache")

    assert seen_at_rename == [WEIGHTS[:1000]]


def test_resolve_path_leaves_no_cache_entry_when_a_download_is_cut_short(tmp_path: Path) -> None:
    with _serve(_TruncatingHandler) as base_url, pytest.raises(aiohttp.ClientPayloadError):
        resolve_path(f"{base_url}/weights.ckpt", local_dir=tmp_path / "cache")

    assert not [path for path in (tmp_path / "cache").rglob("*") if path.is_file()]


class _StallingHandler(http.server.BaseHTTPRequestHandler):
    """Sends the headers and a first chunk, then goes quiet past the read timeout."""

    def do_HEAD(self) -> None:
        self._send_headers()

    def do_GET(self) -> None:
        self._send_headers()
        self.wfile.write(WEIGHTS[:1024])
        self.wfile.flush()
        time.sleep(5)

    def _send_headers(self) -> None:
        self.send_response(200)
        self.send_header("Content-Length", str(len(WEIGHTS)))
        self.send_header("Accept-Ranges", "none")
        self.end_headers()


def test_resolve_path_fails_when_a_download_stalls_past_the_read_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fair.utils import data as data_module

    monkeypatch.setattr(data_module, "_HTTP_READ_TIMEOUT_SECONDS", 1)
    started = time.monotonic()
    with _serve(_StallingHandler) as base_url, pytest.raises(fsspec.exceptions.FSTimeoutError):
        resolve_path(f"{base_url}/weights.ckpt", local_dir=tmp_path / "cache")

    assert time.monotonic() - started < 5


def test_dataset_archive_rejects_an_input_name_that_is_not_a_slug(tmp_path: Path) -> None:
    from fair.utils.data import create_dataset_archive

    with pytest.raises(ValueError, match="slugs"):
        create_dataset_archive({"../pre": str(tmp_path)}, str(tmp_path), str(tmp_path / "archive.zip"))
    assert not (tmp_path / "archive.zip").exists()
