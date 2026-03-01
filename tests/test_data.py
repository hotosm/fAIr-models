from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from fair.utils.data import _is_remote, list_files, resolve_directory, resolve_path


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
        mock_remote.read_bytes.return_value = b"raster-data"

        result = resolve_path("s3://bucket/data/file.tif", local_dir=tmp_path)
        assert result == tmp_path / "data" / "file.tif"
        assert result.read_bytes() == b"raster-data"
        mock_remote.read_bytes.assert_called_once()


class TestResolveDirectory:
    def test_local_passthrough(self) -> None:
        assert resolve_directory("/data/sample/train/oam") == Path("/data/sample/train/oam")

    @patch("fair.utils.data.UPath")
    def test_downloads_prefix(self, mock_upath_cls: MagicMock, tmp_path: Path) -> None:
        mock_remote = MagicMock()
        mock_upath_cls.return_value = mock_remote
        mock_remote.path = "/train/oam"

        f1, f2 = MagicMock(), MagicMock()
        f1.__str__ = lambda _: "s3://bucket/train/oam/OAM-1-2-3.tif"
        f1.is_dir.return_value = False
        f1.path = "/train/oam/OAM-1-2-3.tif"
        f1.read_bytes.return_value = b"tile1"
        f2.__str__ = lambda _: "s3://bucket/train/oam/OAM-4-5-6.tif"
        f2.is_dir.return_value = False
        f2.path = "/train/oam/OAM-4-5-6.tif"
        f2.read_bytes.return_value = b"tile2"
        mock_remote.glob.return_value = [f1, f2]

        # Mock UPath constructor for each individual file resolve_path call
        mock_upath_cls.side_effect = [mock_remote, f1, f2]

        result = resolve_directory("s3://bucket/train/oam", pattern="OAM-*.tif", local_dir=tmp_path)
        assert result == tmp_path / "train" / "oam"
        assert (tmp_path / "train" / "oam" / "OAM-1-2-3.tif").read_bytes() == b"tile1"
        assert (tmp_path / "train" / "oam" / "OAM-4-5-6.tif").read_bytes() == b"tile2"

    @patch("fair.utils.data.UPath")
    def test_skips_cached(self, mock_upath_cls: MagicMock, tmp_path: Path) -> None:
        mock_remote = MagicMock()
        mock_remote.path = "/d"

        f1 = MagicMock()
        f1.__str__ = lambda _: "s3://bucket/d/f.tif"
        f1.is_dir.return_value = False
        f1.path = "/d/f.tif"
        mock_remote.glob.return_value = [f1]

        # Pre-create the cached file
        cached = tmp_path / "d" / "f.tif"
        cached.parent.mkdir(parents=True)
        cached.write_bytes(b"cached")

        mock_upath_cls.side_effect = [mock_remote, f1]
        resolve_directory("s3://bucket/d", local_dir=tmp_path)
        f1.read_bytes.assert_not_called()
