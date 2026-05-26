"""Unit tests for the chips-dir subset helper used by every @step that prepares data."""

from __future__ import annotations

from pathlib import Path

import pytest

from models.yolo11n_detection.pipeline import _subset_chips_dir


def _populate(chips_dir: Path, n: int) -> None:
    chips_dir.mkdir()
    for i in range(n):
        chip = chips_dir / f"OAM-{i:04d}-0000-0000.tif"
        chip.write_bytes(b"")
        chip.with_name(chip.name + ".aux.xml").write_bytes(b"")


def test_subset_chips_dir_full_fraction_returns_input_unchanged(tmp_path: Path) -> None:
    chips = tmp_path / "chips"
    _populate(chips, 4)
    assert _subset_chips_dir(str(chips), 1.0) == str(chips)


def test_subset_chips_dir_fraction_above_one_returns_input_unchanged(tmp_path: Path) -> None:
    chips = tmp_path / "chips"
    _populate(chips, 4)
    assert _subset_chips_dir(str(chips), 1.5) == str(chips)


@pytest.mark.parametrize(
    ("n", "fraction", "expected_count"),
    [
        (10, 0.5, 5),
        (120, 0.1, 12),
        (100, 0.05, 5),
    ],
)
def test_subset_chips_dir_count_matches_fraction(tmp_path: Path, n: int, fraction: float, expected_count: int) -> None:
    chips = tmp_path / "chips"
    _populate(chips, n)
    subset = Path(_subset_chips_dir(str(chips), fraction))
    assert str(subset) != str(chips)
    tifs = sorted(subset.glob("OAM-*.tif"))
    sidecars = sorted(subset.glob("OAM-*.tif.aux.xml"))
    assert len(tifs) == expected_count
    assert len(sidecars) == expected_count
    assert all(f.is_symlink() for f in (*tifs, *sidecars))


def test_subset_chips_dir_symlinks_resolve_to_originals(tmp_path: Path) -> None:
    chips = tmp_path / "chips"
    _populate(chips, 8)
    subset = Path(_subset_chips_dir(str(chips), 0.5))
    for link in subset.iterdir():
        assert link.readlink().exists()
        assert link.resolve().parent == chips.resolve()
