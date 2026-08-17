"""Tests for the deterministic chip subset used by data-preparation steps."""

from pathlib import Path

import pytest

from models.yolo_swag_waste_grid_segmentation.pipeline import _subset_chips_dir


def _populate(chips_dir: Path, chip_count: int) -> None:
    chips_dir.mkdir()
    for i in range(chip_count):
        chip = chips_dir / f"OAM-{i:04d}-0000-0000.tif"
        chip.write_bytes(b"")
        chip.with_name(chip.name + ".aux.xml").write_bytes(b"")


def test_full_sample_uses_the_original_chip_directory(tmp_path: Path) -> None:
    chips = tmp_path / "chips"
    _populate(chips, 4)
    assert _subset_chips_dir(str(chips), 1.0) == str(chips)


def test_fractional_sample_creates_resolving_chip_and_sidecar_links(tmp_path: Path) -> None:
    chips = tmp_path / "chips"
    _populate(chips, 8)

    subset = Path(_subset_chips_dir(str(chips), 0.75))
    tifs = sorted(subset.glob("OAM-*.tif"))
    sidecars = sorted(subset.glob("OAM-*.tif.aux.xml"))

    assert len(tifs) == len(sidecars) == 6
    for link in (*tifs, *sidecars):
        assert link.is_symlink()
        assert link.resolve().parent == chips.resolve()


@pytest.mark.parametrize("fraction", [0.0, -0.5, 1.1])
def test_invalid_sample_fraction_is_rejected(tmp_path: Path, fraction: float) -> None:
    chips = tmp_path / "chips"
    _populate(chips, 1)

    with pytest.raises(ValueError, match="sample_fraction"):
        _subset_chips_dir(str(chips), fraction)
