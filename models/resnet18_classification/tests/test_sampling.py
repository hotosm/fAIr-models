"""Unit tests for the chip-sampling helper used by every @step that loads data."""

from __future__ import annotations

from pathlib import Path

import pytest

from models.resnet18_classification.pipeline import _stride_subset


def _make_samples(n: int) -> list[tuple[Path, int]]:
    return [(Path(f"chip_{i:04d}.tif"), i % 2) for i in range(n)]


def test_stride_subset_full_fraction_returns_input_unchanged() -> None:
    samples = _make_samples(10)
    assert _stride_subset(samples, 1.0) is samples


def test_stride_subset_fraction_above_one_returns_input_unchanged() -> None:
    samples = _make_samples(10)
    assert _stride_subset(samples, 2.5) is samples


@pytest.mark.parametrize(
    ("n", "fraction", "expected_count"),
    [
        (10, 0.5, 5),
        (120, 0.1, 12),
        (100, 0.05, 5),
        (100, 0.01, 1),
        (3, 0.5, 2),
    ],
)
def test_stride_subset_count_matches_fraction(n: int, fraction: float, expected_count: int) -> None:
    out = _stride_subset(_make_samples(n), fraction)
    assert len(out) == expected_count


def test_stride_subset_is_deterministic_and_preserves_order() -> None:
    samples = _make_samples(20)
    out = _stride_subset(samples, 0.25)
    assert out == samples[::4]
    assert _stride_subset(samples, 0.25) == out
