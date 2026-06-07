"""Unit tests for the stride-sampled dataset preparation used by every data step."""

from __future__ import annotations

from pathlib import Path

from models.deepforest_tree_crowns.pipeline import _dataset_cache_dir, _prepare_dataset


def test_full_fraction_uses_every_labeled_chip(toy_chips: Path, toy_labels: Path) -> None:
    _, _, _, train_count, val_count = _prepare_dataset(str(toy_chips), str(toy_labels), 0.2, 42, 1.0)
    assert train_count + val_count == 6


def test_half_fraction_stride_samples_chips(toy_chips: Path, toy_labels: Path) -> None:
    _, _, _, train_count, val_count = _prepare_dataset(str(toy_chips), str(toy_labels), 0.2, 42, 0.5)
    assert train_count + val_count == 3


def test_cache_dir_keyed_on_sample_fraction(toy_chips: Path, toy_labels: Path) -> None:
    full = _dataset_cache_dir(str(toy_chips), str(toy_labels), 0.2, 42, 1.0)
    half = _dataset_cache_dir(str(toy_chips), str(toy_labels), 0.2, 42, 0.5)
    assert full != half
