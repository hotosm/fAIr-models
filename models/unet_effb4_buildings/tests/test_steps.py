"""Step tests for unet-effb4-buildings.

The claim this model makes is that two buildings sharing a wall come out as two
objects. A semantic mask cannot express that: the pair is one connected
component whatever threshold is chosen. So the tests below fix the mask and vary
only the seeding channel, which is what decides the object count.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import scipy.ndimage

MODEL_DIR = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def pipeline():
    spec = importlib.util.spec_from_file_location(
        "unet_effb4_pipeline", MODEL_DIR / "pipeline.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Chip:
    """A 3-channel logit volume built to order: mask, EDT seeds, boundary."""

    SIZE = 64
    HIGH, LOW = 8.0, -8.0

    def __init__(self):
        self.logits = np.full((3, self.SIZE, self.SIZE), self.LOW, np.float32)

    def terrace(self) -> "Chip":
        """One mask blob spanning two buildings, with a seed in each half."""
        self.logits[0, 20:44, 10:54] = self.HIGH
        self.logits[1, 24:40, 14:30] = self.HIGH
        self.logits[1, 24:40, 34:50] = self.HIGH
        self.logits[2, 20:44, 31:33] = self.HIGH
        return self

    def single(self) -> "Chip":
        self.logits[0, 20:44, 10:30] = self.HIGH
        self.logits[1, 24:40, 14:26] = self.HIGH
        return self

    def mask_components(self) -> int:
        return scipy.ndimage.label(self.logits[0] > 0)[1]


def instance_count(labels) -> int:
    return len(np.unique(labels)) - 1  # unique labels minus background


def test_watershed_separates_what_the_mask_merges(pipeline):
    chip = Chip().terrace()
    assert chip.mask_components() == 1, "fixture must present a single blob"
    assert instance_count(pipeline.postprocess(chip.logits)) == 2


def test_single_building_stays_one_instance(pipeline):
    assert instance_count(pipeline.postprocess(Chip().single().logits)) == 1


def test_empty_prediction_yields_no_instances(pipeline):
    assert instance_count(pipeline.postprocess(Chip().logits)) == 0


def test_labels_are_integers_with_zero_background(pipeline):
    labels = pipeline.postprocess(Chip().terrace().logits)
    assert np.issubdtype(labels.dtype, np.integer)
    assert labels.min() == 0
    assert labels.shape == (Chip.SIZE, Chip.SIZE)


def test_threshold_controls_the_mask_extent(pipeline):
    """A higher threshold cannot grow the footprint."""
    chip = Chip().terrace()
    loose = (pipeline.postprocess(chip.logits, threshold=0.1) > 0).sum()
    tight = (pipeline.postprocess(chip.logits, threshold=0.9) > 0).sum()
    assert tight <= loose


def test_preprocess_normalises_and_binarises(pipeline):
    torch = pytest.importorskip("torch")
    batch = {
        "image": torch.full((2, 3, 64, 64), 255, dtype=torch.uint8),
        "mask": torch.full((2, 1, 64, 64), 3, dtype=torch.long),
    }
    images, masks = pipeline.preprocess(batch)
    assert images.shape == (2, 3, 64, 64)
    assert images.dtype == torch.float32
    assert set(masks.unique().tolist()) <= {0, 1}, "mask must be clamped to binary"
    assert masks.shape == (2, 64, 64)


def test_stac_item_declares_instance_segmentation():
    item = json.loads((MODEL_DIR / "stac-item.json").read_text())
    assert item["properties"]["mlm:tasks"] == ["instance-segmentation"]
