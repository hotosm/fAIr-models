"""Step tests for unet-effb4-buildings.

The claim this model makes is that two buildings sharing a wall come out as two
objects. A semantic mask cannot express that: the pair is one connected component
whatever threshold is chosen. So the tests below fix the mask and vary only the
seeding channel, which is what decides the object count.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import scipy.ndimage

MODEL_DIR = Path(__file__).resolve().parents[1]


class Chip:
    """A 4-channel logit volume built to order: mask, EDT, boundary, centreness."""

    SIZE = 64
    HIGH, LOW = 8.0, -8.0

    def __init__(self) -> None:
        self.logits = np.full((4, self.SIZE, self.SIZE), self.LOW, np.float32)

    def terrace(self) -> Chip:
        """One mask blob spanning two buildings, with a seed in each half."""
        self.logits[0, 20:44, 10:54] = self.HIGH
        self.logits[1, 24:40, 14:30] = self.HIGH
        self.logits[1, 24:40, 34:50] = self.HIGH
        self.logits[2, 20:44, 31:33] = self.HIGH
        self.logits[3, 28:36, 18:26] = self.HIGH
        self.logits[3, 28:36, 38:46] = self.HIGH
        return self

    def single(self) -> Chip:
        self.logits[0, 20:44, 10:30] = self.HIGH
        self.logits[1, 24:40, 14:26] = self.HIGH
        self.logits[3, 28:36, 16:24] = self.HIGH
        return self

    def speck(self) -> Chip:
        """A 4 x 4 px detection, below the default minimum instance size."""
        self.logits[0, 2:6, 2:6] = self.HIGH
        self.logits[3, 3:5, 3:5] = self.HIGH
        return self

    def mask_components(self) -> int:
        return int(scipy.ndimage.label(self.logits[0] > 0)[1])


def instance_count(labels: np.ndarray) -> int:
    return len(np.unique(labels)) - 1  # unique labels minus background


def test_watershed_separates_what_the_mask_merges() -> None:
    from models.unet_effb4_buildings.pipeline import postprocess

    chip = Chip().terrace()
    assert chip.mask_components() == 1, "fixture must present a single blob"
    assert instance_count(postprocess(chip.logits)) == 2


def test_single_building_stays_one_instance() -> None:
    from models.unet_effb4_buildings.pipeline import postprocess

    assert instance_count(postprocess(Chip().single().logits)) == 1


def test_empty_prediction_yields_no_instances() -> None:
    from models.unet_effb4_buildings.pipeline import postprocess

    assert instance_count(postprocess(Chip().logits)) == 0


def test_labels_are_integers_with_zero_background() -> None:
    from models.unet_effb4_buildings.pipeline import postprocess

    labels = postprocess(Chip().terrace().logits)
    assert np.issubdtype(labels.dtype, np.integer)
    assert labels.min() == 0
    assert labels.shape == (Chip.SIZE, Chip.SIZE)


def test_threshold_cannot_grow_the_footprint() -> None:
    from models.unet_effb4_buildings.pipeline import postprocess

    chip = Chip().terrace()
    loose = (postprocess(chip.logits, threshold=0.1) > 0).sum()
    tight = (postprocess(chip.logits, threshold=0.9) > 0).sum()
    assert tight <= loose


def test_small_instances_are_dropped() -> None:
    from models.unet_effb4_buildings.pipeline import postprocess

    chip = Chip().single().speck()
    assert instance_count(postprocess(chip.logits)) == 1
    assert instance_count(postprocess(chip.logits, min_instance_px=0)) == 2


def test_d4_views_round_trip() -> None:
    """Every TTA view must be mapped back onto the original pixel grid."""
    from models.unet_effb4_buildings.pipeline import _d4_mean, _d4_views

    batch = np.random.rand(1, 4, 16, 16).astype(np.float32)
    views = _d4_views(batch)
    assert views.shape == (8, 4, 16, 16)
    assert len({v.tobytes() for v in views}) == 8, "views must be distinct"
    np.testing.assert_allclose(_d4_mean(views), batch[0], rtol=1e-6)


def test_preprocess_normalises_and_binarises() -> None:
    torch = pytest.importorskip("torch")
    from models.unet_effb4_buildings.pipeline import preprocess

    batch = {
        "image": torch.full((2, 3, 64, 64), 255, dtype=torch.uint8),
        "mask": torch.full((2, 1, 64, 64), 3, dtype=torch.long),
    }
    images, masks = preprocess(batch)
    assert images.shape == (2, 3, 64, 64)
    assert images.dtype == torch.float32
    assert set(masks.unique().tolist()) <= {0, 1}, "mask must be clamped to binary"
    assert masks.shape == (2, 64, 64)


def test_stac_item_declares_instance_segmentation() -> None:
    item = json.loads((MODEL_DIR / "stac-item.json").read_text())
    assert item["properties"]["mlm:tasks"] == ["instance-segmentation"]


# ---------------------------------------------------------------------------
# Pipeline steps, exercised against the toy dataset from tests/conftest.py.
# ---------------------------------------------------------------------------

TOY_HYPERPARAMETERS: dict = {
    "val_ratio": 0.5,
    "split_seed": 42,
    "block_size": 1,
    "samples_per_epoch": 4,
    "batch_size": 2,
    "epochs": 1,
    "chip_size": 64,
    "freeze_encoder": True,
}


def test_split_dataset(generate_toy_dataset: dict[str, Path]) -> None:
    from models.unet_effb4_buildings.pipeline import split_dataset

    info = split_dataset.entrypoint(
        dataset_chips={"image": str(generate_toy_dataset["chips"])},
        dataset_labels=str(generate_toy_dataset["labels"]),
        hyperparameters=TOY_HYPERPARAMETERS,
    )
    assert info["strategy"] == "spatial_block"
    assert 0.0 < info["val_ratio"] < 1.0
    assert info["train_count"] > 0
    assert info["val_count"] > 0


def _train_one_epoch(toy: dict[str, Path]):
    from models.unet_effb4_buildings.pipeline import split_dataset, train_model

    info = split_dataset.entrypoint(
        dataset_chips={"image": str(toy["chips"])},
        dataset_labels=str(toy["labels"]),
        hyperparameters=TOY_HYPERPARAMETERS,
    )
    model = train_model.entrypoint(
        dataset_chips={"image": str(toy["chips"])},
        dataset_labels=str(toy["labels"]),
        base_model_weights="",
        hyperparameters=TOY_HYPERPARAMETERS,
        split_info=info,
        num_classes=4,
    )
    return model, info


def test_train_model(generate_toy_dataset: dict[str, Path]) -> None:
    pytest.importorskip("torch")
    model, _ = _train_one_epoch(generate_toy_dataset)
    assert model is not None
    assert hasattr(model, "parameters")
    assert next(model.parameters()).device.type == "cpu"


def test_evaluate_model(generate_toy_dataset: dict[str, Path]) -> None:
    pytest.importorskip("torch")
    from models.unet_effb4_buildings.pipeline import evaluate_model

    model, info = _train_one_epoch(generate_toy_dataset)
    metrics = evaluate_model.entrypoint(
        trained_model=model,
        dataset_chips={"image": str(generate_toy_dataset["chips"])},
        dataset_labels=str(generate_toy_dataset["labels"]),
        hyperparameters=TOY_HYPERPARAMETERS,
        split_info=info,
        num_classes=4,
    )
    assert isinstance(metrics, dict) and metrics
    for name, value in metrics.items():
        if isinstance(value, float) and ("iou" in name or "accuracy" in name or "f1" in name):
            assert 0.0 <= value <= 1.0, f"{name} out of range: {value}"


def test_export_onnx(generate_toy_dataset: dict[str, Path]) -> None:
    """The exported graph must name every dynamic axis, or a serving layer
    cannot tell height from batch."""
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("torch")
    from models.unet_effb4_buildings.pipeline import export_onnx

    model, _ = _train_one_epoch(generate_toy_dataset)
    blob = export_onnx.entrypoint(
        trained_model=model,
        hyperparameters=TOY_HYPERPARAMETERS,
        num_classes=4,
    )
    assert isinstance(blob, bytes) and len(blob) > 0

    proto = onnx.load_from_string(blob)
    onnx.checker.check_model(proto)
    out = proto.graph.output[0].type.tensor_type.shape.dim
    named = [d.dim_param for d in out if d.dim_param]
    assert len(named) == len(set(named)), f"dynamic axes must be distinct, got {named}"
