import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from shapely.geometry import box, shape
from shapely.ops import unary_union

from .conftest import NORTH, SOUTH, STEP_DEG, STRIPE, STRIPES, WEST, write_chip

HYPERPARAMETERS: dict[str, Any] = {
    "epochs": 1,
    "batch_size": 2,
    "learning_rate": 1e-4,
    "val_ratio": 0.25,
    "split_seed": 42,
    "block_size": 1,
}


def _split(dataset: dict[str, Path]) -> dict[str, Any]:
    from models.unet_roof_material.pipeline import split_dataset

    return split_dataset.entrypoint(
        dataset_chips={"image": str(dataset["chips"])},
        dataset_labels=str(dataset["labels"]),
        hyperparameters=HYPERPARAMETERS,
    )


def _train(dataset: dict[str, Path], checkpoint: Path, split_info: dict[str, Any], num_classes: int = 4) -> bytes:
    from models.unet_roof_material.pipeline import train_model

    return train_model.entrypoint(
        dataset_chips={"image": str(dataset["chips"])},
        dataset_labels=str(dataset["labels"]),
        base_model_weights=str(checkpoint),
        hyperparameters=HYPERPARAMETERS,
        split_info=split_info,
        num_classes=num_classes,
    )


def test_split_dataset(generate_toy_dataset: dict[str, Path]) -> None:
    info = _split(generate_toy_dataset)
    chips = {p.name for p in generate_toy_dataset["chips"].glob("*.tif")}

    assert info["train_count"] == 3
    assert info["val_count"] == 1
    assert set(info["train_chip_names"]).isdisjoint(info["val_chip_names"])
    assert set(info["train_chip_names"]) | set(info["val_chip_names"]) == chips


def test_spatial_split_keeps_neighbouring_chips_together() -> None:
    from models.unet_roof_material.pipeline import _spatial_split

    # Two 2 x 2 tile blocks, far apart, of four chips each, every chip with one metal sheet roof.
    names = [f"OAM-{x}-{y}-20.tif" for x0 in (100, 200) for x in (x0, x0 + 1) for y in (300, 301)]
    roofs = np.tile([1, 0, 0], (len(names), 1))
    train, val = _spatial_split(names, roofs, val_ratio=0.5, seed=1, block_size=2)

    assert len(train) == len(val) == 4
    assert len({name.split("-")[1][0] for name in val}) == 1


def test_spatial_split_holds_out_the_rare_class() -> None:
    from models.unet_roof_material.pipeline import _spatial_split

    # Four 2 x 2 tile blocks with metal sheet everywhere and other roofs in the first three only.
    names = [f"OAM-{x}-{y}-20.tif" for x0 in (100, 200, 300, 400) for x in (x0, x0 + 1) for y in (300, 301)]
    roofs = np.array([[1, 0, 1 if not name.startswith("OAM-4") else 0] for name in names])
    _, val = _spatial_split(names, roofs, val_ratio=0.25, seed=3, block_size=2)

    assert len(val) == 4
    assert not val[0].startswith("OAM-4")


@pytest.mark.parametrize(
    ("properties", "expected"),
    [
        ({"label": 2}, 2),
        ({"label": "3"}, 3),
        ({"roof:material": "metal_sheet"}, 1),
        ({"roof_material": "concrete"}, 3),
        ({"label": 1, "tags": {"roof:material": "thatch"}}, 2),
        ({"tags": {"roof:material": "roof_tiles"}}, 255),
        ({"label": 0}, 255),
        ({}, 255),
    ],
)
def test_roof_class_prefers_the_material_tag_then_the_label(properties: dict[str, Any], expected: int) -> None:
    from models.unet_roof_material.pipeline import _roof_class

    assert _roof_class(properties) == expected


def test_labels_burn_by_class_and_unknown_roofs_are_ignored(generate_toy_dataset: dict[str, Path]) -> None:
    from models.unet_roof_material.pipeline import IGNORE, _load_chips, _roof_labels

    names = sorted(p.name for p in generate_toy_dataset["chips"].glob("*.tif"))
    images, masks = _load_chips(
        str(generate_toy_dataset["chips"]), names, _roof_labels(str(generate_toy_dataset["labels"]))
    )

    assert images.shape == (4, 3, 256, 256)
    assert masks.shape == (4, 256, 256)
    for value in (1, 2, 3):
        assert (masks[:, :, (value - 1) * STRIPE : value * STRIPE] == value).all()
    assert (masks[:, :128, 3 * STRIPE :] == IGNORE).all()
    assert (masks[:, 128:, 3 * STRIPE :] == 0).all()


def test_class_weights_grow_with_rarity_up_to_the_cap() -> None:
    from models.unet_roof_material.pipeline import IGNORE, _class_weights

    masks = np.repeat(np.array([0, 1, 2, 3, IGNORE], np.uint8), [400, 100, 25, 1, 1000])

    assert _class_weights(masks, max_weight=5.0) == [1.0, 2.0, 4.0, 5.0]


def test_train_model(generate_toy_dataset: dict[str, Path], base_checkpoint: Path) -> None:
    import torch

    from models.unet_roof_material.pipeline import _load_model

    trained = _load_model(_train(generate_toy_dataset, base_checkpoint, _split(generate_toy_dataset)))
    base = _load_model(str(base_checkpoint))

    trained_weights, base_weights = trained.state_dict(), base.state_dict()
    assert trained_weights.keys() == base_weights.keys()
    assert any(not torch.equal(trained_weights[k], base_weights[k]) for k in base_weights)


def test_train_model_rejects_another_class_count(generate_toy_dataset: dict[str, Path], base_checkpoint: Path) -> None:
    with pytest.raises(ValueError, match="num_classes=2"):
        _train(generate_toy_dataset, base_checkpoint, _split(generate_toy_dataset), num_classes=2)


def test_evaluate_model(generate_toy_dataset: dict[str, Path], base_checkpoint: Path, model_dir: Path) -> None:
    from models.unet_roof_material.pipeline import evaluate_model

    split_info = _split(generate_toy_dataset)
    metrics = evaluate_model.entrypoint(
        trained_model=_train(generate_toy_dataset, base_checkpoint, split_info),
        dataset_chips={"image": str(generate_toy_dataset["chips"])},
        dataset_labels=str(generate_toy_dataset["labels"]),
        hyperparameters=HYPERPARAMETERS,
        split_info=split_info,
    )

    spec = json.loads((model_dir / "stac-item.json").read_text())["properties"]["fair:metrics_spec"]
    assert set(metrics) == {entry["key"] for entry in spec}
    assert all(0.0 <= value <= 1.0 for value in metrics.values())


def test_export_onnx(base_checkpoint: Path, roof_chips: Path) -> None:
    import torch
    from onnxruntime import InferenceSession

    from models.unet_roof_material.pipeline import _load_model, export_onnx, preprocess

    weights = base_checkpoint.read_bytes()
    session = InferenceSession(export_onnx.entrypoint(trained_model=weights), providers=["CPUExecutionProvider"])
    chip = preprocess(roof_chips / "OAM-0-0.tif")
    logits = np.asarray(session.run(None, {"image": chip})[0])

    assert [i.name for i in session.get_inputs()] == ["image"]
    assert logits.shape == (1, 4, 256, 256)
    with torch.no_grad():
        expected = _load_model(weights).eval()(torch.from_numpy(chip)).numpy()
    # Random weights give logits of about 10, so allow float32 rounding relative to the largest one.
    assert np.abs(logits - expected).max() < 1e-3 * max(1.0, float(np.abs(expected).max()))


def test_run_inference(roof_chips: Path, toy_model: Path) -> None:
    from models.unet_roof_material.pipeline import run_inference

    result = run_inference.entrypoint(
        model_uri=str(toy_model), input_images={"image": str(roof_chips)}, inference_params={}
    )

    stripe_deg = STEP_DEG / len(STRIPES)
    for value, label in enumerate(["metal_sheet", "thatch", "other_roof"], start=1):
        found = unary_union([shape(f["geometry"]) for f in result["features"] if f["properties"]["label"] == label])
        stripe = box(WEST + (value - 1) * stripe_deg, SOUTH, WEST + value * stripe_deg, NORTH)
        assert {f["properties"]["class"] for f in result["features"] if f["properties"]["label"] == label} == {value}
        assert found.symmetric_difference(stripe).area < 0.01 * stripe.area


# A toy roof pixel's class scores softmax(0, 1.51, -1.50, -1.50), so its winning probability is about 0.76.
@pytest.mark.parametrize(("threshold", "keeps_roofs"), [(0.75, True), (0.77, False)])
def test_confidence_threshold_applies_to_the_winning_class_probability(
    roof_chips: Path, toy_model: Path, threshold: float, keeps_roofs: bool
) -> None:
    from models.unet_roof_material.pipeline import run_inference

    result = run_inference.entrypoint(
        model_uri=str(toy_model),
        input_images={"image": str(roof_chips)},
        inference_params={"confidence_threshold": threshold},
    )

    assert bool(result["features"]) is keeps_roofs


def test_preprocess_rejects_a_chip_the_model_cannot_take(tmp_path: Path) -> None:
    from models.unet_roof_material.pipeline import preprocess

    chip = tmp_path / "small.tif"
    write_chip(chip, pixels=128)

    with pytest.raises(ValueError, match="128x128"):
        preprocess(chip)
