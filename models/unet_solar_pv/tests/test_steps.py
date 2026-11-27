import json
from pathlib import Path
from typing import Any

import pytest
from shapely.geometry import box, shape
from shapely.ops import unary_union

from .conftest import EAST, NORTH, SOUTH, WEST, write_chip

HYPERPARAMETERS: dict[str, Any] = {
    "epochs": 1,
    "batch_size": 2,
    "learning_rate": 1e-4,
    "val_ratio": 0.25,
    "split_seed": 42,
    "block_size": 1,
}


def _split(dataset: dict[str, Path]) -> dict[str, Any]:
    from models.unet_solar_pv.pipeline import split_dataset

    return split_dataset.entrypoint(
        dataset_chips={"image": str(dataset["chips"])},
        dataset_labels=str(dataset["labels"]),
        hyperparameters=HYPERPARAMETERS,
    )


def _train(dataset: dict[str, Path], checkpoint: Path, split_info: dict[str, Any]) -> bytes:
    from models.unet_solar_pv.pipeline import train_model

    return train_model.entrypoint(
        dataset_chips={"image": str(dataset["chips"])},
        dataset_labels=str(dataset["labels"]),
        base_model_weights=str(checkpoint),
        hyperparameters=HYPERPARAMETERS,
        split_info=split_info,
        num_classes=2,
    )


def test_split_dataset(generate_toy_dataset: dict[str, Path]) -> None:
    info = _split(generate_toy_dataset)
    chips = {p.name for p in generate_toy_dataset["chips"].glob("*.tif")}

    assert info["train_count"] == 3
    assert info["val_count"] == 1
    assert set(info["train_chip_names"]).isdisjoint(info["val_chip_names"])
    assert set(info["train_chip_names"]) | set(info["val_chip_names"]) == chips


def test_spatial_split_keeps_neighbouring_chips_together() -> None:
    from models.unet_solar_pv.pipeline import _spatial_split

    # Two 2 x 2 tile blocks, far apart, of four chips each.
    names = [f"OAM-{x}-{y}-19.tif" for x0 in (100, 200) for x in (x0, x0 + 1) for y in (300, 301)]
    train, val = _spatial_split(names, val_ratio=0.5, seed=1, block_size=2)

    assert len(train) == len(val) == 4
    assert len({name.split("-")[1][0] for name in val}) == 1


def test_labels_burn_onto_the_blue_half(generate_toy_dataset: dict[str, Path]) -> None:
    from models.unet_solar_pv.pipeline import _label_geometries, _load_chips

    names = sorted(p.name for p in generate_toy_dataset["chips"].glob("*.tif"))
    images, masks = _load_chips(
        str(generate_toy_dataset["chips"]), names, _label_geometries(str(generate_toy_dataset["labels"]))
    )

    assert images.shape == (4, 3, 256, 256)
    # Each label covers the blue left half of its chip, and only that half.
    assert (masks[:, 0, :, :128] == 1).all()
    assert (masks[:, 0, :, 128:] == 0).all()


def test_train_model(generate_toy_dataset: dict[str, Path], base_checkpoint: Path) -> None:
    import torch

    from models.unet_solar_pv.pipeline import _load_model

    trained = _load_model(_train(generate_toy_dataset, base_checkpoint, _split(generate_toy_dataset)))
    base = _load_model(str(base_checkpoint))

    trained_weights, base_weights = trained.state_dict(), base.state_dict()
    assert trained_weights.keys() == base_weights.keys()
    assert any(not torch.equal(trained_weights[k], base_weights[k]) for k in base_weights)


def test_evaluate_model(generate_toy_dataset: dict[str, Path], base_checkpoint: Path, model_dir: Path) -> None:
    from models.unet_solar_pv.pipeline import evaluate_model

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


def test_export_onnx(base_checkpoint: Path, panel_chips: Path) -> None:
    import numpy as np
    import torch
    from onnxruntime import InferenceSession

    from models.unet_solar_pv.pipeline import _load_model, export_onnx, preprocess

    weights = base_checkpoint.read_bytes()
    session = InferenceSession(export_onnx.entrypoint(trained_model=weights), providers=["CPUExecutionProvider"])
    chip = preprocess(panel_chips / "OAM-0-0.tif")
    logits = np.asarray(session.run(None, {"image": chip})[0])

    assert [i.name for i in session.get_inputs()] == ["image"]
    assert logits.shape == (1, 1, 256, 256)
    with torch.no_grad():
        expected = _load_model(weights).eval()(torch.from_numpy(chip)).numpy()
    assert np.abs(logits - expected).max() < 1e-3


def test_run_inference(panel_chips: Path, toy_model: Path) -> None:
    from models.unet_solar_pv.pipeline import run_inference

    result = run_inference.entrypoint(
        model_uri=str(toy_model), input_images={"image": str(panel_chips)}, inference_params={}
    )

    panels = unary_union([shape(f["geometry"]) for f in result["features"]])
    left_half = box(WEST, SOUTH, (WEST + EAST) / 2, NORTH)
    assert {f["properties"]["label"] for f in result["features"]} == {"solar_panel"}
    assert panels.symmetric_difference(left_half).area < 0.01 * left_half.area


# Toy panel pixels score sigmoid(4 * 224 / 255 - 2), about 0.82, and background pixels about 0.18.
@pytest.mark.parametrize(("threshold", "keeps_panels"), [(0.8, True), (0.85, False)])
def test_confidence_threshold_applies_to_the_sigmoid_panel_probability(
    panel_chips: Path, toy_model: Path, threshold: float, keeps_panels: bool
) -> None:
    from models.unet_solar_pv.pipeline import run_inference

    result = run_inference.entrypoint(
        model_uri=str(toy_model),
        input_images={"image": str(panel_chips)},
        inference_params={"confidence_threshold": threshold},
    )

    assert bool(result["features"]) is keeps_panels


def test_preprocess_rejects_a_chip_the_model_cannot_take(tmp_path: Path) -> None:
    from models.unet_solar_pv.pipeline import preprocess

    chip = tmp_path / "small.tif"
    write_chip(chip, pixels=128)

    with pytest.raises(ValueError, match="128x128"):
        preprocess(chip)
