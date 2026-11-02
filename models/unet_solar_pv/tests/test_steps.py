from pathlib import Path

import pytest
from shapely.geometry import box, shape
from shapely.ops import unary_union

from .conftest import EAST, NORTH, SOUTH, WEST, write_chip


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
