from pathlib import Path

import pytest
from shapely.geometry import box, shape
from shapely.ops import unary_union

from .conftest import EAST, NORTH, SOUTH, WEST, write_chip


def test_run_inference(tree_chips: Path, toy_model: Path) -> None:
    from models.unet_tree_segmentation.pipeline import run_inference

    result = run_inference.entrypoint(
        model_uri=str(toy_model), input_images={"image": str(tree_chips)}, inference_params={}
    )

    trees = unary_union([shape(f["geometry"]) for f in result["features"]])
    left_half = box(WEST, SOUTH, (WEST + EAST) / 2, NORTH)
    assert {f["properties"]["label"] for f in result["features"]} == {"tree"}
    assert trees.symmetric_difference(left_half).area < 0.01 * left_half.area


# Toy tree pixels score a two-class softmax of sigmoid(2 * (4 * 224 / 255 - 2)), about 0.95; a sigmoid of
# the tree logit alone would score about 0.82.
@pytest.mark.parametrize(("threshold", "keeps_trees"), [(0.9, True), (0.97, False)])
def test_confidence_threshold_applies_to_the_softmax_tree_probability(
    tree_chips: Path, toy_model: Path, threshold: float, keeps_trees: bool
) -> None:
    from models.unet_tree_segmentation.pipeline import run_inference

    result = run_inference.entrypoint(
        model_uri=str(toy_model),
        input_images={"image": str(tree_chips)},
        inference_params={"confidence_threshold": threshold},
    )

    assert bool(result["features"]) is keeps_trees


def test_preprocess_rejects_a_chip_the_model_cannot_take(tmp_path: Path) -> None:
    from models.unet_tree_segmentation.pipeline import preprocess

    chip = tmp_path / "small.tif"
    write_chip(chip, pixels=128)

    with pytest.raises(ValueError, match="128x128"):
        preprocess(chip)
