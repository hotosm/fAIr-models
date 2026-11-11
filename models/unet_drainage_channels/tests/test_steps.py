import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from shapely.geometry import LineString, shape

from .conftest import DRAIN_COLUMNS, NO_DATA_FROM, NORTH, SOUTH, drain_longitude, write_chip

HYPERPARAMETERS: dict[str, Any] = {
    "epochs": 2,
    "batch_size": 2,
    "learning_rate": 1e-4,
    "val_ratio": 0.25,
    "split_seed": 42,
    "block_size": 1,
}
# Ground size of a toy pixel in metres.
TOY_GSD = 0.43


def _split(dataset: dict[str, Path]) -> dict[str, Any]:
    from models.unet_drainage_channels.pipeline import split_dataset

    return split_dataset.entrypoint(
        dataset_chips={"image": str(dataset["chips"])},
        dataset_labels=str(dataset["labels"]),
        hyperparameters=HYPERPARAMETERS,
    )


def _train(dataset: dict[str, Path], checkpoint: Path, split_info: dict[str, Any], num_classes: int = 2) -> bytes:
    from models.unet_drainage_channels.pipeline import train_model

    return train_model.entrypoint(
        dataset_chips={"image": str(dataset["chips"])},
        dataset_labels=str(dataset["labels"]),
        base_model_weights=str(checkpoint),
        hyperparameters=HYPERPARAMETERS,
        split_info=split_info,
        num_classes=num_classes,
    )


def _strip(rows: slice, cols: slice) -> np.ndarray:
    mask = np.zeros((256, 256), bool)
    mask[rows, cols] = True
    return mask


def test_split_dataset(generate_toy_dataset: dict[str, Path]) -> None:
    info = _split(generate_toy_dataset)
    chips = {p.name for p in generate_toy_dataset["chips"].glob("*.tif")}

    assert info["train_count"] == 3
    assert info["val_count"] == 1
    assert set(info["train_chip_names"]).isdisjoint(info["val_chip_names"])
    assert set(info["train_chip_names"]) | set(info["val_chip_names"]) == chips


def test_spatial_split_keeps_neighbouring_chips_together() -> None:
    from models.unet_drainage_channels.pipeline import _spatial_split

    # Two 2 x 2 tile blocks, far apart, of four chips each, every chip with the same length of drain.
    names = [f"OAM-{x}-{y}-20.tif" for x0 in (100, 200) for x in (x0, x0 + 1) for y in (300, 301)]
    train, val = _spatial_split(names, np.ones((len(names), 1)), val_ratio=0.5, seed=1, block_size=2)

    assert len(train) == len(val) == 4
    assert len({name.split("-")[1][0] for name in val}) == 1


def test_spatial_split_holds_out_its_share_of_drain_length() -> None:
    from models.unet_drainage_channels.pipeline import _spatial_split

    # Four 2 x 2 tile blocks; the last holds no drain, so holding it out would score nothing.
    names = [f"OAM-{x}-{y}-20.tif" for x0 in (100, 200, 300, 400) for x in (x0, x0 + 1) for y in (300, 301)]
    drains = np.array([[0.0 if name.startswith("OAM-4") else 1.0] for name in names])
    _, val = _spatial_split(names, drains, val_ratio=0.25, seed=3, block_size=2)

    assert len(val) == 4
    assert not val[0].startswith("OAM-4")


@pytest.mark.parametrize(
    ("tags", "expected"),
    [({"width": "2.5"}, 2.5), ({"width": "0.6 m"}, 0.6), ({"width": 40}, 10.0), ({"width": "wide"}, 1.0), ({}, 1.0)],
)
def test_width_comes_from_the_tag_or_the_default(tags: dict[str, Any], expected: float) -> None:
    from models.unet_drainage_channels.pipeline import _width_m

    assert _width_m(tags) == expected


def test_only_drain_lines_are_labels_and_culverts_are_covered(generate_toy_dataset: dict[str, Path]) -> None:
    from models.unet_drainage_channels.pipeline import _drain_labels

    labels = _drain_labels(str(generate_toy_dataset["labels"]))

    # Four drains and four culverts; the stream is left out.
    assert len(labels) == 8
    assert sorted((width, covered) for _, width, covered in labels) == [(2.0, True)] * 4 + [(7.0, False)] * 4


def test_labels_burn_as_drain_margin_and_ignore(generate_toy_dataset: dict[str, Path]) -> None:
    from models.unet_drainage_channels.pipeline import IGNORE, _drain_labels, _load_chips

    names = sorted(p.name for p in generate_toy_dataset["chips"].glob("*.tif"))
    images, masks, centres, skips, gsds = _load_chips(
        str(generate_toy_dataset["chips"]), names, _drain_labels(str(generate_toy_dataset["labels"]))
    )
    first, last = DRAIN_COLUMNS
    middle = (first + last) // 2

    assert images.shape == (4, 3, 256, 256)
    assert np.allclose(gsds, TOY_GSD, atol=0.01)
    # The drain at its 7 m width, a 1 m margin on each side of it, then background.
    assert (masks[:, :, first + 1 : last - 1] == 1).all()
    assert (masks[:, :, first - 2] == IGNORE).all()
    assert (masks[:, :, last + 1] == IGNORE).all()
    assert (masks[:, 0, first - 6 : first - 4] == 0).all()
    assert centres[:, :, middle - 1 : middle + 1].any(axis=2).all()
    assert centres.sum() <= 4 * 2 * 256
    # The culvert on the west side is ignored and is no line to find; a predicted line there is not an error.
    assert (masks[:, 128, :40] == IGNORE).all()
    assert not centres[:, :, :40].any()
    assert skips[:, 128, :40].all()
    # No data, and the 2 m beside it, is ignored too.
    assert (masks[:, :, NO_DATA_FROM - 4 :] == IGNORE).all()
    assert skips[:, :, NO_DATA_FROM - 4 :].all()
    assert not skips[:, 0, : NO_DATA_FROM - 6].any()


def test_drain_weight_grows_with_rarity_up_to_the_cap() -> None:
    from models.unet_drainage_channels.pipeline import IGNORE, _drain_weight

    assert _drain_weight(np.repeat(np.array([0, 1, IGNORE], np.uint8), [400, 100, 1000]), max_weight=10.0) == 2.0
    assert _drain_weight(np.repeat(np.array([0, 1], np.uint8), [1000, 1]), max_weight=10.0) == 10.0
    assert _drain_weight(np.zeros(10, np.uint8), max_weight=10.0) == pytest.approx(10**0.5)


def test_slide_moves_a_label_onto_the_drain_the_model_sees() -> None:
    import torch

    from models.unet_drainage_channels.pipeline import _slide_labels

    # Two chips labelled with a drain down columns 100 to 107. The model sees the first chip's drain four
    # pixels to the east and six pixels south of where its label ends; it sees the second where it is labelled.
    target = torch.zeros((2, 256, 256), dtype=torch.int64)
    target[:, 40:200, 100:108] = 1
    logits = torch.full((2, 1, 256, 256), -6.0)
    logits[0, 0, 46:206, 104:112] = 6.0
    logits[1, 0, 40:200, 100:108] = 6.0

    moved, shifts = _slide_labels(logits, target, radius=12, drain_weight=5.0)

    assert shifts == [(6, 4), (0, 0)]
    assert torch.equal(moved[0] == 1, logits[0, 0] > 0)
    assert torch.equal(moved[1], target[1])


def test_slide_radius_is_whole_steps_within_the_tolerance() -> None:
    from models.unet_drainage_channels.pipeline import _slide_radius

    assert _slide_radius(0.148, 2.0) == 12
    assert _slide_radius(0.3, 2.0) == 6
    assert _slide_radius(0.148, 0.0) == 0


def test_clean_drops_specks_and_the_edge_of_the_image() -> None:
    from models.unet_drainage_channels.pipeline import _clean

    mask = _strip(slice(0, 256), slice(100, 110))
    # A speck of 4 pixels is 0.09 square metres at 0.15 m per pixel; the strip is far larger.
    mask[10:12, 10:12] = True
    no_data = _strip(slice(0, 128), slice(0, 256))

    cleaned = _clean(mask, no_data, gsd=0.15)

    assert not cleaned[:12, :12].any()
    # 2 m is 13.3 pixels, so the strip is cut back to 14 pixels below the no-data edge.
    assert not cleaned[: 128 + 13].any()
    assert cleaned[128 + 14 :, 100:110].all()


def test_centrelines_follow_a_strip_from_edge_to_edge() -> None:
    from models.unet_drainage_channels.pipeline import _centrelines

    lines = _centrelines(_strip(slice(120, 136), slice(0, 256)), gsd=0.15, min_length_m=2.0)

    assert len(lines) == 1
    xs, ys = np.asarray(lines[0].coords).T
    assert (xs.min(), xs.max()) == (0.5, 255.5)
    assert np.abs(ys - 128).max() <= 1


def test_centrelines_meet_at_a_junction_and_drop_short_side_branches() -> None:
    from models.unet_drainage_channels.pipeline import _centrelines

    cross = _strip(slice(120, 136), slice(0, 256)) | _strip(slice(0, 256), slice(100, 116))
    # A bump on a strip thins to a side branch about 2.7 m long, from the middle of the strip to the bump's end.
    bump = _strip(slice(120, 136), slice(0, 256)) | _strip(slice(136, 146), slice(60, 70))

    arms = _centrelines(cross, gsd=0.15, min_length_m=2.0)
    assert len(arms) == 4
    assert len({end for line in arms for end in (line.coords[0], line.coords[-1])}) == 5
    # Kept, the branch splits the strip in two; dropped, the two halves join into one line again.
    assert len(_centrelines(bump, gsd=0.15, min_length_m=2.0)) == 3
    assert len(_centrelines(bump, gsd=0.15, min_length_m=3.0)) == 1
    assert _centrelines(_strip(slice(100, 104), slice(100, 104)), gsd=0.15, min_length_m=2.0) == []


def test_scores_are_perfect_on_the_label_and_fall_with_an_offset() -> None:
    from models.unet_drainage_channels.pipeline import _scores

    # At 0.5 m per pixel: a labelled drain down column 100, and lines traced on it, 3 pixels (1.5 m) and
    # 6 pixels (3 m) to its east.
    centre = _strip(slice(0, 256), slice(100, 101))[None]
    mask = np.where(_strip(slice(0, 256), slice(99, 102)), 1, 0).astype(np.uint8)[None]
    nowhere, gsds = np.zeros_like(centre), np.array([0.5])

    def scores(col: int) -> dict[str, float]:
        line = _strip(slice(0, 256), slice(col, col + 1))[None]
        return _scores(mask == 1 if col == 100 else line, line, mask, centre, nowhere, gsds)

    assert set(scores(100).values()) == {1.0}
    near, far = scores(103), scores(106)
    assert (near["completeness_1m"], near["correctness_1m"]) == (0.0, 0.0)
    assert (near["completeness_2m"], near["correctness_2m"], near["f1_2m"]) == (1.0, 1.0, 1.0)
    assert (far["completeness_2m"], far["correctness_2m"], far["pixel_iou"]) == (0.0, 0.0, 0.0)
    assert (far["completeness_4m"], far["correctness_4m"]) == (1.0, 1.0)


def test_scores_skip_lines_over_covered_drains() -> None:
    from models.unet_drainage_channels.pipeline import _scores

    centre = _strip(slice(0, 256), slice(100, 101))[None]
    mask = centre.astype(np.uint8)
    # A second line is traced far from the label, along a covered drain.
    line = centre | _strip(slice(0, 256), slice(200, 201))[None]
    covered = _strip(slice(0, 256), slice(195, 206))[None]

    assert _scores(line, line, mask, centre, covered, np.array([0.5]))["correctness_1m"] == 1.0
    assert _scores(line, line, mask, centre, np.zeros_like(centre), np.array([0.5]))["correctness_1m"] == 0.5


def test_train_model(generate_toy_dataset: dict[str, Path], base_checkpoint: Path) -> None:
    import torch

    from models.unet_drainage_channels.pipeline import _load_model

    trained = _load_model(_train(generate_toy_dataset, base_checkpoint, _split(generate_toy_dataset)))
    base = _load_model(str(base_checkpoint))

    trained_weights, base_weights = trained.state_dict(), base.state_dict()
    assert trained_weights.keys() == base_weights.keys()
    assert any(not torch.equal(trained_weights[k], base_weights[k]) for k in base_weights)


def test_train_model_rejects_another_class_count(generate_toy_dataset: dict[str, Path], base_checkpoint: Path) -> None:
    with pytest.raises(ValueError, match="num_classes=3"):
        _train(generate_toy_dataset, base_checkpoint, _split(generate_toy_dataset), num_classes=3)


def test_evaluate_model(generate_toy_dataset: dict[str, Path], base_checkpoint: Path, model_dir: Path) -> None:
    from models.unet_drainage_channels.pipeline import evaluate_model

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


def test_export_onnx(base_checkpoint: Path, drain_chips: Path) -> None:
    import torch
    from onnxruntime import InferenceSession

    from models.unet_drainage_channels.pipeline import _load_model, export_onnx, preprocess

    weights = base_checkpoint.read_bytes()
    session = InferenceSession(export_onnx.entrypoint(trained_model=weights), providers=["CPUExecutionProvider"])
    chip = preprocess(drain_chips / "OAM-0-0.tif")
    logits = np.asarray(session.run(None, {"image": chip})[0])

    assert [i.name for i in session.get_inputs()] == ["image"]
    assert logits.shape == (1, 1, 256, 256)
    with torch.no_grad():
        expected = _load_model(weights).eval()(torch.from_numpy(chip)).numpy()
    # Random weights give large logits, so allow float32 rounding relative to the largest one.
    assert np.abs(logits - expected).max() < 1e-3 * max(1.0, float(np.abs(expected).max()))


def test_run_inference_traces_the_drain_as_one_line(drain_chips: Path, toy_model: Path) -> None:
    from models.unet_drainage_channels.pipeline import run_inference

    result = run_inference.entrypoint(
        model_uri=str(toy_model), input_images={"image": str(drain_chips)}, inference_params={}
    )

    assert len(result["features"]) == 1
    feature = result["features"][0]
    # The toy drain's logit is 8 * 224 / 255 - 4, a probability of about 0.954.
    assert feature["properties"] == {"class": 1, "label": "drain", "score": 0.954}
    assert feature["geometry"]["type"] == "LineString"
    middle = drain_longitude()
    drain = LineString([(middle, SOUTH), (middle, NORTH)])
    line = shape(feature["geometry"])
    # Within a pixel (0.000004 degrees) of the middle of the drain, from the top of the chip to the bottom.
    assert line.hausdorff_distance(drain) < 0.000008
    assert line.length == pytest.approx(drain.length, rel=0.01)


@pytest.mark.parametrize(("threshold", "keeps_drain"), [(0.95, True), (0.96, False)])
def test_confidence_threshold_applies_to_the_drain_probability(
    drain_chips: Path, toy_model: Path, threshold: float, keeps_drain: bool
) -> None:
    from models.unet_drainage_channels.pipeline import run_inference

    result = run_inference.entrypoint(
        model_uri=str(toy_model),
        input_images={"image": str(drain_chips)},
        inference_params={"confidence_threshold": threshold},
    )

    assert bool(result["features"]) is keeps_drain


def test_a_threshold_of_zero_still_returns_lines_inside_the_image(drain_chips: Path, toy_model: Path) -> None:
    from models.unet_drainage_channels.pipeline import run_inference

    # fAIr's integration test predicts at threshold 0, where every pixel of the image is drain.
    result = run_inference.entrypoint(
        model_uri=str(toy_model), input_images={"image": str(drain_chips)}, inference_params={"confidence_threshold": 0}
    )

    assert result["features"]
    assert {feature["geometry"]["type"] for feature in result["features"]} == {"LineString"}
    no_data_west = drain_longitude() + (NO_DATA_FROM - sum(DRAIN_COLUMNS) / 2) / 256 * 0.001
    assert all(lon < no_data_west for f in result["features"] for lon, _ in f["geometry"]["coordinates"])


def test_min_length_drops_short_lines(drain_chips: Path, toy_model: Path) -> None:
    from models.unet_drainage_channels.pipeline import run_inference

    # The toy drain is about 110 m long.
    result = run_inference.entrypoint(
        model_uri=str(toy_model), input_images={"image": str(drain_chips)}, inference_params={"min_length_m": 150}
    )

    assert result["features"] == []


def test_preprocess_rejects_a_chip_the_model_cannot_take(tmp_path: Path) -> None:
    from models.unet_drainage_channels.pipeline import preprocess

    chip = tmp_path / "small.tif"
    write_chip(chip, pixels=128)

    with pytest.raises(ValueError, match="128x128"):
        preprocess(chip)
