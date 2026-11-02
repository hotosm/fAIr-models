"""Step tests for dinov3-damage on toy pre/post chips; the base checkpoint and encoder come from Hugging Face."""

import io
import json
from pathlib import Path
from typing import Any

import pytest

from fair.params import training_params
from models.dinov3_damage.pipeline import DAMAGE_CLASS_COUNT

TRAINING_HYPERPARAMETERS = training_params(
    json.loads((Path(__file__).parents[1] / "stac-item.json").read_text())["properties"]["mlm:hyperparameters"]
)


@pytest.fixture(scope="session")
def pretrained_weights() -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id="hotosm/earthquake-damage-assessment-model", filename="model.ckpt")


@pytest.fixture
def split_info(toy_chips: Path, toy_labels: Path) -> dict[str, Any]:
    from models.dinov3_damage.pipeline import split_dataset

    return split_dataset.entrypoint(
        dataset_chips={"post": str(toy_chips)},
        dataset_labels=str(toy_labels),
        hyperparameters={**TRAINING_HYPERPARAMETERS, "val_ratio": 0.3},
    )


def test_split_dataset(split_info: dict[str, Any]) -> None:
    train, val = split_info["train_chip_names"], split_info["val_chip_names"]
    assert (split_info["train_count"], split_info["val_count"]) == (len(train), len(val)) == (6, 2)
    assert set(train).isdisjoint(val)


def test_split_dataset_keeps_each_tile_block_in_one_split_and_skips_unlabelled_blocks(
    tmp_path: Path, toy_chips: Path, toy_labels: Path
) -> None:
    from dda.pipeline.chip_windows import parse_chip_name

    from models.dinov3_damage.pipeline import split_dataset

    for chip in toy_chips.glob("OAM-*.tif"):
        x, y, zoom = parse_chip_name(chip.name)
        for dx in (0, 1):
            for dy in (0, 1):
                (tmp_path / f"OAM-{x + dx}-{y + dy}-{zoom}.tif").touch()
    unlabelled = {f"OAM-{x}-{y}-18.tif" for x in (10, 11) for y in (40, 41)}
    for name in unlabelled:
        (tmp_path / name).touch()
    info = split_dataset.entrypoint(
        dataset_chips={"post": str(tmp_path)},
        dataset_labels=str(toy_labels),
        hyperparameters={**TRAINING_HYPERPARAMETERS, "val_ratio": 0.3},
    )

    val_blocks = {int(name.split("-")[1]) // 2 for name in info["val_chip_names"]}
    assert info["val_count"] == 4 * len(val_blocks)
    assert all(int(name.split("-")[1]) // 2 not in val_blocks for name in info["train_chip_names"])
    assert unlabelled.isdisjoint(info["train_chip_names"] + info["val_chip_names"])


def test_split_dataset_rejects_labels_outside_epsg_4326(tmp_path: Path, toy_chips: Path, toy_labels: Path) -> None:
    import geopandas as gpd

    from models.dinov3_damage.pipeline import split_dataset

    gpd.read_file(next(toy_labels.glob("*.geojson"))).to_crs(3857).to_file(tmp_path / "labels.geojson")
    with pytest.raises(ValueError, match="EPSG:4326"):
        split_dataset.entrypoint(
            dataset_chips={"post": str(toy_chips)},
            dataset_labels=str(tmp_path),
            hyperparameters=TRAINING_HYPERPARAMETERS,
        )


@pytest.mark.parametrize("seed", range(10))
def test_split_dataset_puts_each_damage_class_in_both_splits(toy_chips: Path, toy_labels: Path, seed: int) -> None:
    import geopandas as gpd
    import mercantile
    from dda.pipeline.chip_windows import parse_chip_name
    from shapely.geometry import box

    from models.dinov3_damage.pipeline import split_dataset

    info = split_dataset.entrypoint(
        dataset_chips={"post": str(toy_chips)},
        dataset_labels=str(toy_labels),
        hyperparameters={**TRAINING_HYPERPARAMETERS, "val_ratio": 0.3, "split_seed": seed},
    )
    labels = gpd.read_file(next(toy_labels.glob("*.geojson")))

    def classes(chip_names: list[str]) -> set[int]:
        tiles = [box(*mercantile.bounds(*parse_chip_name(name))) for name in chip_names]
        return {int(row.label) for row in labels.itertuples() if any(tile.contains(row.geometry) for tile in tiles)}

    assert classes(info["train_chip_names"]) == classes(info["val_chip_names"]) == {1, 4}


def test_train_model(
    toy_chips: Path, toy_labels: Path, toy_pre_chips: Path, pretrained_weights: str, split_info: dict[str, Any]
) -> None:
    import torch

    from models.dinov3_damage.pipeline import train_model

    checkpoint = train_model.entrypoint(
        dataset_chips={"post": str(toy_chips), "pre": str(toy_pre_chips)},
        dataset_labels=str(toy_labels),
        base_model_weights=pretrained_weights,
        hyperparameters={**TRAINING_HYPERPARAMETERS, "epochs": 1, "batch_size": 2},
        split_info=split_info,
        num_classes=DAMAGE_CLASS_COUNT,
    )
    state = torch.load(io.BytesIO(checkpoint), map_location="cpu", weights_only=False)
    assert any(key.startswith("net.dmg_head") for key in state["state_dict"])


def test_train_model_rejects_a_class_count_other_than_four(
    toy_chips: Path, toy_labels: Path, toy_pre_chips: Path, split_info: dict[str, Any]
) -> None:
    from models.dinov3_damage.pipeline import train_model

    with pytest.raises(ValueError, match="num_classes=2"):
        train_model.entrypoint(
            dataset_chips={"post": str(toy_chips), "pre": str(toy_pre_chips)},
            dataset_labels=str(toy_labels),
            base_model_weights="unused.ckpt",
            hyperparameters={**TRAINING_HYPERPARAMETERS, "epochs": 1},
            split_info=split_info,
            num_classes=2,
        )


@pytest.mark.parametrize(("delta", "ships"), [(-0.1, "pretrained"), (0.1, "fine-tuned")])
def test_train_model_ships_the_fine_tuned_checkpoint_only_when_val_f1_improves(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    toy_chips: Path,
    toy_labels: Path,
    toy_pre_chips: Path,
    split_info: dict[str, Any],
    delta: float,
    ships: str,
) -> None:
    from types import SimpleNamespace

    from dda.pipeline import fewshot_damage

    from models.dinov3_damage.pipeline import train_model

    pretrained, fine_tuned = tmp_path / "pretrained.ckpt", tmp_path / "fine-tuned.ckpt"
    pretrained.write_bytes(b"pretrained")
    fine_tuned.write_bytes(b"fine-tuned")
    result = SimpleNamespace(delta=delta, best_ckpt=fine_tuned, to_dict=dict)
    monkeypatch.setattr(fewshot_damage, "fit_damage_windows", lambda *args, **kwargs: result)

    checkpoint = train_model.entrypoint(
        dataset_chips={"post": str(toy_chips), "pre": str(toy_pre_chips)},
        dataset_labels=str(toy_labels),
        base_model_weights=str(pretrained),
        hyperparameters={**TRAINING_HYPERPARAMETERS, "epochs": 1},
        split_info=split_info,
        num_classes=DAMAGE_CLASS_COUNT,
    )

    assert checkpoint == ships.encode()


def test_preprocess_returns_one_normalised_chip(toy_chips: Path) -> None:
    import numpy as np
    from dda.config import DINOV3_MEAN, DINOV3_STD

    from models.dinov3_damage.pipeline import preprocess

    tensor = preprocess(next(toy_chips.glob("OAM-*.tif")))

    assert tensor.shape == (1, 3, 256, 256)
    assert tensor.dtype == np.float32
    for channel, (mean, std) in enumerate(zip(DINOV3_MEAN, DINOV3_STD, strict=True)):
        assert tensor[0, channel].max() == pytest.approx((1.0 - mean) / std, abs=1e-5)
        assert tensor[0, channel].min() == pytest.approx((1 / 255 - mean) / std, abs=1e-5)


def test_evaluate_model(
    pretrained_weights: str, toy_chips: Path, toy_labels: Path, toy_pre_chips: Path, split_info: dict[str, Any]
) -> None:
    from models.dinov3_damage.pipeline import evaluate_model

    metrics = evaluate_model.entrypoint(
        trained_model=Path(pretrained_weights).read_bytes(),
        dataset_chips={"post": str(toy_chips), "pre": str(toy_pre_chips)},
        dataset_labels=str(toy_labels),
        hyperparameters={**TRAINING_HYPERPARAMETERS, "batch_size": 2},
        split_info=split_info,
    )
    assert set(metrics) == {"dmg_macro_f1", "dmg_f1", "destroyed_f1", "loc_iou"}
    assert all(0.0 <= value <= 1.0 for value in metrics.values())


def test_export_onnx(pretrained_weights: str) -> None:
    import onnx

    from models.dinov3_damage.pipeline import export_onnx

    onnx_bytes = export_onnx.entrypoint(trained_model=Path(pretrained_weights).read_bytes(), hyperparameters={})
    graph = onnx.load_from_string(onnx_bytes).graph
    assert [node.name for node in graph.input] == ["post", "pre"]
    assert [node.name for node in graph.output] == ["damage_logits"]
