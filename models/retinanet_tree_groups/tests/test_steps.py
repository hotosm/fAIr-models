"""Step and serving tests for retinanet-tree-groups.

Each test runs the real @step entrypoint or serving function on the toy OAM chips and OSM labels. The start
weights are a random two-class RetinaNet, so the tests are offline and the 129 MB base checkpoint is never
downloaded. Accuracy is not under test. Telemetry sinks (zenml/mlflow) are no-ops via models/conftest.py.
"""

import io
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pystac
import pytest
import torch
from fair_trees.evaluate import MIN_PICK_GT, PRIOR_THRESHOLD
from fair_trees.georef import parse_oam_name, tile_bounds_4326
from fair_trees.params import CLASSES, THRESHOLDS_KEY, WINDOWS_KEY, parse_inference_params
from fair_trees.train import build_model, checkpoint_num_classes

from fair.params import inference_params
from fair.stac.validators import validate_predictions_geojson

REPO_ROOT = Path(__file__).parents[3]
ITEM = json.loads((Path(__file__).parents[1] / "stac-item.json").read_text())
METRIC_KEYS = {entry["key"] for entry in ITEM["properties"]["fair:metrics_spec"]}
SPLIT_INFO_KEYS = {"strategy", "val_ratio", "seed", "train_count", "val_count", "description"}
BASE_WINDOWS = {"window_m": 60.0, "group_window_m": 120.0}
DEFAULT_THRESHOLD = ITEM["properties"]["mlm:hyperparameters"]["inference.confidence_threshold"]
TREES_PER_CHIP, GROUPS_PER_CHIP = 3, 1  # see conftest.py
# Under MIN_PICK_GT val boxes the library picks its prior for a two-class model; the toy val set is that small.
PICKED_THRESHOLD = PRIOR_THRESHOLD[2]
# The random model scores every box near 0.02: below the item's default threshold, above 0.
EVERY_BOX = {"confidence_threshold": 0.0}
SMOKE_TEST_PARAMS = {"confidence_threshold": 0.5, "iou_threshold": 0.45, "min_class_value": 1}  # scripts/ sends these
# Not an item parameter: at window overlap 0.125 the library's 0.05 band drops tree boxes lying across a seam.
EDGE_BAND = 0.02


@pytest.fixture(scope="session")
def random_checkpoint() -> bytes:
    torch.manual_seed(0)
    buffer = io.BytesIO()
    torch.save(build_model(None, trainable_layers=0, num_classes=2).state_dict(), buffer)
    return buffer.getvalue()


@pytest.fixture(scope="session")
def base_weights(tmp_path_factory: pytest.TempPathFactory, random_checkpoint: bytes) -> str:
    path = tmp_path_factory.mktemp("weights") / "random_init.pt"
    path.write_bytes(random_checkpoint)
    return str(path)


@pytest.fixture(scope="session")
def onnx_model(tmp_path_factory: pytest.TempPathFactory, random_checkpoint: bytes) -> Path:
    from models.retinanet_tree_groups.pipeline import export_onnx

    # The model's own thresholds are 0, so any prediction that keeps nothing was served at another threshold.
    onnx_bytes = export_onnx.entrypoint(
        trained_model=random_checkpoint,
        hyperparameters={},
        serving_metadata={"thresholds": {"tree": 0.0, "tree_group": 0.0}, "windows": BASE_WINDOWS},
    )
    path = tmp_path_factory.mktemp("onnx") / "model.onnx"
    path.write_bytes(onnx_bytes)
    return path


@pytest.fixture(scope="session")
def session(onnx_model: Path) -> Any:
    import onnxruntime as ort

    return ort.InferenceSession(str(onnx_model), providers=["CPUExecutionProvider"])


@pytest.fixture(scope="session")
def one_chip(tmp_path_factory: pytest.TempPathFactory, toy_chips: Path) -> Path:
    """A directory with one toy chip: one window per prediction, for tests that need features, not many."""
    chips = tmp_path_factory.mktemp("one_chip")
    shutil.copy(sorted(toy_chips.glob("*.tif"))[0], chips)
    return chips


@pytest.fixture()
def library_calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[Path, dict[str, Any]]]:
    """The chips path and params each predict call hands the library; the library itself is not run."""
    calls: list[tuple[Path, dict[str, Any]]] = []

    def predict_session(session: Any, input_images: Path, params: dict[str, Any], bbox: Any = None) -> dict[str, Any]:
        calls.append((input_images, params))
        return {"type": "FeatureCollection", "features": []}

    monkeypatch.setattr("fair_trees.serve.predict_session", predict_session)
    return calls


# Stand-in network outputs: one confident anchor per box, decoding to that box exactly.
STUB_LOGIT = 6.0  # sigmoid 0.9975, above any threshold the tests serve at
FALSE_GROUP_PX = (10.0, 200.0, 50.0, 250.0)  # a group box no label matches, clear of every tree's centre


def _stub_outputs(
    input_hw: tuple[int, int], boxes_per_class: tuple[np.ndarray, ...], window: Any, scale: Any
) -> tuple[np.ndarray, np.ndarray]:
    """`cls_logits` and `bbox_regression` for one window whose decode gives exactly `boxes_per_class` (mosaic px)."""
    from fair_trees.decode import make_anchors
    from fair_trees.evaluate import iou_matrix

    anchors = np.concatenate(make_anchors(*input_hw))
    cls = np.full((1, len(anchors), len(boxes_per_class)), -20.0, np.float32)
    reg = np.zeros((1, len(anchors), 4), np.float32)
    shift = np.array([window.x, window.y, window.x, window.y], np.float32)
    factor = np.array([scale.scale_x, scale.scale_y, scale.scale_x, scale.scale_y], np.float32)
    used: set[int] = set()
    for class_index, boxes in enumerate(boxes_per_class):
        for box in boxes:
            b = (np.asarray(box, np.float32) - shift) * factor
            i = next(int(k) for k in np.argsort(-iou_matrix(b, anchors)[0]) if int(k) not in used)
            used.add(i)
            a = anchors[i]
            aw, ah, bw, bh = a[2] - a[0], a[3] - a[1], b[2] - b[0], b[3] - b[1]
            reg[0, i] = [
                (b[0] + bw / 2 - a[0] - aw / 2) / aw,
                (b[1] + bh / 2 - a[1] - ah / 2) / ah,
                np.log(bw / aw),
                np.log(bh / ah),
            ]
            cls[0, i, class_index] = STUB_LOGIT
    return cls, reg


@pytest.fixture(scope="session")
def stub_run_fn(toy_chips: Path, toy_labels: Path) -> Any:
    """A network stand-in for the toy chips: every labelled tree and tree group, one more tree centred in each
    group, and one group box no label matches. Keyed by the network input, so it serves any window of the
    serving grids."""
    from fair_trees.dataset import load_fair_dataset
    from fair_trees.evaluate import region_units
    from fair_trees.preprocess import preprocess_window
    from fair_trees.serve import _grid

    from models.retinanet_tree_groups import pipeline

    regions = load_fair_dataset(sorted(toy_chips.glob("*.tif")), next(toy_labels.glob("*.geojson")))
    groups = {u.mosaic.rgb.tobytes(): u.gt_boxes for u in region_units(regions, label=2)}
    params = pipeline._serving_params()
    outputs: dict[bytes, tuple[np.ndarray, np.ndarray]] = {}
    for unit in region_units(regions, label=1):
        mosaic, trees, group = unit.mosaic, unit.gt_boxes, groups[unit.mosaic.rgb.tobytes()]
        side = trees[0, 2] - trees[0, 0]
        cx, cy = (group[0, 0] + group[0, 2]) / 2, (group[0, 1] + group[0, 3]) / 2
        in_group = [cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2]
        boxes = (np.vstack([trees, [in_group]]), np.vstack([group, [FALSE_GROUP_PX]]))
        for window_m in (params.window_m, params.group_window_m):
            for w in _grid(mosaic, params, window_m):
                batch, scale = preprocess_window(mosaic.rgb[w.y : w.y + w.height, w.x : w.x + w.width])
                outputs[batch.tobytes()] = _stub_outputs(batch.shape[2:], boxes, w, scale)

    def run(batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return outputs[np.ascontiguousarray(batch, np.float32).tobytes()]

    return run


class _StubSession:
    """The onnxruntime session surface fair_trees.serve uses, answering with `stub_run_fn`."""

    def __init__(self, run: Any) -> None:
        self._run = run

    def get_inputs(self) -> list[Any]:
        return [type("Input", (), {"name": "images"})()]

    def get_modelmeta(self) -> Any:
        return type("Meta", (), {"custom_metadata_map": {WINDOWS_KEY: json.dumps(BASE_WINDOWS)}})()

    def run(self, names: list[str], feeds: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        return self._run(feeds["images"])


def _split(chips: Path, labels: Path, hyperparameters: dict[str, Any]) -> dict[str, Any]:
    from models.retinanet_tree_groups.pipeline import split_dataset

    return split_dataset.entrypoint(
        dataset_chips={"image": str(chips)}, dataset_labels=str(labels), hyperparameters=hyperparameters
    )


def _block(name: str) -> tuple[int, int]:
    x, y, _ = parse_oam_name(name)
    return x // 4, y // 4


def _first_lon_lat(feature: dict[str, Any]) -> list[float]:
    coordinates = feature["geometry"]["coordinates"]
    return coordinates if feature["geometry"]["type"] == "Point" else coordinates[0][0]


def _state(checkpoint: bytes) -> dict[str, torch.Tensor]:
    return torch.load(io.BytesIO(checkpoint), map_location="cpu", weights_only=True)


def test_split_dataset(toy_chips: Path, toy_labels: Path, base_hyperparameters: dict[str, Any]) -> None:
    info = _split(toy_chips, toy_labels, base_hyperparameters)

    assert info.keys() >= SPLIT_INFO_KEYS
    assert info["strategy"] == "spatial"
    assert info["train_count"] > 0
    assert info["val_count"] > 0
    assert info["train_count"] + info["val_count"] == len(list(toy_chips.glob("*.tif")))
    # The share the split gave, not the requested one: whole blocks move, so the two differ.
    assert info["val_ratio"] == info["val_count"] / (info["train_count"] + info["val_count"])
    assert info["val_ratio"] != base_hyperparameters.get("val_ratio", 0.2)
    assert {_block(n) for n in info["train_chip_names"]}.isdisjoint(_block(n) for n in info["val_chip_names"])
    assert _split(toy_chips, toy_labels, base_hyperparameters) == info


def test_split_dataset_counts_only_oam_named_chips(
    toy_chips: Path, toy_labels: Path, base_hyperparameters: dict[str, Any], tmp_path: Path
) -> None:
    chips = tmp_path / "chips"
    shutil.copytree(toy_chips, chips)
    shutil.copyfile(next(toy_chips.glob("*.tif")), chips / "extra.tif")

    info = _split(chips, toy_labels, base_hyperparameters)

    # The library does not read a chip whose name is not OAM-x-y-z.
    assert info["train_count"] + info["val_count"] == len(list(toy_chips.glob("*.tif")))
    assert "extra.tif" not in info["train_chip_names"] + info["val_chip_names"]


def test_train_model(
    toy_chips: Path,
    toy_labels: Path,
    base_weights: str,
    random_checkpoint: bytes,
    base_hyperparameters: dict[str, Any],
) -> None:
    from models.retinanet_tree_groups.pipeline import train_model

    split_info = _split(toy_chips, toy_labels, base_hyperparameters)
    checkpoint, serving_metadata = train_model.entrypoint(
        dataset_chips={"image": str(toy_chips)},
        dataset_labels=str(toy_labels),
        base_model_weights=base_weights,
        hyperparameters=base_hyperparameters,
        split_info=split_info,
        num_classes=2,
    )

    assert isinstance(checkpoint, bytes)
    trained, base = _state(checkpoint), _state(random_checkpoint)
    assert trained.keys() == base.keys()
    assert checkpoint_num_classes(trained) == 2
    # trainable_layers 0 (the item's default) keeps the backbone frozen; the heads train.
    assert torch.equal(trained["backbone.body.layer4.2.conv3.weight"], base["backbone.body.layer4.2.conv3.weight"])
    assert not torch.equal(
        trained["head.classification_head.cls_logits.weight"], base["head.classification_head.cls_logits.weight"]
    )
    assert serving_metadata["windows"] == BASE_WINDOWS
    # Both classes are scored on val, so both carry the library's pick, not the pack's default.
    assert TREES_PER_CHIP * split_info["val_count"] < MIN_PICK_GT
    assert PICKED_THRESHOLD != DEFAULT_THRESHOLD
    assert serving_metadata["thresholds"] == {"tree": PICKED_THRESHOLD, "tree_group": PICKED_THRESHOLD}


def test_train_model_validates_at_the_serving_settings(
    toy_chips: Path,
    toy_labels: Path,
    base_weights: str,
    random_checkpoint: bytes,
    base_hyperparameters: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from models.retinanet_tree_groups import pipeline

    calls: list[dict[str, Any]] = []
    best = tmp_path / "best.pt"
    best.write_bytes(random_checkpoint)

    def fine_tune(*args: Any, **kwargs: Any) -> Any:
        calls.append(kwargs)
        return type("Result", (), {"best_checkpoint": best})()

    monkeypatch.setattr("fair_trees.train.fine_tune", fine_tune)
    pipeline.train_model.entrypoint(
        dataset_chips={"image": str(toy_chips)},
        dataset_labels=str(toy_labels),
        base_model_weights=base_weights,
        hyperparameters=base_hyperparameters,
        split_info=_split(toy_chips, toy_labels, base_hyperparameters),
        num_classes=2,
    )

    # Without it the library validates, picks thresholds and stops early at its own defaults (overlap 0.5).
    [kwargs] = calls
    assert kwargs["val_params"] == pipeline._serving_params()


def test_schedules_are_the_items_scheduler_values() -> None:
    from fair_trees.params import parse_train_params

    from models.retinanet_tree_groups import pipeline

    spec = {entry["key"]: entry for entry in ITEM["properties"]["fair:hyperparameters_spec"]}
    assert set(pipeline.SCHEDULES) == set(spec["scheduler"]["values"])
    for schedule in pipeline.SCHEDULES.values():
        assert parse_train_params({"schedule": schedule}).schedule == schedule


@pytest.mark.parametrize(("scheduler", "schedule"), [(None, "cosine"), ("cosine", "cosine"), ("none", "constant")])
def test_fine_tune_gets_the_librarys_schedule_name(scheduler: str | None, schedule: str) -> None:
    from models.retinanet_tree_groups import pipeline

    given = {} if scheduler is None else {"scheduler": scheduler}
    params = pipeline._fine_tune_params(pipeline._hyperparameters(given), num_classes=2)

    # The item uses the contribution guide's name and values; the library calls the same setting schedule.
    assert params["schedule"] == schedule and "scheduler" not in params


def test_train_model_on_tree_only_data_keeps_tree_group(
    toy_chips: Path, toy_labels: Path, base_weights: str, tmp_path: Path
) -> None:
    from models.retinanet_tree_groups.pipeline import train_model

    collection = json.loads((toy_labels / "labels.geojson").read_text())
    collection["features"] = [f for f in collection["features"] if f["properties"]["tags"]["natural"] == "tree"]
    labels = tmp_path / "labels.geojson"
    labels.write_text(json.dumps(collection))
    # The overrides fAIr's CI sends, with one epoch.
    hyperparameters = {"epochs": 1, "batch_size": 2, "sample_fraction": 0.01, "samples_per_epoch": 2}

    checkpoint, serving_metadata = train_model.entrypoint(
        dataset_chips={"image": str(toy_chips)},
        dataset_labels=str(labels),
        base_model_weights=base_weights,
        hyperparameters=hyperparameters,
        split_info=_split(toy_chips, labels, hyperparameters),
        num_classes=2,
    )

    assert checkpoint_num_classes(_state(checkpoint)) == 2
    # Validation cannot pick a tree_group threshold without its labels: the model carries the default.
    assert serving_metadata["thresholds"] == {"tree": PICKED_THRESHOLD, "tree_group": DEFAULT_THRESHOLD}
    assert serving_metadata["windows"] == BASE_WINDOWS


def test_train_model_without_val_chips(
    toy_chips: Path, toy_labels: Path, base_weights: str, base_hyperparameters: dict[str, Any]
) -> None:
    from models.retinanet_tree_groups.pipeline import train_model

    hyperparameters = {**base_hyperparameters, "batch_size": 2, "samples_per_epoch": 2}
    split_info = {**_split(toy_chips, toy_labels, hyperparameters), "val_count": 0, "val_chip_names": []}

    checkpoint, serving_metadata = train_model.entrypoint(
        dataset_chips={"image": str(toy_chips)},
        dataset_labels=str(toy_labels),
        base_model_weights=base_weights,
        hyperparameters=hyperparameters,
        split_info=split_info,
        num_classes=2,
    )

    assert checkpoint_num_classes(_state(checkpoint)) == 2
    # Nothing to pick thresholds on: the model still carries a threshold per class and its windows.
    assert serving_metadata == {
        "thresholds": {"tree": DEFAULT_THRESHOLD, "tree_group": DEFAULT_THRESHOLD},
        "windows": BASE_WINDOWS,
    }


def test_train_model_reads_only_the_kept_val_chips(toy_chips: Path, toy_labels: Path, base_weights: str) -> None:
    from models.retinanet_tree_groups.pipeline import train_model

    hyperparameters = {"epochs": 1, "batch_size": 2, "samples_per_epoch": 2, "sample_fraction": 0.01}
    split_info = _split(toy_chips, toy_labels, hyperparameters)
    # Sorts after the real val chips, so sample_fraction drops it; fine_tune reads every val chip it is given.
    absent = "OAM-485799-499508-20.tif"
    assert not (toy_chips / absent).exists() and sorted([*split_info["val_chip_names"], absent])[-1] == absent
    split_info["val_chip_names"] = [*split_info["val_chip_names"], absent]

    checkpoint, _ = train_model.entrypoint(
        dataset_chips={"image": str(toy_chips)},
        dataset_labels=str(toy_labels),
        base_model_weights=base_weights,
        hyperparameters=hyperparameters,
        split_info=split_info,
        num_classes=2,
    )

    assert checkpoint_num_classes(_state(checkpoint)) == 2


def test_train_model_never_thins_the_train_chips(
    toy_chips: Path, toy_labels: Path, base_weights: str, tmp_path: Path
) -> None:
    from models.retinanet_tree_groups.pipeline import train_model

    hyperparameters = {"epochs": 1, "batch_size": 2, "samples_per_epoch": 2, "sample_fraction": 0.01}
    split_info = _split(toy_chips, toy_labels, hyperparameters)
    # Labels on the last train chip only: the library refuses training if that chip is left out.
    west, south, east, north = tile_bounds_4326(*parse_oam_name(sorted(split_info["train_chip_names"])[-1]))
    collection = json.loads((toy_labels / "labels.geojson").read_text())
    collection["features"] = [
        f for f in collection["features"] if west < _first_lon_lat(f)[0] < east and south < _first_lon_lat(f)[1] < north
    ]
    assert collection["features"]
    labels = tmp_path / "labels.geojson"
    labels.write_text(json.dumps(collection))

    checkpoint, _ = train_model.entrypoint(
        dataset_chips={"image": str(toy_chips)},
        dataset_labels=str(labels),
        base_model_weights=base_weights,
        hyperparameters=hyperparameters,
        split_info=split_info,
        num_classes=2,
    )

    assert checkpoint_num_classes(_state(checkpoint)) == 2


def test_evaluate_model(
    toy_chips: Path, toy_labels: Path, random_checkpoint: bytes, base_hyperparameters: dict[str, Any]
) -> None:
    from models.retinanet_tree_groups.pipeline import evaluate_model

    split_info = _split(toy_chips, toy_labels, base_hyperparameters)
    metrics = evaluate_model.entrypoint(
        trained_model=random_checkpoint,
        dataset_chips={"image": str(toy_chips)},
        dataset_labels=str(toy_labels),
        hyperparameters=base_hyperparameters,
        split_info=split_info,
        class_names=["tree", "tree_group"],
    )

    assert set(metrics) == METRIC_KEYS
    for key, value in metrics.items():
        if key.startswith("val_"):
            assert type(value) is int, key
        else:
            assert type(value) is float and 0.0 <= value <= 1.0, key
    # Scored on the val chips only.
    assert metrics["val_trees"] == TREES_PER_CHIP * split_info["val_count"]
    assert metrics["val_tree_groups"] == GROUPS_PER_CHIP * split_info["val_count"]


def test_evaluate_model_scores_each_class_against_its_own_labels(
    toy_chips: Path,
    toy_labels: Path,
    random_checkpoint: bytes,
    base_hyperparameters: dict[str, Any],
    stub_run_fn: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from models.retinanet_tree_groups.pipeline import evaluate_model

    monkeypatch.setattr("fair_trees.train.torch_run_fn", lambda model, device: stub_run_fn)
    split_info = _split(toy_chips, toy_labels, base_hyperparameters)
    metrics = evaluate_model.entrypoint(
        trained_model=random_checkpoint,
        dataset_chips={"image": str(toy_chips)},
        dataset_labels=str(toy_labels),
        hyperparameters=base_hyperparameters,
        split_info=split_info,
    )

    # Every labelled tree is found, and the extra tree lies in a labelled group, where trees are not scored.
    assert (metrics["precision_tree"], metrics["recall_tree"], metrics["f1_tree"]) == (1.0, 1.0, 1.0)
    # Every labelled group is found, next to one false group per chip.
    assert (metrics["precision_tree_group"], metrics["recall_tree_group"]) == (0.5, 1.0)
    assert metrics["f1_tree_group"] == round(2 / 3, 4)
    assert metrics["val_trees"] == TREES_PER_CHIP * split_info["val_count"]
    assert metrics["val_tree_groups"] == GROUPS_PER_CHIP * split_info["val_count"]


def test_evaluate_model_sample_fraction_keeps_at_least_one_val_chip(
    toy_chips: Path, toy_labels: Path, random_checkpoint: bytes
) -> None:
    from models.retinanet_tree_groups.pipeline import evaluate_model

    hyperparameters = {"epochs": 1, "sample_fraction": 0.01}
    metrics = evaluate_model.entrypoint(
        trained_model=random_checkpoint,
        dataset_chips={"image": str(toy_chips)},
        dataset_labels=str(toy_labels),
        hyperparameters=hyperparameters,
        split_info=_split(toy_chips, toy_labels, hyperparameters),
    )

    assert metrics["val_trees"] == TREES_PER_CHIP
    assert metrics["val_tree_groups"] == GROUPS_PER_CHIP


def test_evaluate_model_without_val_chips_reports_zeros(
    toy_chips: Path, toy_labels: Path, random_checkpoint: bytes, base_hyperparameters: dict[str, Any]
) -> None:
    from models.retinanet_tree_groups.pipeline import evaluate_model

    split_info = {**_split(toy_chips, toy_labels, base_hyperparameters), "val_count": 0, "val_chip_names": []}
    metrics = evaluate_model.entrypoint(
        trained_model=random_checkpoint,
        dataset_chips={"image": str(toy_chips)},
        dataset_labels=str(toy_labels),
        hyperparameters=base_hyperparameters,
        split_info=split_info,
    )

    assert metrics == dict.fromkeys(METRIC_KEYS, 0)
    assert all(type(value) is (int if key.startswith("val_") else float) for key, value in metrics.items())


def test_export_onnx(random_checkpoint: bytes, base_hyperparameters: dict[str, Any]) -> None:
    import onnx
    import onnxruntime as ort

    from models.retinanet_tree_groups.pipeline import export_onnx

    thresholds = {"tree": 0.2, "tree_group": 0.15}
    onnx_bytes = export_onnx.entrypoint(
        trained_model=random_checkpoint,
        hyperparameters=base_hyperparameters,
        serving_metadata={"thresholds": thresholds, "windows": BASE_WINDOWS},
    )

    assert isinstance(onnx_bytes, bytes)
    proto = onnx.load_from_string(onnx_bytes)
    onnx.checker.check_model(proto)
    metadata = {entry.key: json.loads(entry.value) for entry in proto.metadata_props}
    assert metadata[THRESHOLDS_KEY] == thresholds
    assert metadata[WINDOWS_KEY] == BASE_WINDOWS

    session = ort.InferenceSession(onnx_bytes, providers=["CPUExecutionProvider"])
    outputs = session.run(["cls_logits", "bbox_regression"], {"images": np.zeros((1, 3, 128, 160), np.float32)})
    cls_logits, bbox_regression = (np.asarray(output) for output in outputs)
    assert cls_logits.shape[0] == 1 and cls_logits.shape[2] == 2
    assert bbox_regression.shape == (1, cls_logits.shape[1], 4)


def test_scoring_settings_are_the_items_inference_defaults() -> None:
    from models.retinanet_tree_groups import pipeline

    defaults = ITEM["properties"]["mlm:hyperparameters"]
    # Field by field, what validation and evaluate_model run with.
    params = pipeline._serving_params()
    assert params.confidence_threshold == defaults["inference.confidence_threshold"]
    assert params.window_overlap == defaults["inference.window_overlap"]
    assert params.group_merge == defaults["inference.group_merge"]
    assert {"window_m": params.window_m, "group_window_m": params.group_window_m} == BASE_WINDOWS
    assert params.edge_band == EDGE_BAND


def test_serving_defaults_are_the_items_inference_parameters() -> None:
    from models.retinanet_tree_groups import pipeline

    # predict sends the library these keys and no other, so a key the item declares later must be added here.
    assert inference_params(ITEM["properties"]["mlm:hyperparameters"]) == pipeline.INFERENCE_DEFAULTS


def test_run_inference(
    toy_chips: Path,
    onnx_model: Path,
    generate_toy_dataset: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from models.retinanet_tree_groups.pipeline import run_inference

    # load_session copies the model into a temp dir it never removes; keep that copy under tmp_path.
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    result = run_inference.entrypoint(
        model_uri=str(onnx_model), input_images={"image": str(toy_chips)}, inference_params=EVERY_BOX
    )

    assert validate_predictions_geojson(result, pystac.Item.from_dict(ITEM)) == []
    features = result["features"]
    assert {f["properties"]["class"] for f in features} == {1, 2}
    for feature in features:
        props = feature["properties"]
        assert props.keys() == {"class", "label", "confidence", "in_group"}
        assert type(props["class"]) is int and props["label"] == CLASSES[props["class"] - 1]
        assert type(props["confidence"]) is float and 0.0 <= props["confidence"] <= 1.0
        assert type(props["in_group"]) is (bool if props["class"] == 1 else type(None))
    # Longitude first, inside the toy chips; boxes clipped to a chip edge may differ from it in the last float digit.
    west, south, east, north = json.loads(generate_toy_dataset["dataset_stac_item"].read_text())["bbox"]
    margin = 1e-9  # degrees, about 0.1 mm
    for lon, lat in (point for f in features for point in f["geometry"]["coordinates"][0]):
        assert type(lon) is float and type(lat) is float
        assert west - margin <= lon <= east + margin and south - margin <= lat <= north + margin


def test_predict_keeps_a_threshold_of_zero_for_both_classes(session: Any, one_chip: Path) -> None:
    from models.retinanet_tree_groups.pipeline import predict

    result = predict(session, {"image": str(one_chip)}, EVERY_BOX)

    assert {f["properties"]["class"] for f in result["features"]} == {1, 2}


def test_predict_keeps_only_features_centred_in_the_bbox(session: Any, one_chip: Path) -> None:
    from models.retinanet_tree_groups.pipeline import predict

    # The west half of the chip, as fair.serve.base sends it: [west, south, east, north] in EPSG:4326.
    west, south, east, north = tile_bounds_4326(*parse_oam_name(next(one_chip.glob("*.tif")).name))
    bbox = [west, south, (west + east) / 2, north]
    everything = predict(session, {"image": str(one_chip)}, EVERY_BOX)["features"]
    inside = predict(session, {"image": str(one_chip)}, EVERY_BOX, bbox=bbox)["features"]

    assert 0 < len(inside) < len(everything)
    for feature in inside:
        ring = feature["geometry"]["coordinates"][0]
        lon, lat = (ring[0][0] + ring[2][0]) / 2, (ring[0][1] + ring[2][1]) / 2
        assert bbox[0] <= lon <= bbox[2] and bbox[1] <= lat <= bbox[3]


def test_predict_marks_the_trees_inside_a_predicted_group(stub_run_fn: Any, one_chip: Path) -> None:
    from models.retinanet_tree_groups.pipeline import predict

    features = predict(_StubSession(stub_run_fn), {"image": str(one_chip)}, {})["features"]

    trees = [f for f in features if f["properties"]["class"] == 1]
    groups = [f for f in features if f["properties"]["class"] == 2]
    assert len(trees) == TREES_PER_CHIP + 1 and len(groups) == GROUPS_PER_CHIP + 1
    assert {f["properties"]["label"] for f in trees} == {"tree"}
    assert {f["properties"]["label"] for f in groups} == {"tree_group"}
    assert all(f["properties"]["in_group"] is None for f in groups)
    [inside] = [f for f in trees if f["properties"]["in_group"] is True]
    assert sum(f["properties"]["in_group"] is False for f in trees) == TREES_PER_CHIP

    def centre(feature: dict[str, Any]) -> tuple[float, float]:
        ring = feature["geometry"]["coordinates"][0]
        return (ring[0][0] + ring[2][0]) / 2, (ring[0][1] + ring[2][1]) / 2

    lon, lat = centre(inside)
    assert any(
        min(p[0] for p in ring) < lon < max(p[0] for p in ring)
        and min(p[1] for p in ring) < lat < max(p[1] for p in ring)
        for ring in (g["geometry"]["coordinates"][0] for g in groups)
    )


def test_predict_serves_the_items_threshold_not_the_models(session: Any, one_chip: Path) -> None:
    from models.retinanet_tree_groups.pipeline import predict

    # The model's own thresholds (0) would keep every box.
    assert predict(session, {"image": str(one_chip)}, {})["features"] == []


@pytest.mark.parametrize(
    ("params", "expected"),
    [
        pytest.param({}, {}, id="nothing"),
        pytest.param({"confidence_threshold": None}, {}, id="none"),
        pytest.param({"group_merge": ""}, {}, id="empty"),
        pytest.param({"confidence_threshold": 0.0}, {"confidence_threshold": 0.0}, id="zero"),
        pytest.param(
            {"confidence_threshold": 0.3, "window_overlap": 0.5, "group_merge": "join"},
            {"confidence_threshold": 0.3, "window_overlap": 0.5, "group_merge": "join"},
            id="all",
        ),
        pytest.param(SMOKE_TEST_PARAMS, {"confidence_threshold": 0.5}, id="smoke-test"),
    ],
)
def test_predict_fills_the_items_defaults(
    params: dict[str, Any],
    expected: dict[str, Any],
    library_calls: list[tuple[Path, dict[str, Any]]],
    toy_chips: Path,
) -> None:
    from models.retinanet_tree_groups.pipeline import predict

    predict(None, {"image": str(toy_chips)}, params)

    defaults = inference_params(ITEM["properties"]["mlm:hyperparameters"])
    # No window sizes: the library takes them from the model. The edge band is the pack's, whatever the caller sends.
    assert library_calls == [(toy_chips, {**defaults, **expected, "edge_band": EDGE_BAND})]


def test_inference_limits_are_the_items() -> None:
    from models.retinanet_tree_groups import pipeline

    spec = {entry["key"]: entry for entry in ITEM["properties"]["fair:hyperparameters_spec"]}
    for key, (low, high) in pipeline.INFERENCE_RANGES.items():
        assert (spec[key]["min"], spec[key]["max"]) == (low, high), key
    assert tuple(spec["group_merge"]["values"]) == pipeline.GROUP_MERGE_RULES
    assert {*pipeline.INFERENCE_RANGES, "group_merge"} == set(pipeline.INFERENCE_DEFAULTS)


@pytest.mark.parametrize(
    "params",
    [
        pytest.param({"window_overlap": 0.0, "confidence_threshold": 1.0}, id="low-and-high-ends"),
        pytest.param({"window_overlap": 0.5, "confidence_threshold": 0}, id="high-and-low-ends"),
        pytest.param({"group_merge": "drop_edge"}, id="declared-rule"),
    ],
)
def test_predict_keeps_values_inside_the_items_ranges(
    params: dict[str, Any], library_calls: list[tuple[Path, dict[str, Any]]], toy_chips: Path
) -> None:
    from models.retinanet_tree_groups.pipeline import predict

    predict(None, {"image": str(toy_chips)}, params)

    [(_, sent)] = library_calls
    assert {key: sent[key] for key in params} == params


@pytest.mark.parametrize(
    ("params", "message"),
    [
        pytest.param(
            {"window_overlap": 0.9}, r"window_overlap must be a number from 0 to 0\.5, got 0\.9", id="overlap-high"
        ),
        pytest.param({"window_overlap": -0.1}, r"window_overlap must be a number from 0 to 0\.5", id="overlap-low"),
        pytest.param(
            {"confidence_threshold": 1.5},
            r"confidence_threshold must be a number from 0 to 1, got 1\.5",
            id="threshold-high",
        ),
        pytest.param(
            {"confidence_threshold": float("nan")}, r"confidence_threshold must be a number", id="threshold-nan"
        ),
        pytest.param({"confidence_threshold": True}, r"confidence_threshold must be a number", id="threshold-bool"),
        pytest.param({"window_overlap": "wide"}, r"window_overlap must be a number", id="overlap-text"),
        pytest.param(
            {"group_merge": 0}, r"group_merge must be one of overlap, join, drop_edge, got 0", id="merge-zero"
        ),
        pytest.param({"group_merge": False}, r"group_merge must be one of .*, got False", id="merge-false"),
        pytest.param({"group_merge": "union"}, r"group_merge must be one of .*, got 'union'", id="merge-unknown"),
    ],
)
def test_predict_refuses_values_outside_the_items_ranges(
    params: dict[str, Any], message: str, library_calls: list[tuple[Path, dict[str, Any]]], toy_chips: Path
) -> None:
    from models.retinanet_tree_groups.pipeline import predict

    with pytest.raises(ValueError, match=message):
        predict(None, {"image": str(toy_chips)}, params)
    assert library_calls == []


def test_predict_drops_parameters_the_item_does_not_declare(
    library_calls: list[tuple[Path, dict[str, Any]]], toy_chips: Path, caplog: pytest.LogCaptureFixture
) -> None:
    from models.retinanet_tree_groups import pipeline

    with caplog.at_level(logging.WARNING, logger=pipeline.__name__):
        pipeline.predict(None, {"image": str(toy_chips)}, SMOKE_TEST_PARAMS)

    [(_, sent)] = library_calls
    # Served at the decode settings the metrics were scored at (the model's windows are 60 m and 120 m).
    served = replace(parse_inference_params(sent), **BASE_WINDOWS)
    assert served == replace(pipeline._serving_params(), confidence_threshold=SMOKE_TEST_PARAMS["confidence_threshold"])
    assert "iou_threshold" in caplog.text and "min_class_value" in caplog.text
    warnings = [r for r in caplog.records if r.name == pipeline.__name__ and r.levelno == logging.WARNING]
    assert len(warnings) == 1


def test_preprocess_gives_the_network_input_of_one_window(session: Any, one_chip: Path) -> None:
    from models.retinanet_tree_groups.pipeline import preprocess

    batch = preprocess(next(one_chip.glob("*.tif")))

    # A 256 px chip, resized to 800 px on its short side; 800 is a multiple of 32, so nothing is padded.
    assert batch.dtype == np.float32 and batch.shape == (1, 3, 800, 800)
    cls_logits = session.run(["cls_logits"], {"images": batch})[0]
    assert cls_logits.shape[0] == 1 and cls_logits.shape[2] == 2


_SERVE_WITHOUT_TORCH = """
import json
import sys
from pathlib import Path

# Training-only packages the inference image does not hold: importing one now raises ImportError.
sys.modules["torch"] = sys.modules["torchvision"] = sys.modules["onnx"] = None

from fair.serve.base import load_session
from models.retinanet_tree_groups.pipeline import predict, preprocess

model, chips = sys.argv[1:]
batch = preprocess(next(Path(chips).glob("*.tif")))
result = predict(load_session(model), {"image": chips}, {"confidence_threshold": 0.0})
print(json.dumps({"features": len(result["features"]), "shape": list(batch.shape)}))
"""


def test_serving_runs_without_torch(onnx_model: Path, one_chip: Path, tmp_path: Path) -> None:
    script = tmp_path / "serve_without_torch.py"
    script.write_text(_SERVE_WITHOUT_TORCH)

    # A fresh interpreter: this one has torch loaded already. zenml needs the script as a file.
    run = subprocess.run(
        [sys.executable, str(script), str(onnx_model), str(one_chip)],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT), "TMPDIR": str(tmp_path)},  # load_session's model copy
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )

    assert run.returncode == 0, run.stderr[-3000:]
    out = json.loads(run.stdout.strip().splitlines()[-1])
    assert out["features"] > 0 and out["shape"] == [1, 3, 800, 800]
