"""retinanet-tree-groups: a RetinaNet that finds trees and tree groups, fine-tuned, scored and exported by the
MIT-licensed `fair_trees` library; this module only adapts fAIr's step contract to it. Library and torch imports
stay inside the functions, so that importing this module needs neither."""

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Annotated, Any

from zenml import log_metadata, pipeline, step

from fair.params import training_params
from fair.stac.inputs import IMAGE_INPUT
from fair.utils.data import resolve_directory, resolve_path
from fair.zenml.instrumentation import log_evaluation_results, mlflow_training_context
from fair.zenml.materializers import CheckpointBytesMaterializer, ONNXMaterializer

logger = logging.getLogger(__name__)

STAC_ITEM = Path(__file__).with_name("stac-item.json")
# fAIr serves every model at the item's inference defaults, so validation and evaluate_model use them too.
CONFIDENCE_THRESHOLD = 0.175
WINDOW_OVERLAP = 0.125
GROUP_MERGE = "overlap"
# Share of the window side at a seam where tree boxes are dropped and group boxes merged. Not an item parameter:
# at overlap 0.125 the library's 0.05 drops tree boxes lying across a seam, and 0.02 wins most of them back.
EDGE_BAND = 0.02
# The item's inference parameters: the only keys predict passes on.
INFERENCE_DEFAULTS = {
    "confidence_threshold": CONFIDENCE_THRESHOLD,
    "window_overlap": WINDOW_OVERLAP,
    "group_merge": GROUP_MERGE,
}
# The item's allowed values of those keys (a test ties them to its fair:hyperparameters_spec); the library itself
# would accept window overlaps up to 0.9, tens of thousands of window passes for a live request.
INFERENCE_RANGES = {"confidence_threshold": (0.0, 1.0), "window_overlap": (0.0, 0.5)}
GROUP_MERGE_RULES = ("overlap", "join", "drop_edge")
# The base release's windows; validation picks a fine-tune's thresholds there.
WINDOWS_M = {"window_m": 60.0, "group_window_m": 120.0}
SPLIT_KEYS = ("val_ratio", "split_seed", "block_size")
PACK_KEYS = (*SPLIT_KEYS, "sample_fraction")  # applied here, not by fine_tune
# The item's `scheduler` values (the contribution guide's) -> the library's `schedule`.
SCHEDULES = {"cosine": "cosine", "none": "constant"}


def predict(
    session: Any, input_images: dict[str, str], params: dict[str, Any], bbox: list[float] | None = None
) -> dict[str, Any]:
    """Module-level predict for `fair.serve.base` and `run_inference`: `fair_trees.serve.predict_session` over the
    chip mosaic at the item's inference parameters, its features renamed by `postprocess`.

    The windows come from the model's metadata, and the caller's threshold (or the item's) serves both classes in
    place of the thresholds the model carries. `bbox` ([west, south, east, north] in EPSG:4326, what live serving
    sends) keeps the features whose box centre lies in it: the downloaded tiles reach past the requested area.
    """
    from fair_trees.serve import predict_session

    chips = resolve_directory(input_images[IMAGE_INPUT])
    return postprocess(predict_session(session, chips, _inference_params(params), bbox=bbox))


def preprocess(image_path: Any) -> Any:
    """An OAM chip as the network input of one window, (1, 3, H, W) float32: `fair_trees.preprocess.preprocess_window`,
    the transform serving applies to every window of the mosaic."""
    from fair_trees.preprocess import preprocess_window, read_chip

    rgb, _, _ = read_chip(image_path)
    return preprocess_window(rgb)[0]


def postprocess(raw_output: Any) -> Any:
    """The FeatureCollection `fair_trees` decodes from the network outputs -> the item's variables: `score` becomes
    `confidence`, the class gains its `label`, and tree groups get `in_group` null (the library sets it on trees)."""
    from fair_trees.params import CLASSES

    def variables(props: dict[str, Any]) -> dict[str, Any]:
        return {
            "class": props["class"],
            "label": CLASSES[props["class"] - 1],
            "confidence": props["score"],
            "in_group": props.get("in_group"),
        }

    features = [{**feature, "properties": variables(feature["properties"])} for feature in raw_output["features"]]
    return {**raw_output, "features": features}


def _inference_params(params: dict[str, Any] | None) -> dict[str, Any]:
    """The caller's value of each declared key, else the item's default, plus the pack's edge band; None and "" count
    as unset, 0 does not. A value outside the item's range or values raises ValueError, so nothing is served at a
    setting the item does not allow (the library would take a falsy `group_merge` as its own default, not the item's).

    Other keys are dropped, so live serving decodes, merges and edge-filters as evaluate_model scored (the CI smoke
    test sends `iou_threshold`, which the library would otherwise apply when merging boxes across windows).
    """
    given = params or {}
    if ignored := sorted(set(given) - set(INFERENCE_DEFAULTS)):
        logger.warning("Ignoring inference parameters the model does not declare: %s", ", ".join(ignored))
    served = {
        key: default if given.get(key) in (None, "") else _allowed(key, given[key])
        for key, default in INFERENCE_DEFAULTS.items()
    }
    return {**served, "edge_band": EDGE_BAND}


def _allowed(key: str, value: Any) -> Any:
    """`value` of the inference parameter `key` if the item allows it, numbers as float; else ValueError."""
    if key == "group_merge":
        if isinstance(value, str) and value in GROUP_MERGE_RULES:
            return value
        raise ValueError(f"group_merge must be one of {', '.join(GROUP_MERGE_RULES)}, got {value!r}")
    low, high = INFERENCE_RANGES[key]
    try:
        number = None if isinstance(value, bool) else float(value)
    except (TypeError, ValueError):
        number = None
    if number is None or not low <= number <= high:  # NaN fails the comparison
        raise ValueError(f"{key} must be a number from {low:g} to {high:g}, got {value!r}")
    return number


def _item() -> dict[str, Any]:
    return json.loads(STAC_ITEM.read_text())


def _hyperparameters(hyperparameters: dict[str, Any]) -> dict[str, Any]:
    """The caller's values over the item's training defaults: fAIr sends every declared key, a test only some."""
    return {**training_params(_item()["properties"]["mlm:hyperparameters"]), **hyperparameters}


def _device() -> str:
    if os.environ.get("FAIR_FORCE_CPU", "").lower() in {"1", "true", "yes", "on"}:
        return "cpu"
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _serving_params() -> Any:
    from fair_trees.params import InferenceParams

    return InferenceParams(**INFERENCE_DEFAULTS, edge_band=EDGE_BAND, **WINDOWS_M)


def _chips_dir(dataset_chips: dict[str, str]) -> Path:
    return resolve_directory(dataset_chips[IMAGE_INPUT], "*.tif*")


def _chip_names(chips_dir: Path) -> list[str]:
    """The chips the library reads: `.tif`/`.tiff` named `OAM-x-y-z`, so the split counts no other file."""
    from fair_trees.georef import parse_oam_name

    return sorted(
        p.name for p in chips_dir.iterdir() if p.suffix.lower() in (".tif", ".tiff") and parse_oam_name(p.name)
    )


def _labels_file(dataset_labels: str) -> Path:
    if dataset_labels.lower().endswith((".geojson", ".json")):
        return resolve_path(dataset_labels)
    found = sorted(resolve_directory(dataset_labels, "*.geojson").glob("*.geojson"))
    if len(found) != 1:
        raise ValueError(f"expected exactly one labels GeoJSON under {dataset_labels}, found {len(found)}")
    return found[0]


def _val_names(split_info: dict[str, Any], hp: dict[str, Any]) -> list[str]:
    """The first max(1, round(sample_fraction * n)) val chips in name order; the train chips are never thinned."""
    names = sorted(split_info["val_chip_names"])
    return names[: max(1, round(hp["sample_fraction"] * len(names)))]


def _start_weights(base_model_weights: str, directory: Path) -> Path:
    """The base checkpoint, alone in `directory`: fine_tune takes thresholds.json and windows.json beside the
    start weights as theirs, and fAIr's download cache could hold stray ones beside the cached file."""
    directory.mkdir()
    weights = directory / "base.pt"
    weights.symlink_to(resolve_path(base_model_weights).resolve())
    return weights


def _fine_tune_params(hp: dict[str, Any], num_classes: int) -> dict[str, Any]:
    params = {key: value for key, value in hp.items() if key not in PACK_KEYS and key != "scheduler"}
    if hp["scheduler"] not in SCHEDULES:
        raise ValueError(f"scheduler must be one of {', '.join(SCHEDULES)}, got {hp['scheduler']!r}")
    params["schedule"] = SCHEDULES[hp["scheduler"]]
    # The item's 0 stands for the library's default, None: one crop per train chip.
    params["samples_per_epoch"] = params.get("samples_per_epoch") or None
    return {**params, "num_classes": num_classes}


def _restore(checkpoint: bytes) -> Any:
    from fair_trees.train import build_model

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.pt"
        path.write_bytes(checkpoint)
        return build_model(path, trainable_layers=0).eval()


@step
def split_dataset(
    dataset_chips: dict[str, str],
    dataset_labels: str,
    hyperparameters: dict[str, Any],
) -> Annotated[dict[str, Any], "split_info"]:
    """Spatial block split of the chips on their OAM tile coordinates (`fair_trees.dataset.spatial_split`)."""
    from fair_trees.dataset import spatial_split
    from fair_trees.params import parse_split_params

    hp = _hyperparameters(hyperparameters)
    cfg = parse_split_params({key: hp[key] for key in SPLIT_KEYS})
    chip_names = _chip_names(_chips_dir(dataset_chips))
    if not chip_names:
        raise FileNotFoundError(f"no OAM-x-y-z.tif chips found at {dataset_chips[IMAGE_INPUT]}")
    train_names, val_names = spatial_split(chip_names, cfg.val_ratio, cfg.split_seed, block_size=cfg.block_size)

    info = {
        "strategy": "spatial",
        "val_ratio": len(val_names) / len(chip_names),  # the share the split gave: whole blocks move
        "seed": cfg.split_seed,
        "block_size": cfg.block_size,
        "train_count": len(train_names),
        "val_count": len(val_names),
        "description": _item()["properties"]["fair:split_spec"]["description"],
        "train_chip_names": train_names,
        "val_chip_names": val_names,
    }
    log_metadata(
        metadata={"fair/split": {k: v for k, v in info.items() if k not in ("train_chip_names", "val_chip_names")}}
    )
    return info


@step(output_materializers={"trained_model_artifact": CheckpointBytesMaterializer})
def train_model(
    dataset_chips: dict[str, str],
    dataset_labels: str,
    base_model_weights: str,
    hyperparameters: dict[str, Any],
    split_info: dict[str, Any],
    num_classes: int,
    model_name: str | None = None,
    base_model_id: str | None = None,
    dataset_id: str | None = None,
) -> tuple[Annotated[bytes, "trained_model_artifact"], Annotated[dict[str, Any], "serving_metadata"]]:
    """Fine-tune on the train chips with `fair_trees.train.fine_tune`, validating on the val chips.

    Returns the best epoch's state dict (`best.pt` bytes) and the thresholds and windows `export_onnx` writes
    into the model: steps run in separate pods, so they travel as an artifact, not as files.
    """
    import mlflow
    from fair_trees.export import checkpoint_thresholds
    from fair_trees.params import CLASSES
    from fair_trees.train import fine_tune

    hp = _hyperparameters(hyperparameters)
    chips_dir = _chips_dir(dataset_chips)

    def log_epoch(metrics: dict[str, float], epoch: int) -> None:
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=epoch)  # ty: ignore[possibly-missing-attribute]

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        weights = _start_weights(base_model_weights, work / "start")
        with mlflow_training_context(hp, model_name, base_model_id, dataset_id):
            result = fine_tune(
                [chips_dir / name for name in split_info["train_chip_names"]],
                [chips_dir / name for name in _val_names(split_info, hp)],
                _labels_file(dataset_labels),
                _fine_tune_params(hp, num_classes),
                weights=weights,
                out_dir=work / "run",
                on_epoch_end=log_epoch,
                val_params=_serving_params(),
                num_workers=0,  # the pods' /dev/shm is 64 MB, too small for DataLoader workers
                device=_device(),
                tmp_dir=work / "staging",
            )
        # train writes a threshold only for a class validation scored, and windows.json only beside one; the
        # model carries a threshold per class (the default where none was picked) and the windows it validated at.
        picked = checkpoint_thresholds(result.best_checkpoint) or {}
        serving_metadata = {
            "thresholds": {name: picked.get(name, CONFIDENCE_THRESHOLD) for name in CLASSES[:num_classes]},
            "windows": dict(WINDOWS_M),
        }
        return result.best_checkpoint.read_bytes(), serving_metadata


@step
def evaluate_model(
    trained_model: bytes,
    dataset_chips: dict[str, str],
    dataset_labels: str,
    hyperparameters: dict[str, Any],
    split_info: dict[str, Any],
    class_names: list[str] | None = None,
) -> Annotated[dict[str, Any], "metrics"]:
    """Per-class precision, recall and F1 on the val chips, at the settings fAIr serves the model with.

    The trained torch model runs here, not the exported ONNX: `fair_trees.evaluate` slides the serving windows
    over each val mosaic and decodes with the serving parameters (threshold, overlap, edge band, group merge)
    and decode path, then matches boxes one-to-one (Hungarian, IoU > 0.4) against the labels of that class;
    counts are pooled over the chips.
    """
    import torch
    from fair_trees.dataset import load_fair_dataset
    from fair_trees.evaluate import evaluate_units, region_units
    from fair_trees.params import CLASSES
    from fair_trees.train import torch_run_fn

    hp = _hyperparameters(hyperparameters)
    chips_dir = _chips_dir(dataset_chips)
    regions = load_fair_dataset(
        [chips_dir / name for name in _val_names(split_info, hp)], _labels_file(dataset_labels), split="val"
    )
    run_fn = torch_run_fn(_restore(trained_model), torch.device(_device()))
    params = _serving_params()
    metrics: dict[str, Any] = {}
    for index, name in enumerate(CLASSES):
        report = evaluate_units(run_fn, region_units(regions, label=index + 1), params, class_index=index)
        metrics |= {
            f"precision_{name}": round(float(report.precision), 4),
            f"recall_{name}": round(float(report.recall), 4),
            f"f1_{name}": round(float(report.f1), 4),
            f"val_{name}s": int(sum(unit["tp"] + unit["fn"] for unit in report.units)),
        }
    log_evaluation_results(metrics)
    return metrics


@step(output_materializers={"onnx_model": ONNXMaterializer})
def export_onnx(
    trained_model: bytes,
    hyperparameters: dict[str, Any],
    serving_metadata: dict[str, Any],
) -> Annotated[bytes, "onnx_model"]:
    """The raw network (`images` -> `cls_logits`, `bbox_regression`) via `fair_trees.export.export_onnx_bytes`,
    which checks it with `onnx.checker`; the thresholds and windows go into the model's metadata."""
    from fair_trees.export import export_onnx_bytes

    return export_onnx_bytes(
        _restore(trained_model), thresholds=serving_metadata["thresholds"], windows=serving_metadata["windows"]
    )


@step
def run_inference(
    model_uri: str,
    input_images: dict[str, str],
    inference_params: dict[str, Any],
) -> Annotated[dict[str, Any], "predictions"]:
    from fair.serve.base import load_session

    return predict(load_session(model_uri), input_images, inference_params)


@pipeline
def training_pipeline(
    base_model_weights: str,
    dataset_chips: dict[str, str],
    dataset_labels: str,
    num_classes: int,
    hyperparameters: dict[str, Any],
) -> None:
    split_info = split_dataset(
        dataset_chips=dataset_chips,
        dataset_labels=dataset_labels,
        hyperparameters=hyperparameters,
    )
    trained_model, serving_metadata = train_model(
        dataset_chips=dataset_chips,
        dataset_labels=dataset_labels,
        base_model_weights=base_model_weights,
        hyperparameters=hyperparameters,
        split_info=split_info,
        num_classes=num_classes,
    )
    evaluate_model(
        trained_model=trained_model,
        dataset_chips=dataset_chips,
        dataset_labels=dataset_labels,
        hyperparameters=hyperparameters,
        split_info=split_info,
    )
    export_onnx(trained_model=trained_model, hyperparameters=hyperparameters, serving_metadata=serving_metadata)


@pipeline
def inference_pipeline(
    model_uri: str,
    input_images: dict[str, str],
    inference_params: dict[str, Any] | None = None,
) -> None:
    run_inference(
        model_uri=model_uri,
        input_images=input_images,
        inference_params=inference_params or {},
    )
