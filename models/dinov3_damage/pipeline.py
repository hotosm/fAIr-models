"""dinov3-damage: frozen DINOv3 ViT-L/16 siamese model that grades building damage from pre- and post-event
imagery, delegating training and inference to the `dda` package."""

import json
import tempfile
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, Any

from zenml import log_metadata, pipeline, step

from fair.utils.data import resolve_directory, resolve_path
from fair.zenml.instrumentation import log_evaluation_results, mlflow_training_context
from fair.zenml.materializers import CheckpointBytesMaterializer, ONNXMaterializer

MODEL_NAME = "dinov3-damage"
POST_INPUT = "post"
PRE_INPUT = "pre"
DAMAGE_CLASS_COUNT = 4
OUTPUT_VARIABLES = ["damage_class", "damage", "damage_confidence", "osm_id"]
# Every footprint that stood before the event, including those mapped since as destroyed.
OSM_FOOTPRINT_FAMILIES = [
    {"key": "building", "status": "standing"},
    {"key": "destroyed:building", "status": "destroyed"},
]


def predict(session: Any, input_images: dict[str, str], params: dict[str, Any]) -> dict[str, Any]:
    """Damage per building footprint over the area the post-event chips cover; the footprints come from
    `buildings_geojson_uri`, or from OpenStreetMap when it is empty."""
    from dda.config import load_config
    from dda.onnx_infer import predict_damage_onnx
    from dda.pipeline.chip_windows import area_inputs
    from dda.pool import filter_by_confidence

    if missing := [name for name in (POST_INPUT, PRE_INPUT) if name not in input_images]:
        raise ValueError(f"imagery for input(s) {missing} is required")
    config = load_config(None)
    config.osm_tag_families = OSM_FOOTPRINT_FAMILIES
    with tempfile.TemporaryDirectory() as work_dir:
        post_raster, pre_raster, buildings = area_inputs(
            resolve_directory(input_images[POST_INPUT], "*.tif*"),
            resolve_directory(input_images[PRE_INPUT], "*.tif*"),
            params.get("buildings_geojson_uri") or None,
            Path(work_dir),
            config,
        )
        scored = predict_damage_onnx(session, post_raster, pre_raster, buildings, config)
    kept = filter_by_confidence(scored, float(params["confidence_threshold"]))
    if "osm_id" not in kept.columns:
        kept = kept.assign(osm_id=None)
    return json.loads(kept[[*OUTPUT_VARIABLES, "geometry"]].to_crs("EPSG:4326").to_json(na="null"))


def preprocess(image_path: Any) -> Any:
    """One RGB chip as a normalised (1, 3, H, W) float32 input; post and pre chips take the same transform."""
    import numpy as np
    import rasterio
    from dda.config import DINOV3_MEAN, DINOV3_STD, load_config
    from dda.tiling import radiometric_normalize

    with rasterio.open(image_path) as src:
        rgb = src.read([1, 2, 3]).transpose(1, 2, 0)
    if load_config(None).radiometric_normalize:
        rgb = radiometric_normalize(rgb)
    mean = np.asarray(DINOV3_MEAN, dtype=np.float32).reshape(3, 1, 1)
    std = np.asarray(DINOV3_STD, dtype=np.float32).reshape(3, 1, 1)
    return ((rgb.transpose(2, 0, 1).astype(np.float32) / 255.0 - mean) / std)[np.newaxis, ...]


def postprocess(damage_logits: Any) -> Any:
    """Per-pixel damage class (0 no damage to 3 destroyed) from the (N, 4, H, W) damage logits."""
    import numpy as np

    return np.asarray(damage_logits).argmax(axis=1).astype(np.uint8)


@contextmanager
def _checkpoint_file(checkpoint: bytes) -> Iterator[Path]:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "model.ckpt"
        path.write_bytes(checkpoint)
        yield path


def _read_labels(dataset_labels: str) -> Any:
    """The dataset's labels GeoDataFrame; label i (1..4) is xBD damage code i."""
    import geopandas as gpd

    labels_files = sorted(resolve_directory(dataset_labels, "*.geojson").glob("*.geojson"))
    if len(labels_files) != 1:
        raise ValueError(f"expected one labels GeoJSON in {dataset_labels}, found {len(labels_files)}")
    labels = gpd.read_file(labels_files[0])
    if labels.crs is None or not labels.crs.equals("EPSG:4326", ignore_axis_order=True):
        raise ValueError(f"labels must be EPSG:4326, got {labels.crs}")
    if not labels["label"].isin(range(1, DAMAGE_CLASS_COUNT + 1)).all():
        raise ValueError(
            f"labels must be 1..{DAMAGE_CLASS_COUNT} (no damage, minor, major, destroyed), "
            f"got {sorted(labels['label'].unique())}"
        )
    return labels


def _damage_windows(
    dataset_chips: dict[str, str],
    dataset_labels: str,
    chip_names: list[str],
) -> list[dict[str, Any]]:
    """dda training windows for `chip_names`."""
    from dda.config import load_config
    from dda.pipeline.chip_windows import windows_from_chips

    labels = _read_labels(dataset_labels)
    labels["damage"] = labels["label"]
    return windows_from_chips(
        resolve_directory(dataset_chips[POST_INPUT], "*.tif*"),
        resolve_directory(dataset_chips[PRE_INPUT], "*.tif*"),
        labels,
        chip_names,
        load_config(None),
    )


@step
def split_dataset(
    dataset_chips: dict[str, str],
    dataset_labels: str,
    hyperparameters: dict[str, Any],
) -> Annotated[dict[str, Any], "split_info"]:
    """Seeded split of the labelled 2x2 tile blocks, the model's 512 px windows, stratified by the set of
    damage classes each block holds."""
    import mercantile
    import numpy as np
    from dda.pipeline.chip_windows import parse_chip_name
    from shapely.geometry import box

    chip_names = sorted(path.name for path in resolve_directory(dataset_chips[POST_INPUT], "*.tif*").glob("OAM-*.tif"))
    blocks: dict[tuple[int, int, int], list[str]] = defaultdict(list)
    for name in chip_names:
        x, y, z = parse_chip_name(name)
        blocks[(x // 2, y // 2, z)].append(name)
    labels = _read_labels(dataset_labels)
    blocks_by_classes: dict[tuple[int, ...], list[tuple[int, int, int]]] = defaultdict(list)
    for key in sorted(blocks):
        block_x, block_y, zoom = key
        west, _, _, north = mercantile.bounds(2 * block_x, 2 * block_y, zoom)
        _, south, east, _ = mercantile.bounds(2 * block_x + 1, 2 * block_y + 1, zoom)
        hits = labels.iloc[labels.sindex.query(box(west, south, east, north), predicate="intersects")]
        if len(hits):
            blocks_by_classes[tuple(sorted(hits["label"].unique()))].append(key)
    if not blocks_by_classes:
        raise ValueError(f"no chip block in {dataset_chips[POST_INPUT]} intersects the labels")
    val_ratio = float(hyperparameters["val_ratio"])
    seed = int(hyperparameters["split_seed"])
    rng = np.random.default_rng(seed)
    val_blocks: set[tuple[int, int, int]] = set()
    for classes in sorted(blocks_by_classes):
        keys = blocks_by_classes[classes]
        val_blocks |= {keys[index] for index in rng.permutation(len(keys))[: round(len(keys) * val_ratio)]}
    labelled = sorted(key for keys in blocks_by_classes.values() for key in keys)
    if not val_blocks or val_blocks == set(labelled):
        raise ValueError(f"val_ratio={val_ratio} leaves no labelled block for one split of {len(labelled)}")
    info = {
        "strategy": "spatial",
        "val_ratio": val_ratio,
        "seed": seed,
        "train_chip_names": [name for key in labelled if key not in val_blocks for name in blocks[key]],
        "val_chip_names": [name for key in labelled if key in val_blocks for name in blocks[key]],
    }
    info["train_count"] = len(info["train_chip_names"])
    info["val_count"] = len(info["val_chip_names"])
    log_metadata(metadata={"fair/split": info})
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
) -> Annotated[bytes, "trained_model_artifact"]:
    """Few-shot fine-tune of the damage heads; ships the base checkpoint unless val macro F1 improves."""
    from dda.pipeline.fewshot_damage import fit_damage_windows

    if num_classes != DAMAGE_CLASS_COUNT:
        raise ValueError(f"{MODEL_NAME} grades {DAMAGE_CLASS_COUNT} damage classes, got num_classes={num_classes}")
    train_windows = _damage_windows(dataset_chips, dataset_labels, split_info["train_chip_names"])
    val_windows = _damage_windows(dataset_chips, dataset_labels, split_info["val_chip_names"])
    pretrained = resolve_path(base_model_weights)

    with (
        tempfile.TemporaryDirectory() as out_dir,
        mlflow_training_context(hyperparameters, model_name, base_model_id, dataset_id),
    ):
        result = fit_damage_windows(
            train_windows,
            val_windows,
            Path(out_dir),
            ckpt=pretrained,
            epochs=int(hyperparameters["epochs"]),
            lr=float(hyperparameters["learning_rate"]),
            patience=int(hyperparameters["early_stop_patience"]),
            batch_size=int(hyperparameters["batch_size"]),
            strict_head=bool(hyperparameters["strict_head"]),
            seed=int(split_info["seed"]),
        )
        log_metadata(metadata={"fair/finetune_summary": result.to_dict()})
        if result.delta > 0:
            return result.best_ckpt.read_bytes()
    log_metadata(metadata={"fair/finetune_reverted_to_baseline": True, "fair/finetune_delta": result.delta})
    return pretrained.read_bytes()


@step
def evaluate_model(
    trained_model: bytes,
    dataset_chips: dict[str, str],
    dataset_labels: str,
    hyperparameters: dict[str, Any],
    split_info: dict[str, Any],
    num_classes: int = DAMAGE_CLASS_COUNT,
    class_names: list[str] | None = None,
) -> Annotated[dict[str, Any], "metrics"]:
    """Damage macro and harmonic F1 over the labelled classes, destroyed F1 and localization IoU on the val windows."""
    from dda.pipeline.fewshot_damage import evaluate_damage_windows

    val_windows = _damage_windows(dataset_chips, dataset_labels, split_info["val_chip_names"])
    with _checkpoint_file(trained_model) as checkpoint:
        metrics = evaluate_damage_windows(val_windows, checkpoint, batch_size=int(hyperparameters["batch_size"]))
    log_evaluation_results(metrics)
    return metrics


@step
def run_inference(
    model_uri: str,
    input_images: dict[str, str],
    inference_params: dict[str, Any],
) -> Annotated[dict[str, Any], "predictions"]:
    from fair.serve.base import load_session

    return predict(load_session(model_uri), input_images, inference_params)


@step(output_materializers={"onnx_model": ONNXMaterializer})
def export_onnx(
    trained_model: bytes,
    hyperparameters: dict[str, Any],
    num_classes: int = DAMAGE_CLASS_COUNT,
) -> Annotated[bytes, "onnx_model"]:
    """Single-file ONNX of the damage logits for (post, pre), parity-checked against torch by dda."""
    import onnx
    from dda.config import load_config
    from dda.export import export_onnx as export_damage_onnx

    with tempfile.TemporaryDirectory() as export_directory, _checkpoint_file(trained_model) as checkpoint:
        path = Path(export_directory) / "model.onnx"
        export_damage_onnx(load_config(None), checkpoint, path)
        onnx.save(onnx.load(str(path)), str(path), save_as_external_data=False)
        onnx.checker.check_model(str(path))
        return path.read_bytes()


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
    trained = train_model(
        dataset_chips=dataset_chips,
        dataset_labels=dataset_labels,
        base_model_weights=base_model_weights,
        hyperparameters=hyperparameters,
        split_info=split_info,
        num_classes=num_classes,
    )
    evaluate_model(
        trained_model=trained,
        dataset_chips=dataset_chips,
        dataset_labels=dataset_labels,
        hyperparameters=hyperparameters,
        split_info=split_info,
        num_classes=num_classes,
    )
    export_onnx(trained_model=trained, hyperparameters=hyperparameters, num_classes=num_classes)


@pipeline
def inference_pipeline(
    model_uri: str,
    input_images: dict[str, str],
    inference_params: dict[str, Any],
) -> None:
    run_inference(
        model_uri=model_uri,
        input_images=input_images,
        inference_params=inference_params,
    )
