"""ZenML pipeline for DeepForest tree-crown detection.

Entrypoints referenced by models/deepforest_tree_crowns/stac-item.json.
Pretrained backbone: DeepForest RetinaNet-ResNet50 (weecology/deepforest-tree).
Torch/DeepForest/ONNX imports stay inside the training-only functions so the
distroless inference image (ONNX runtime, no torch) can import this module to
reach ``predict``.
"""

import hashlib
import random
import tempfile
from pathlib import Path
from typing import Annotated, Any

from zenml import log_metadata, pipeline, step

from fair.zenml.instrumentation import log_evaluation_results, mlflow_training_context
from fair.zenml.materializers import CheckpointBytesMaterializer, ONNXMaterializer

MODEL_INPUT_SIZE = 256
CHIP_SIZE = 256
# Class string written to the DeepForest annotation CSV; matches the pretrained
# label_dict so finetuning continues the single existing class.
_TRAIN_LABEL = "Tree"
# Class name emitted on output features.
_LABEL_NAME = "tree"


def _get_device() -> str:
    import os

    if os.environ.get("FAIR_FORCE_CPU", "").lower() in {"1", "true", "yes", "on"}:
        return "cpu"
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _apply_device(model: Any) -> None:
    device = _get_device()
    model.config.accelerator = {"cuda": "gpu", "mps": "mps"}.get(device, "cpu")
    model.config.devices = 1


def _parse_hf_ref(url: str) -> tuple[str, str]:
    """Extract (repo_id, revision) from a huggingface.co resolve URL."""
    from urllib.parse import urlparse

    parsed = urlparse(url)
    parts = parsed.path.strip("/").split("/")
    if parsed.netloc != "huggingface.co" or len(parts) < 5 or parts[2] != "resolve":
        msg = f"checkpoint href must be a huggingface.co '/resolve/<revision>/' URL, got: {url}"
        raise ValueError(msg)
    return f"{parts[0]}/{parts[1]}", parts[3]


def _load_base_model(base_model_weights: str) -> Any:
    from deepforest import main

    repo_id, revision = _parse_hf_ref(base_model_weights)
    model = main.deepforest()
    model.load_model(repo_id, revision=revision)
    return model


def _restore_model(trained_model: Any) -> Any:
    import io

    import torch
    from deepforest import main

    if isinstance(trained_model, bytes):
        model = main.deepforest()
        model.model.load_state_dict(torch.load(io.BytesIO(trained_model), map_location="cpu"))
        model.model.eval()
        return model
    return trained_model


def _resize_chw(arr: Any, size: int) -> Any:
    import numpy as np
    from PIL import Image

    if arr.shape[-2:] == (size, size):
        return arr
    channels = [
        np.asarray(Image.fromarray(arr[c]).resize((size, size), Image.Resampling.BILINEAR)) for c in range(arr.shape[0])
    ]
    return np.stack(channels, axis=0)


def preprocess(image_path: Any, chip_size: int = MODEL_INPUT_SIZE) -> Any:
    import numpy as np
    import rasterio

    with rasterio.open(image_path) as src:
        arr = src.read([1, 2, 3]).astype(np.float32) / 255.0
    return _resize_chw(arr, chip_size).astype(np.float32)


def _decode(boxes: Any, scores: Any, labels: Any, confidence_threshold: float) -> list[dict[str, Any]]:
    detections: list[dict[str, Any]] = []
    for i in range(len(scores)):
        score = float(scores[i])
        if score < confidence_threshold:
            continue
        x1, y1, x2, y2 = (float(v) for v in boxes[i])
        detections.append({"bbox": [x1, y1, x2, y2], "confidence": score, "class": int(labels[i])})
    return detections


def postprocess(raw_output: Any, confidence_threshold: float = 0.38) -> list[dict[str, Any]]:
    boxes, scores, labels = raw_output
    return _decode(boxes, scores, labels, confidence_threshold)


def _preprocess_onnx_image(img_path: Any) -> tuple[Any, Any, Any, float, float]:
    import numpy as np
    import rasterio

    with rasterio.open(img_path) as src:
        arr = src.read([1, 2, 3]).astype(np.float32) / 255.0
        transform = src.transform
        crs = src.crs
        width, height = src.width, src.height

    arr = _resize_chw(arr, MODEL_INPUT_SIZE)
    scale_x = width / MODEL_INPUT_SIZE
    scale_y = height / MODEL_INPUT_SIZE
    return arr.astype(np.float32), transform, crs, scale_x, scale_y


def _box_feature(
    det: dict[str, Any], transform: Any, crs: Any, scale_x: float, scale_y: float, source: str
) -> dict[str, Any]:
    from pyproj import Transformer

    x1, y1, x2, y2 = det["bbox"]
    corners_px = [(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)]
    reproject = None
    if crs is not None and str(crs) != "EPSG:4326":
        reproject = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform
    ring: list[list[float]] = []
    for col_px, row_px in corners_px:
        geo_x, geo_y = transform * (col_px * scale_x, row_px * scale_y)
        if reproject is not None:
            geo_x, geo_y = reproject(geo_x, geo_y)
        ring.append([geo_x, geo_y])
    return {
        "type": "Feature",
        "properties": {"confidence": round(det["confidence"], 4), "class": _LABEL_NAME, "source": source},
        "geometry": {"type": "Polygon", "coordinates": [ring]},
    }


def _build_feature_collection(features: list[dict[str, Any]]) -> dict[str, Any]:
    return {"type": "FeatureCollection", "features": features}


def predict(session: Any, input_images: str, params: dict[str, Any]) -> dict[str, Any]:
    from fair.utils.data import resolve_directory

    if "confidence_threshold" not in params:
        raise ValueError("params['confidence_threshold'] is required")
    confidence_threshold = float(params["confidence_threshold"])
    input_name = session.get_inputs()[0].name

    input_dir = resolve_directory(input_images)
    patterns = ("*.png", "*.tif", "*.tiff", "*.jpg")
    img_paths = sorted(p for pat in patterns for p in input_dir.glob(pat))
    if not img_paths:
        msg = f"No input images found in {input_dir}"
        raise FileNotFoundError(msg)

    features: list[dict[str, Any]] = []
    for img_path in img_paths:
        chw, transform, crs, scale_x, scale_y = _preprocess_onnx_image(img_path)
        boxes, scores, labels = session.run(None, {input_name: chw})
        for det in _decode(boxes, scores, labels, confidence_threshold):
            features.append(_box_feature(det, transform, crs, scale_x, scale_y, img_path.name))
    return _build_feature_collection(features)


def _resolve_geojson(labels_path: str) -> Path:
    from fair.utils.data import resolve_directory, resolve_path

    if labels_path.endswith((".geojson", ".json")):
        return resolve_path(labels_path)
    local = resolve_directory(labels_path, "*.geojson")
    files = sorted(local.rglob("*.geojson")) or sorted(local.rglob("*.json"))
    if not files:
        msg = f"No .geojson labels found under {labels_path}"
        raise FileNotFoundError(msg)
    return files[0]


def _boxes_by_chip(chips_dir: Path, geojson_path: Path) -> dict[str, list[tuple[str, int, int, int, int]]]:
    """Pixel bounding boxes per chip, clipping each label polygon to the chip."""
    import json

    import rasterio
    from rasterio.transform import rowcol
    from shapely.geometry import box, shape

    data = json.loads(geojson_path.read_text())
    polygons = [shape(feat["geometry"]) for feat in data.get("features", []) if feat.get("geometry")]

    chips = sorted(chips_dir.rglob("*.tif"))
    if not chips:
        msg = f"No .tif chips found under {chips_dir}"
        raise FileNotFoundError(msg)

    rows_by_chip: dict[str, list[tuple[str, int, int, int, int]]] = {}
    for chip in chips:
        with rasterio.open(chip) as src:
            chip_box = box(*src.bounds)
            transform = src.transform
            width, height = src.width, src.height
        rows: list[tuple[str, int, int, int, int]] = []
        for poly in polygons:
            if not poly.intersects(chip_box):
                continue
            clipped = poly.intersection(chip_box)
            if clipped.is_empty:
                continue
            minx, miny, maxx, maxy = clipped.bounds
            row_tl, col_tl = rowcol(transform, minx, maxy)
            row_br, col_br = rowcol(transform, maxx, miny)
            xmin = max(0, min(int(col_tl), int(col_br)))
            xmax = min(width, max(int(col_tl), int(col_br)))
            ymin = max(0, min(int(row_tl), int(row_br)))
            ymax = min(height, max(int(row_tl), int(row_br)))
            if xmax - xmin <= 0 or ymax - ymin <= 0:
                continue
            rows.append((chip.name, xmin, ymin, xmax, ymax))
        if rows:
            rows_by_chip[chip.name] = rows
    return rows_by_chip


def _write_annotation_csv(path: Path, rows: list[tuple[str, int, int, int, int]]) -> None:
    import csv

    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "xmin", "ymin", "xmax", "ymax", "label"])
        for name, xmin, ymin, xmax, ymax in rows:
            writer.writerow([name, xmin, ymin, xmax, ymax, _TRAIN_LABEL])


def _count_images(csv_path: Path) -> int:
    import csv

    with csv_path.open(newline="") as f:
        return len({row["image_path"] for row in csv.DictReader(f)})


def _dataset_cache_dir(chips_path: str, labels_path: str, val_ratio: float, seed: int, sample_fraction: float) -> Path:
    key = hashlib.sha256(f"{chips_path}|{labels_path}|{val_ratio}|{seed}|{sample_fraction}".encode()).hexdigest()[:16]
    return Path(tempfile.gettempdir()) / f"deepforest_dataset_{key}"


def _prepare_dataset(
    chips_path: str,
    labels_path: str,
    val_ratio: float,
    seed: int,
    sample_fraction: float = 1.0,
) -> tuple[Path, Path, Path, int, int]:
    from fair.utils.data import resolve_directory

    cache = _dataset_cache_dir(chips_path, labels_path, val_ratio, seed, sample_fraction)
    train_csv = cache / "train.csv"
    val_csv = cache / "val.csv"
    chips_dir = resolve_directory(chips_path)
    if train_csv.exists() and val_csv.exists():
        return chips_dir, train_csv, val_csv, _count_images(train_csv), _count_images(val_csv)

    rows_by_chip = _boxes_by_chip(chips_dir, _resolve_geojson(labels_path))
    chip_names = sorted(rows_by_chip)
    if not chip_names:
        msg = f"No labeled chips found for {chips_path}"
        raise FileNotFoundError(msg)
    if sample_fraction < 1.0:
        chip_names = chip_names[:: max(1, round(1 / sample_fraction))]

    order = chip_names[:]
    random.Random(seed).shuffle(order)
    val_count = max(1, int(len(order) * val_ratio))
    if val_count >= len(order):
        val_count = max(1, len(order) - 1)
    val_set = set(order[-val_count:])

    train_rows = [r for name in chip_names if name not in val_set for r in rows_by_chip[name]]
    val_rows = [r for name in chip_names if name in val_set for r in rows_by_chip[name]]

    cache.mkdir(parents=True, exist_ok=True)
    _write_annotation_csv(train_csv, train_rows)
    _write_annotation_csv(val_csv, val_rows)
    return chips_dir, train_csv, val_csv, len(chip_names) - val_count, val_count


def _ensure_split(
    dataset_chips: str,
    dataset_labels: str,
    hyperparameters: dict[str, Any],
    split_info: dict[str, Any],
) -> tuple[Path, Path, Path]:
    chips_dir = Path(split_info.get("_chips_dir", ""))
    train_csv = Path(split_info.get("_train_csv", ""))
    val_csv = Path(split_info.get("_val_csv", ""))
    if chips_dir.exists() and train_csv.exists() and val_csv.exists():
        return chips_dir, train_csv, val_csv

    chips_dir, train_csv, val_csv, _, _ = _prepare_dataset(
        dataset_chips,
        dataset_labels,
        split_info.get("val_ratio", hyperparameters.get("val_ratio", 0.2)),
        split_info.get("seed", hyperparameters.get("split_seed", 42)),
        hyperparameters.get("sample_fraction", 1.0),
    )
    return chips_dir, train_csv, val_csv


@step
def split_dataset(
    dataset_chips: str,
    dataset_labels: str,
    hyperparameters: dict[str, Any],
) -> Annotated[dict[str, Any], "split_info_artifact"]:
    val_ratio = hyperparameters.get("val_ratio", 0.2)
    seed = hyperparameters.get("split_seed", 42)
    sample_fraction = hyperparameters.get("sample_fraction", 1.0)

    chips_dir, train_csv, val_csv, train_count, val_count = _prepare_dataset(
        dataset_chips, dataset_labels, val_ratio, seed, sample_fraction
    )

    split_info = {
        "strategy": "random",
        "val_ratio": val_ratio,
        "seed": seed,
        "train_count": train_count,
        "val_count": val_count,
        "description": f"Seeded random shuffle of labeled chips, last {val_ratio:.0%} held out for validation",
        "_chips_dir": str(chips_dir),
        "_train_csv": str(train_csv),
        "_val_csv": str(val_csv),
    }
    log_metadata(metadata={"fair/split": {k: v for k, v in split_info.items() if not k.startswith("_")}})
    return split_info


@step(output_materializers={"trained_model_artifact": CheckpointBytesMaterializer})
def train_model(
    dataset_chips: str,
    dataset_labels: str,
    base_model_weights: str,
    hyperparameters: dict[str, Any],
    split_info: dict[str, Any],
    num_classes: int = 1,
    model_name: str | None = None,
    base_model_id: str | None = None,
    dataset_id: str | None = None,
) -> Annotated[Any, "trained_model_artifact"]:
    import io

    import torch

    epochs = int(hyperparameters["epochs"])
    batch_size = int(hyperparameters.get("batch_size", 4))
    learning_rate = float(hyperparameters.get("learning_rate", 1e-4))

    chips_dir, train_csv, val_csv = _ensure_split(dataset_chips, dataset_labels, hyperparameters, split_info)

    model = _load_base_model(base_model_weights)
    model.config.train.csv_file = str(train_csv)
    model.config.train.root_dir = str(chips_dir)
    model.config.train.epochs = epochs
    model.config.train.lr = learning_rate
    model.config.validation.csv_file = str(val_csv)
    model.config.validation.root_dir = str(chips_dir)
    model.config.batch_size = batch_size
    model.config.workers = 0
    _apply_device(model)

    with mlflow_training_context(hyperparameters, model_name, base_model_id, dataset_id):
        model.create_trainer(enable_checkpointing=False, enable_progress_bar=False, default_root_dir=tempfile.mkdtemp())
        model.trainer.fit(model)
    log_metadata(metadata={"epoch": epochs})

    buffer = io.BytesIO()
    torch.save(model.model.state_dict(), buffer)
    return buffer.getvalue()


@step
def evaluate_model(
    trained_model: Any,
    dataset_chips: str,
    dataset_labels: str,
    hyperparameters: dict[str, Any],
    split_info: dict[str, Any],
    class_names: list[str] | None = None,
) -> Annotated[dict[str, Any], "metrics"]:
    import math

    chips_dir, _, val_csv = _ensure_split(dataset_chips, dataset_labels, hyperparameters, split_info)

    model = _restore_model(trained_model)
    _apply_device(model)
    model.create_trainer(enable_progress_bar=False, default_root_dir=tempfile.mkdtemp())
    result = model.evaluate(csv_file=str(val_csv), root_dir=str(chips_dir))

    def _finite(value: Any) -> float:
        return float(value) if value is not None and math.isfinite(value) else 0.0

    metrics = {"precision": _finite(result.get("box_precision")), "recall": _finite(result.get("box_recall"))}
    log_evaluation_results(metrics)
    return metrics


@step(output_materializers={"onnx_model": ONNXMaterializer})
def export_onnx(trained_model: Any) -> Annotated[bytes, "onnx_model"]:
    import onnx
    import torch

    model = _restore_model(trained_model)
    net = model.model.eval()

    class _RetinaNetOnnx(torch.nn.Module):
        def __init__(self, detector: torch.nn.Module) -> None:
            super().__init__()
            self.detector = detector

        def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            out = self.detector([image])
            det = out[1][0] if isinstance(out, tuple) else out[0]
            return det["boxes"], det["scores"], det["labels"]

    wrapper = _RetinaNetOnnx(net).eval()
    dummy = torch.rand(3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, dtype=torch.float32)
    onnx_path = Path(tempfile.mkdtemp()) / "deepforest_tree_crowns.onnx"
    torch.onnx.export(
        wrapper,
        (dummy,),
        str(onnx_path),
        input_names=["image"],
        output_names=["boxes", "scores", "labels"],
        dynamic_axes={"boxes": {0: "n"}, "scores": {0: "n"}, "labels": {0: "n"}},
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    onnx.checker.check_model(str(onnx_path))
    try:
        return onnx_path.read_bytes()
    finally:
        onnx_path.unlink(missing_ok=True)


@step
def run_inference(
    model_uri: str,
    input_images: str,
    inference_params: dict[str, Any],
) -> Annotated[dict[str, Any], "predictions"]:
    from fair.serve.base import load_session

    session = load_session(model_uri)
    return predict(session, input_images, inference_params)


@pipeline
def training_pipeline(
    base_model_weights: str,
    dataset_chips: str,
    dataset_labels: str,
    num_classes: int,
    hyperparameters: dict[str, Any],
) -> None:
    split_info = split_dataset(
        dataset_chips=dataset_chips,
        dataset_labels=dataset_labels,
        hyperparameters=hyperparameters,
    )
    trained_model = train_model(
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
    export_onnx(trained_model=trained_model)


@pipeline
def inference_pipeline(
    model_uri: str,
    input_images: str,
    inference_params: dict[str, Any],
) -> None:
    run_inference(
        model_uri=model_uri,
        input_images=input_images,
        inference_params=inference_params,
    )
