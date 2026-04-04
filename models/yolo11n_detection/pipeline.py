"""ZenML pipeline for YOLOv11n building detection.

Entrypoints referenced by models/yolo11n_detection/stac-item.json.
Pretrained backbone: Ultralytics YOLOv11n COCO.
"""

import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Annotated, Any

from zenml import log_metadata, pipeline, step

from fair.zenml.instrumentation import log_evaluation_results, mlflow_training_context
from fair.zenml.steps import load_model


def _get_device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _pixel_bbox_to_geo_feature(bbox_xyxy, transform, crs, properties):
    from pyproj import Transformer

    x1, y1, x2, y2 = bbox_xyxy
    corners_pixel = [(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)]
    corners_crs = [transform * (col, row) for col, row in corners_pixel]

    if crs is not None and str(crs) != "EPSG:4326":
        t = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        corners_geo = [list(t.transform(cx, cy)) for cx, cy in corners_crs]
    else:
        corners_geo = [list(c) for c in corners_crs]

    return {
        "type": "Feature",
        "properties": properties,
        "geometry": {"type": "Polygon", "coordinates": [corners_geo]},
    }


def _build_feature_collection(features):
    return {"type": "FeatureCollection", "features": features}


def postprocess(results: Any) -> list[dict[str, Any]]:
    detections: list[dict[str, Any]] = []
    for result in results:
        for box in result.boxes:
            detections.append(
                {
                    "bbox": box.xyxy[0].tolist(),
                    "confidence": box.conf.item(),
                    "class": int(box.cls.item()),
                }
            )
    return detections


def _dataset_cache_dir(chips_path: str, coco_json_path: str) -> Path:
    key = hashlib.sha256(f"{chips_path}|{coco_json_path}".encode()).hexdigest()[:16]
    return Path(tempfile.gettempdir()) / f"yolo_dataset_{key}"


def _prepare_yolo_dataset(
    chips_path: str,
    coco_json_path: str,
    chip_size: int,
    val_ratio: float = 0.2,
) -> tuple[Path, int, int]:
    from fair.utils.data import resolve_directory, resolve_path

    yolo_dir = _dataset_cache_dir(chips_path, coco_json_path)
    if (yolo_dir / "data.yaml").exists():
        train_count = len(list((yolo_dir / "images" / "train").iterdir()))
        val_count = len(list((yolo_dir / "images" / "val").iterdir()))
        return yolo_dir, train_count, val_count

    local_chips = resolve_directory(chips_path)
    local_json = resolve_path(coco_json_path)

    with open(local_json, encoding="utf-8") as f:
        coco = json.load(f)

    img_id_to_name = {img["id"]: img["file_name"] for img in coco["images"]}
    img_id_to_size = {img["id"]: (img["width"], img["height"]) for img in coco["images"]}
    annotations_by_image: dict[int, list[dict[str, Any]]] = {}
    for ann in coco["annotations"]:
        annotations_by_image.setdefault(ann["image_id"], []).append(ann)

    if yolo_dir.exists():
        shutil.rmtree(yolo_dir)
    for split in ("train", "val"):
        (yolo_dir / "images" / split).mkdir(parents=True)
        (yolo_dir / "labels" / split).mkdir(parents=True)

    available_ids = [img_id for img_id, fn in img_id_to_name.items() if (local_chips / fn).exists()]
    val_count = max(1, int(len(available_ids) * val_ratio))
    val_ids = set(available_ids[-val_count:])

    for img_id in available_ids:
        filename = img_id_to_name[img_id]
        split = "val" if img_id in val_ids else "train"
        shutil.copy2(local_chips / filename, yolo_dir / "images" / split / filename)

        w, h = img_id_to_size[img_id]
        label_file = yolo_dir / "labels" / split / Path(filename).with_suffix(".txt").name
        lines = []
        for ann in annotations_by_image.get(img_id, []):
            bx, by, bw, bh = ann["bbox"]
            cx = (bx + bw / 2) / w
            cy = (by + bh / 2) / h
            nw = bw / w
            nh = bh / h
            lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
        label_file.write_text("\n".join(lines))

    data_yaml = yolo_dir / "data.yaml"
    data_yaml.write_text(f"path: {yolo_dir}\ntrain: images/train\nval: images/val\nnc: 1\nnames: ['building']\n")

    train_count = len(available_ids) - val_count
    return yolo_dir, train_count, val_count


@step
def split_dataset(
    dataset_chips: str,
    dataset_labels: str,
    hyperparameters: dict[str, Any],
) -> Annotated[dict[str, Any], "split_info"]:
    val_ratio = hyperparameters.get("val_ratio", 0.2)
    chip_size = hyperparameters.get("chip_size", 640)

    yolo_dir, train_count, val_count = _prepare_yolo_dataset(dataset_chips, dataset_labels, chip_size, val_ratio)

    split_info = {
        "strategy": "random",
        "val_ratio": val_ratio,
        "seed": 0,
        "train_count": train_count,
        "val_count": val_count,
        "description": f"Last {val_ratio:.0%} of sorted image IDs held out for validation",
        "_yolo_dir": str(yolo_dir),
    }
    log_metadata(metadata={"fair/split": {k: v for k, v in split_info.items() if not k.startswith("_")}})
    return split_info


@step
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
) -> Annotated[Any, "trained_model"]:
    from ultralytics import YOLO

    epochs = hyperparameters["epochs"]
    batch_size = hyperparameters.get("batch_size", 8)
    chip_size = hyperparameters.get("chip_size", 640)
    learning_rate = hyperparameters.get("learning_rate", 0.01)

    yolo_dir = Path(split_info["_yolo_dir"])
    if not (yolo_dir / "data.yaml").exists():
        val_ratio = split_info["val_ratio"]
        yolo_dir, _, _ = _prepare_yolo_dataset(dataset_chips, dataset_labels, chip_size, val_ratio)

    device = _get_device()

    from ultralytics import settings as yolo_settings

    yolo_settings.update({"mlflow": False})

    with mlflow_training_context(
        hyperparameters,
        model_name,
        base_model_id,
        dataset_id,
    ):
        model = YOLO(base_model_weights)
        results = model.train(
            data=str(yolo_dir / "data.yaml"),
            epochs=epochs,
            batch=batch_size,
            imgsz=chip_size,
            device=device,
            lr0=learning_rate,
            cos_lr=True,
            verbose=False,
        )
        if results and hasattr(results, "results_dict"):
            log_metadata(metadata={"loss": results.results_dict.get("train/box_loss", 0.0), "epoch": epochs})

    saved_path = Path(tempfile.mkdtemp()) / "best.pt"
    model.save(str(saved_path))
    return str(saved_path)


@step
def evaluate_model(
    trained_model: Any,
    dataset_chips: str,
    dataset_labels: str,
    hyperparameters: dict[str, Any],
    split_info: dict[str, Any],
    class_names: list[str] | None = None,
) -> Annotated[dict[str, Any], "metrics"]:
    from ultralytics import YOLO

    chip_size = hyperparameters.get("chip_size", 640)

    yolo_dir = Path(split_info["_yolo_dir"])
    if not (yolo_dir / "data.yaml").exists():
        val_ratio = split_info["val_ratio"]
        yolo_dir, _, _ = _prepare_yolo_dataset(dataset_chips, dataset_labels, chip_size, val_ratio)

    model = trained_model if isinstance(trained_model, YOLO) else YOLO(trained_model)
    results = model.val(data=str(yolo_dir / "data.yaml"), imgsz=chip_size, verbose=False)

    if not hasattr(results, "results_dict") or not results.results_dict:
        msg = "YOLO validation produced no results"
        raise RuntimeError(msg)

    metrics_dict: dict[str, Any] = {
        "accuracy": results.results_dict.get("metrics/mAP50(B)", 0.0),
        "mean_iou": results.results_dict.get("metrics/mAP50-95(B)", 0.0),
        "precision": results.results_dict.get("metrics/precision(B)", 0.0),
        "recall": results.results_dict.get("metrics/recall(B)", 0.0),
    }
    log_evaluation_results(metrics_dict)
    return metrics_dict


@step
def export_onnx(trained_model: Any) -> Annotated[str, "onnx_model"]:
    import onnx
    from ultralytics import YOLO

    model = trained_model if isinstance(trained_model, YOLO) else YOLO(trained_model)
    onnx_path = model.export(format="onnx")
    onnx.checker.check_model(onnx_path)
    return onnx_path


@step
def detect(
    model: Any,
    input_images: str,
    chip_size: int = 640,
) -> Annotated[dict[str, Any], "predictions"]:
    import rasterio
    from ultralytics import YOLO

    from fair.utils.data import resolve_directory

    yolo_model = model if isinstance(model, YOLO) else YOLO(model)

    input_dir = resolve_directory(input_images)
    patterns = ("*.png", "*.tif", "*.tiff", "*.jpg")
    img_paths = sorted(p for pat in patterns for p in input_dir.glob(pat))
    if not img_paths:
        msg = f"No input images found in {input_dir}"
        raise FileNotFoundError(msg)

    all_features: list[dict[str, Any]] = []
    for img_path in img_paths:
        results = yolo_model(str(img_path), imgsz=chip_size, verbose=False)
        with rasterio.open(img_path) as src:
            img_transform = src.transform
            crs = src.crs

        for det in postprocess(results):
            feature = _pixel_bbox_to_geo_feature(
                det["bbox"],
                img_transform,
                crs,
                {
                    "confidence": round(det["confidence"], 4),
                    "class": det["class"],
                    "source": img_path.name,
                },
            )
            all_features.append(feature)

    return _build_feature_collection(all_features)


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
    chip_size: int = 640,
    num_classes: int = 1,
    zenml_artifact_version_id: str = "",
) -> None:
    model = load_model(model_uri=model_uri, zenml_artifact_version_id=zenml_artifact_version_id)
    detect(model=model, input_images=input_images, chip_size=chip_size)
