"""ZenML pipeline for YOLOv11n building detection.

Entrypoints referenced by models/yolo11n_detection/stac-item.json.
Pretrained backbone: Ultralytics YOLOv11n COCO.
"""

import json
import shutil
import tempfile
import time
from pathlib import Path
from typing import Annotated, Any, Literal

from annotated_types import Ge, Le
from zenml import log_metadata, pipeline, step

from fair.zenml.metrics import log_fair_metrics, log_training_wall_time
from fair.zenml.steps import load_model


def _get_device() -> Literal["mps", "cuda", "cpu"]:
    import torch

    return "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


def preprocess(image_path: str) -> Any:
    from PIL import Image

    return Image.open(image_path).convert("RGB")


def postprocess(results: Any) -> list[dict[str, Any]]:
    detections = []
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


def _prepare_yolo_dataset(
    chips_path: str,
    coco_json_path: str,
    chip_size: int,
) -> Path:
    """Convert COCO JSON to YOLO txt format in a temp directory."""
    from fair.utils.data import resolve_directory, resolve_path

    local_chips = resolve_directory(chips_path)
    local_json = resolve_path(coco_json_path)

    with open(local_json, encoding="utf-8") as f:
        coco = json.load(f)

    img_id_to_name = {img["id"]: img["file_name"] for img in coco["images"]}
    img_id_to_size = {img["id"]: (img["width"], img["height"]) for img in coco["images"]}

    annotations_by_image: dict[int, list[dict[str, Any]]] = {}
    for ann in coco["annotations"]:
        annotations_by_image.setdefault(ann["image_id"], []).append(ann)

    yolo_dir = Path(tempfile.mkdtemp(prefix="yolo_dataset_"))
    images_dir = yolo_dir / "images" / "train"
    labels_dir = yolo_dir / "labels" / "train"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    for img_id, filename in img_id_to_name.items():
        src = local_chips / filename
        if not src.exists():
            continue

        shutil.copy2(src, images_dir / filename)
        w, h = img_id_to_size[img_id]
        label_file = labels_dir / Path(filename).with_suffix(".txt").name

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
    data_yaml.write_text(f"path: {yolo_dir}\ntrain: images/train\nval: images/train\nnc: 1\nnames: ['building']\n")

    return yolo_dir


@step
def train_model(
    dataset_chips: str,
    dataset_labels: str,
    base_model_weights: str,
    epochs: int,
    batch_size: int,
    chip_size: int,
    learning_rate: float = 0.01,
) -> Any:
    import mlflow
    from ultralytics import YOLO  # ty: ignore[unresolved-import]

    mlflow.autolog()  # ty: ignore[possibly-missing-attribute]
    mlflow.log_params(  # ty: ignore[possibly-missing-attribute]
        {
            "epochs": epochs,
            "batch_size": batch_size,
            "chip_size": chip_size,
            "learning_rate": learning_rate,
        }
    )

    yolo_dir = _prepare_yolo_dataset(dataset_chips, dataset_labels, chip_size)
    device = _get_device()

    model = YOLO(base_model_weights)
    wall_start = time.perf_counter()
    results = model.train(
        data=str(yolo_dir / "data.yaml"),
        epochs=epochs,
        batch=batch_size,
        imgsz=chip_size,
        device=device,
        lr0=learning_rate,
        verbose=False,
    )
    wall_seconds = time.perf_counter() - wall_start
    log_training_wall_time(wall_seconds)

    if results and hasattr(results, "results_dict"):
        metrics = results.results_dict
        log_metadata(
            metadata={
                "loss": metrics.get("train/box_loss", 0.0),
                "epoch": epochs,
            }
        )

    shutil.rmtree(yolo_dir, ignore_errors=True)
    return model


@step
def evaluate_model(
    trained_model: Any,
    dataset_chips: str,
    dataset_labels: str,
    chip_size: int = 640,
) -> dict[str, Any]:
    import mlflow
    from ultralytics import YOLO  # ty: ignore[unresolved-import]

    yolo_dir = _prepare_yolo_dataset(dataset_chips, dataset_labels, chip_size)

    model = trained_model if isinstance(trained_model, YOLO) else YOLO(trained_model)

    results = model.val(
        data=str(yolo_dir / "data.yaml"),
        imgsz=chip_size,
        verbose=False,
    )

    metrics_dict = {
        "accuracy": results.results_dict.get("metrics/mAP50(B)", 0.0),
        "mean_iou": results.results_dict.get("metrics/mAP50-95(B)", 0.0),
        "precision": results.results_dict.get("metrics/precision(B)", 0.0),
        "recall": results.results_dict.get("metrics/recall(B)", 0.0),
    }

    mlflow.log_metrics(metrics_dict)  # ty: ignore[possibly-missing-attribute]
    log_fair_metrics(metrics_dict)

    shutil.rmtree(yolo_dir, ignore_errors=True)
    return metrics_dict


@step
def export_onnx(trained_model: Any) -> str:
    from ultralytics import YOLO  # ty: ignore[unresolved-import]

    model = trained_model if isinstance(trained_model, YOLO) else YOLO(trained_model)
    return model.export(format="onnx")


@pipeline
def training_pipeline(
    base_model_weights: str,
    dataset_chips: str,
    dataset_labels: str,
    epochs: Annotated[int, Ge(1), Le(1000)],
    batch_size: Annotated[int, Ge(1), Le(64)],
    chip_size: Annotated[int, Ge(64), Le(2048)],
    learning_rate: Annotated[float, Ge(1e-6), Le(1.0)] = 0.01,
    class_names: list[str] | None = None,
) -> None:
    trained_model = train_model(
        dataset_chips=dataset_chips,
        dataset_labels=dataset_labels,
        base_model_weights=base_model_weights,
        epochs=epochs,
        batch_size=batch_size,
        chip_size=chip_size,
        learning_rate=learning_rate,
    )
    evaluate_model(
        trained_model=trained_model,
        dataset_chips=dataset_chips,
        dataset_labels=dataset_labels,
        chip_size=chip_size,
    )
    export_onnx(trained_model=trained_model)


@pipeline
def inference_pipeline(
    model_uri: str,
    input_images: str,
    chip_size: Annotated[int, Ge(64), Le(1024)] = 640,
    num_classes: Annotated[int, Ge(1), Le(256)] = 1,
    zenml_artifact_version_id: str = "",
) -> None:
    model = load_model(
        model_uri=model_uri,
        zenml_artifact_version_id=zenml_artifact_version_id,
    )
    detect(model=model, input_images=input_images, chip_size=chip_size)


@step
def detect(
    model: Any,
    input_images: str,
    chip_size: int = 640,
) -> dict[str, Any]:
    from ultralytics import YOLO  # ty: ignore[unresolved-import]

    from fair.utils.data import resolve_directory

    yolo_model = model if isinstance(model, YOLO) else YOLO(model)

    input_dir = resolve_directory(input_images)
    patterns = ("*.png", "*.tif", "*.tiff", "*.jpg")
    img_paths = sorted(p for pat in patterns for p in input_dir.glob(pat))
    if not img_paths:
        msg = f"No input images found in {input_dir}"
        raise FileNotFoundError(msg)

    all_detections: dict[str, list[dict[str, Any]]] = {}
    for img_path in img_paths:
        results = yolo_model(str(img_path), imgsz=chip_size, verbose=False)
        all_detections[img_path.name] = postprocess(results)

    return {"detections": all_detections}
