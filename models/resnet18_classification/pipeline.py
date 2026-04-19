"""ZenML pipeline for ResNet18 binary building classification.

Entrypoints referenced by models/resnet18_classification/stac-item.json.
Pretrained backbone: torchvision ResNet18 ImageNet.
"""

import csv
import random
import tempfile
from pathlib import Path
from typing import Annotated, Any

from zenml import log_metadata, pipeline, step

from fair.zenml.instrumentation import log_evaluation_results, mlflow_training_context

MODEL_INPUT_SIZE = 256


def _get_device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _download_checkpoint(url: str) -> Path:
    from upath import UPath

    local_path = Path(tempfile.mkdtemp()) / UPath(url).name
    local_path.write_bytes(UPath(url).read_bytes())
    return local_path


def _bounds_to_geo_feature(left, bottom, right, top, crs, properties):
    from pyproj import Transformer

    if crs is not None and str(crs) != "EPSG:4326":
        t = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        left, bottom = t.transform(left, bottom)
        right, top = t.transform(right, top)

    coords = [[left, bottom], [right, bottom], [right, top], [left, top], [left, bottom]]
    return {
        "type": "Feature",
        "properties": properties,
        "geometry": {"type": "Polygon", "coordinates": [coords]},
    }


def _build_feature_collection(features):
    return {"type": "FeatureCollection", "features": features}


def preprocess(batch: dict[str, Any]) -> tuple[Any, Any]:
    images = batch["image"].float()
    labels = batch["label"].float()
    return images, labels


def postprocess(logits: Any) -> Any:
    import torch

    return (torch.sigmoid(logits) > 0.5).int().cpu().numpy()


def _sigmoid(x: float) -> float:
    import math

    return 1.0 / (1.0 + math.exp(-x))


def _preprocess_onnx_image(img_path: Any) -> tuple[Any, Any, Any, Any]:
    import numpy as np
    import rasterio
    from PIL import Image

    with rasterio.open(img_path) as src:
        bounds = src.bounds
        crs = src.crs

    img = Image.open(img_path).convert("RGB").resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), Image.Resampling.BILINEAR)
    arr = np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
    return arr[np.newaxis, ...], bounds, crs, img_path.name


def predict(session: Any, input_images: str, params: dict[str, Any]) -> dict[str, Any]:
    from fair.utils.data import resolve_directory

    threshold = float(params.get("threshold", 0.5))
    input_name = session.get_inputs()[0].name

    input_dir = resolve_directory(input_images)
    patterns = ("*.png", "*.tif", "*.tiff", "*.jpg")
    img_paths = sorted(p for pat in patterns for p in input_dir.glob(pat))
    if not img_paths:
        msg = f"No input images found in {input_dir}"
        raise FileNotFoundError(msg)

    features: list[dict[str, Any]] = []
    for img_path in img_paths:
        batch, bounds, crs, source = _preprocess_onnx_image(img_path)
        logits = session.run(None, {input_name: batch})[0]
        logit_value = float(logits.reshape(-1)[0])
        confidence = _sigmoid(logit_value)
        label = "building" if confidence > threshold else "no_building"

        feature = _bounds_to_geo_feature(
            bounds.left,
            bounds.bottom,
            bounds.right,
            bounds.top,
            crs,
            {
                "label": label,
                "confidence": round(confidence, 4),
                "source": source,
            },
        )
        features.append(feature)
    return _build_feature_collection(features)


def _load_samples(chips_path: str, labels_csv_path: str) -> list[tuple[Path, int]]:
    from fair.utils.data import resolve_directory, resolve_path

    local_chips = resolve_directory(chips_path)
    local_csv = resolve_path(labels_csv_path)

    samples: list[tuple[Path, int]] = []
    with open(local_csv, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_path = local_chips / row["filename"]
            label = 1 if row["class_name"] == "building" else 0
            if img_path.exists():
                samples.append((img_path, label))
    return samples


def _split_samples(
    samples: list[tuple[Path, int]],
    val_ratio: float,
    seed: int,
) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]]]:
    ordered = sorted(samples, key=lambda s: s[0].name)
    rng = random.Random(seed)
    rng.shuffle(ordered)
    split_idx = max(1, int(len(ordered) * (1 - val_ratio)))
    return ordered[:split_idx], ordered[split_idx:]


def _build_classification_dataset(
    samples: list[tuple[Path, int]],
    chip_size: int,
    batch_size: int = 4,
    *,
    shuffle: bool = True,
) -> Any:
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms

    transform = transforms.Compose(
        [
            transforms.Resize((chip_size, chip_size)),
            transforms.ToTensor(),
        ]
    )

    class ChipDataset(Dataset):
        def __init__(self, items: list[tuple[Path, int]]) -> None:
            self.samples = items

        def __len__(self) -> int:
            return len(self.samples)

        def __getitem__(self, index: int) -> dict[str, Any]:
            path, label = self.samples[index]
            img = Image.open(path).convert("RGB")
            tensor = transform(img)
            return {"image": tensor, "label": torch.tensor(label, dtype=torch.float32)}

    dataset = ChipDataset(samples)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


@step
def split_dataset(
    dataset_chips: str,
    dataset_labels: str,
    hyperparameters: dict[str, Any],
) -> Annotated[dict[str, Any], "split_info"]:
    val_ratio = hyperparameters.get("val_ratio", 0.2)
    seed = hyperparameters.get("split_seed", 42)

    all_samples = _load_samples(dataset_chips, dataset_labels)
    train_samples, val_samples = _split_samples(all_samples, val_ratio, seed)

    split_info = {
        "strategy": "random",
        "val_ratio": val_ratio,
        "seed": seed,
        "train_count": len(train_samples),
        "val_count": len(val_samples),
        "description": "Random split by seeded shuffle of sorted filenames",
    }
    log_metadata(metadata={"fair/split": split_info})
    return split_info


@step
def train_model(
    dataset_chips: str,
    dataset_labels: str,
    base_model_weights: str,
    hyperparameters: dict[str, Any],
    split_info: dict[str, Any],
    num_classes: int = 2,
    model_name: str | None = None,
    base_model_id: str | None = None,
    dataset_id: str | None = None,
) -> Annotated[Any, "trained_model"]:
    import torch
    import torch.nn as nn
    from torchvision.models import resnet18

    epochs = hyperparameters["epochs"]
    batch_size = hyperparameters.get("batch_size", 8)
    learning_rate = hyperparameters.get("learning_rate", 0.001)
    weight_decay = hyperparameters.get("weight_decay", 0.0001)
    chip_size = hyperparameters.get("chip_size", 256)
    max_grad_norm = hyperparameters.get("max_grad_norm", 1.0)
    scheduler_name = hyperparameters.get("scheduler", "cosine")
    freeze_encoder = hyperparameters.get("freeze_encoder", True)
    val_ratio = split_info["val_ratio"]
    seed = split_info["seed"]

    with mlflow_training_context(hyperparameters, model_name, base_model_id, dataset_id):
        device = _get_device()
        model = resnet18(weights=None)
        local_path = _download_checkpoint(base_model_weights)
        state = torch.load(local_path, map_location=device, weights_only=True)
        model.load_state_dict(state, strict=False)
        if freeze_encoder:
            for param in model.parameters():
                param.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, 1)
        model.to(device)

        all_samples = _load_samples(dataset_chips, dataset_labels)
        train_samples, val_samples = _split_samples(all_samples, val_ratio, seed)
        train_loader = _build_classification_dataset(train_samples, chip_size, batch_size)
        val_loader = _build_classification_dataset(val_samples, chip_size, batch_size, shuffle=False)
        criterion = nn.BCEWithLogitsLoss()
        trainable = filter(lambda p: p.requires_grad, model.parameters())
        opt = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=weight_decay)

        scheduler = None
        if scheduler_name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

        train_losses: list[float] = []
        val_losses: list[float] = []

        model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            count = 0
            for batch in train_loader:
                images, labels = preprocess(batch)
                images, labels = images.to(device), labels.to(device)
                logits = model(images).squeeze(-1)
                batch_loss = criterion(logits, labels)
                opt.zero_grad()
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                opt.step()
                total_loss += batch_loss.item()
                count += 1
            if scheduler:
                scheduler.step()
            avg_train_loss = total_loss / max(count, 1)

            model.eval()
            val_total = 0.0
            val_count = 0
            with torch.no_grad():
                for batch in val_loader:
                    images, labels = preprocess(batch)
                    images, labels = images.to(device), labels.to(device)
                    logits = model(images).squeeze(-1)
                    val_total += criterion(logits, labels).item()
                    val_count += 1
            avg_val_loss = val_total / max(val_count, 1)
            model.train()

            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)

            import mlflow

            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)  # ty: ignore[possibly-missing-attribute]
            mlflow.log_metric("val_loss", avg_val_loss, step=epoch)  # ty: ignore[possibly-missing-attribute]
            log_metadata(metadata={"loss": avg_train_loss, "epoch": epoch + 1})

        from fair.zenml.metrics import log_loss_history

        log_loss_history(train_losses, val_losses)

    return model.cpu()


@step
def evaluate_model(
    trained_model: Any,
    dataset_chips: str,
    dataset_labels: str,
    hyperparameters: dict[str, Any],
    split_info: dict[str, Any],
    class_names: list[str] | None = None,
) -> Annotated[dict[str, Any], "metrics"]:
    import torch

    chip_size = hyperparameters.get("chip_size", 256)
    val_ratio = split_info["val_ratio"]
    seed = split_info["seed"]

    device = _get_device()
    model = trained_model.to(device)
    model.eval()

    all_samples = _load_samples(dataset_chips, dataset_labels)
    _, val_samples = _split_samples(all_samples, val_ratio, seed)
    loader = _build_classification_dataset(val_samples, chip_size, shuffle=False)
    tp = fp = tn = fn = 0

    with torch.no_grad():
        for batch in loader:
            images, labels = preprocess(batch)
            images, labels = images.to(device), labels.to(device)
            preds = (torch.sigmoid(model(images).squeeze(-1)) > 0.5).float()
            tp += ((preds == 1) & (labels == 1)).sum().item()
            fp += ((preds == 1) & (labels == 0)).sum().item()
            tn += ((preds == 0) & (labels == 0)).sum().item()
            fn += ((preds == 0) & (labels == 1)).sum().item()

    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    metrics = {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
    log_evaluation_results(metrics)
    return metrics


@step
def export_onnx(
    trained_model: Any,
    hyperparameters: dict[str, Any],
) -> Annotated[bytes, "onnx_model"]:
    import os
    import tempfile

    import onnx
    import torch

    chip_size = hyperparameters.get("chip_size", 256)

    model = trained_model.cpu()
    model.eval()
    dummy = torch.randn(1, 3, chip_size, chip_size)
    fd, path = tempfile.mkstemp(suffix=".onnx")
    os.close(fd)
    try:
        torch.onnx.export(
            model,
            (dummy,),
            path,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            opset_version=18,
        )
        proto = onnx.load(path)
        onnx.save_model(proto, path, save_as_external_data=False)
        onnx.checker.check_model(path)
        return Path(path).read_bytes()
    finally:
        Path(path).unlink(missing_ok=True)


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
    export_onnx(trained_model=trained_model, hyperparameters=hyperparameters)


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
def inference_pipeline(
    model_uri: str,
    input_images: str,
    inference_params: dict[str, Any] | None = None,
) -> None:
    run_inference(
        model_uri=model_uri,
        input_images=input_images,
        inference_params=inference_params or {},
    )
