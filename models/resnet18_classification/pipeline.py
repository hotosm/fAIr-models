"""ZenML pipeline for ResNet18 binary building classification.

Entrypoints referenced by models/resnet18_classification/stac-item.json.
Pretrained backbone: torchvision ResNet18 ImageNet.
"""

import csv
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


def preprocess(batch: dict[str, Any]) -> tuple[Any, Any]:
    images = batch["image"].float() / 255.0
    labels = batch["label"].float()
    return images, labels


def postprocess(logits: Any) -> Any:
    import torch

    return (torch.sigmoid(logits) > 0.5).int().cpu().numpy()


def _build_classification_dataset(
    chips_path: str,
    labels_csv_path: str,
    chip_size: int,
    batch_size: int = 4,
) -> Any:
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms

    from fair.utils.data import resolve_directory, resolve_path

    local_chips = resolve_directory(chips_path)
    local_csv = resolve_path(labels_csv_path)

    transform = transforms.Compose(
        [
            transforms.Resize((chip_size, chip_size)),
            transforms.ToTensor(),
        ]
    )

    class ChipDataset(Dataset):
        def __init__(self) -> None:
            self.samples: list[tuple[Path, int]] = []
            with open(local_csv, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    img_path = local_chips / row["filename"]
                    label = 1 if row["class_name"] == "building" else 0
                    if img_path.exists():
                        self.samples.append((img_path, label))

        def __len__(self) -> int:
            return len(self.samples)

        def __getitem__(self, idx: int) -> dict[str, Any]:  # ty: ignore[invalid-method-override]
            path, label = self.samples[idx]
            img = Image.open(path).convert("RGB")
            tensor = transform(img)
            return {"image": tensor, "label": torch.tensor(label, dtype=torch.float32)}

    dataset = ChipDataset()
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


@step
def train_model(
    dataset_chips: str,
    dataset_labels: str,
    base_model_weights: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    chip_size: int,
    num_classes: int = 2,
    optimizer: str = "AdamW",
    loss: str = "BCEWithLogitsLoss",
) -> Any:
    import mlflow
    import torch
    import torch.nn as nn
    from torchvision.models import ResNet18_Weights, resnet18

    mlflow.autolog()  # ty: ignore[possibly-missing-attribute]
    mlflow.log_params(  # ty: ignore[possibly-missing-attribute]
        {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "optimizer": optimizer,
            "loss": loss,
            "chip_size": chip_size,
        }
    )

    device = _get_device()
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.to(device)

    loader = _build_classification_dataset(dataset_chips, dataset_labels, chip_size, batch_size)
    criterion = nn.BCEWithLogitsLoss()
    trainable = filter(lambda p: p.requires_grad, model.parameters())
    opt = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=weight_decay)

    model.train()
    wall_start = time.perf_counter()
    for epoch in range(epochs):
        total_loss = 0.0
        count = 0
        for batch in loader:
            images, labels = preprocess(batch)
            images, labels = images.to(device), labels.to(device)
            logits = model(images).squeeze(-1)
            batch_loss = criterion(logits, labels)
            opt.zero_grad()
            batch_loss.backward()
            opt.step()
            total_loss += batch_loss.item()
            count += 1
        avg_loss = total_loss / max(count, 1)
        mlflow.log_metric("train_loss", avg_loss, step=epoch)  # ty: ignore[possibly-missing-attribute]
        log_metadata(metadata={"loss": avg_loss, "epoch": epoch + 1})
    wall_seconds = time.perf_counter() - wall_start
    log_training_wall_time(wall_seconds)

    return model.cpu()


@step
def evaluate_model(
    trained_model: Any,
    dataset_chips: str,
    dataset_labels: str,
    chip_size: int = 256,
    class_names: list[str] | None = None,
) -> dict[str, Any]:
    import mlflow
    import torch

    device = _get_device()
    model = trained_model.to(device)
    model.eval()

    loader = _build_classification_dataset(dataset_chips, dataset_labels, chip_size)
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

    mlflow.log_metrics(  # ty: ignore[possibly-missing-attribute]
        {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    )
    metrics = {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
    log_fair_metrics(metrics)
    return metrics


@step
def export_onnx(trained_model: Any, chip_size: int = 256) -> str:
    import os
    import tempfile

    import torch

    model = trained_model.cpu()
    model.eval()
    dummy = torch.randn(1, 3, chip_size, chip_size)
    fd, path = tempfile.mkstemp(suffix=".onnx")
    os.close(fd)
    torch.onnx.export(
        model,
        (dummy,),
        path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=17,
    )
    return path


@pipeline
def training_pipeline(
    base_model_weights: str,
    dataset_chips: str,
    dataset_labels: str,
    epochs: Annotated[int, Ge(1), Le(1000)],
    batch_size: Annotated[int, Ge(1), Le(64)],
    learning_rate: Annotated[float, Ge(1e-6), Le(1.0)],
    weight_decay: Annotated[float, Ge(0.0), Le(1.0)],
    chip_size: Annotated[int, Ge(64), Le(2048)],
    num_classes: Annotated[int, Ge(2), Le(256)] = 2,
    optimizer: Literal["AdamW"] = "AdamW",
    loss: Literal["BCEWithLogitsLoss"] = "BCEWithLogitsLoss",
    class_names: list[str] | None = None,
) -> None:
    trained_model = train_model(
        dataset_chips=dataset_chips,
        dataset_labels=dataset_labels,
        base_model_weights=base_model_weights,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        chip_size=chip_size,
        num_classes=num_classes,
        optimizer=optimizer,
        loss=loss,
    )
    evaluate_model(
        trained_model=trained_model,
        dataset_chips=dataset_chips,
        dataset_labels=dataset_labels,
        chip_size=chip_size,
        class_names=class_names,
    )
    export_onnx(trained_model=trained_model, chip_size=chip_size)


@pipeline
def inference_pipeline(
    model_uri: str,
    input_images: str,
    chip_size: Annotated[int, Ge(64), Le(1024)] = 256,
    num_classes: Annotated[int, Ge(1), Le(256)] = 2,
    zenml_artifact_version_id: str = "",
) -> None:
    model = load_model(model_uri=model_uri, zenml_artifact_version_id=zenml_artifact_version_id)
    classify(model=model, input_images=input_images, chip_size=chip_size)


@step
def classify(
    model: Any,
    input_images: str,
    chip_size: int = 256,
) -> dict[str, Any]:
    import torch
    from PIL import Image
    from torchvision import transforms

    from fair.utils.data import resolve_directory

    device = _get_device()
    model = model.to(device)
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize((chip_size, chip_size)),
            transforms.ToTensor(),
        ]
    )

    input_dir = resolve_directory(input_images)
    patterns = ("*.png", "*.tif", "*.tiff", "*.jpg")
    img_paths = sorted(p for pat in patterns for p in input_dir.glob(pat))
    if not img_paths:
        msg = f"No input images found in {input_dir}"
        raise FileNotFoundError(msg)

    results: dict[str, str] = {}
    with torch.no_grad():
        for img_path in img_paths:
            img = Image.open(img_path).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(device)
            logit = model(tensor).squeeze()
            label = "building" if torch.sigmoid(logit).item() > 0.5 else "no_building"
            results[img_path.name] = label

    return {"predictions": results}
