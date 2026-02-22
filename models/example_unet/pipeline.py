"""ZenML pipeline for UNet building segmentation.

Entrypoints referenced by models/example_unet/stac-item.json.
Pretrained weights: OAM-TCD (arxiv.org/abs/2407.11743).
"""

from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from annotated_types import Ge, Le
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader
from torchgeo.datasets import OpenAerialMap, OpenStreetMap, stack_samples
from torchgeo.models import Unet_Weights, unet
from torchgeo.samplers import RandomGeoSampler, Units
from zenml import log_metadata, pipeline, step

_OptimizerFactory = Any  # Callable[[Iterable, ...], Optimizer] — avoids base-class __init__ overload
_LossFactory = Any  # Callable[[], nn.Module]

DEVICE: Literal["mps", "cuda", "cpu"] = (
    "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
)
_ENCODER = "resnet50"
_BUILDING_CLASSES = [{"name": "building", "selector": [{"building": "*"}]}]


def _resolve_weights(weight_id: str) -> Unet_Weights:
    """Map 'torchgeo.models.Unet_Weights.MEMBER' or bare MEMBER name to enum."""
    return Unet_Weights[weight_id.rsplit(".", 1)[-1]]


def preprocess(batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
    images = batch["image"].float() / 255.0
    masks = batch["mask"].long().squeeze(1)
    return images, masks


def postprocess(logits: Tensor) -> np.ndarray:
    return logits.argmax(dim=1).cpu().numpy().astype(np.uint8)


def _build_dataset(chips_path: str, labels_path: str, chip_size: int, length: int, batch_size: int = 4) -> DataLoader:
    """Intersect OAM + OSM GeoDatasets. chip_size in pixels; bounds are slices per torchgeo 0.10.x dev.
    labels_path is the exact GeoJSON file path stored in STAC; OpenStreetMap.paths requires its parent dir.
    """
    oam = OpenAerialMap(paths=chips_path, download=False)
    b = oam.bounds
    bbox = (b[0].start, b[1].start, b[0].stop, b[1].stop)
    osm = OpenStreetMap(bbox=bbox, classes=_BUILDING_CLASSES, paths=str(Path(labels_path).parent), download=False)
    dataset = oam & osm
    sampler = RandomGeoSampler(dataset, size=chip_size, length=length, units=Units.PIXELS)
    return DataLoader(dataset, sampler=sampler, batch_size=batch_size, collate_fn=stack_samples)


# Keep in sync with mlm:hyperparameters in stac-item.json.
_OPTIMIZERS: dict[str, _OptimizerFactory] = {
    "Adam": torch.optim.Adam,
    "AdamW": torch.optim.AdamW,
    "SGD": torch.optim.SGD,
}
_LOSSES: dict[str, _LossFactory] = {
    "CrossEntropyLoss": nn.CrossEntropyLoss,
    "BCEWithLogitsLoss": nn.BCEWithLogitsLoss,
}


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
    num_classes: int,
    optimizer: str = "AdamW",
    loss: str = "CrossEntropyLoss",
) -> str:
    """Finetune UNet from OAM-TCD pretrained weights. Returns saved weights path."""
    model = unet(weights=_resolve_weights(base_model_weights), classes=num_classes).to(DEVICE)
    loader = _build_dataset(dataset_chips, dataset_labels, chip_size, length=10, batch_size=batch_size)

    criterion = _LOSSES[loss]()
    opt = _OPTIMIZERS[optimizer](model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    model.train()
    for epoch in range(epochs):
        total_loss = sum(_train_step(model, batch, criterion, opt) for batch in loader)
        log_metadata(metadata={"loss": total_loss / len(loader), "epoch": epoch + 1})

    save_path = Path("artifacts") / "finetuned_weights.pth"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    return str(save_path)


def _train_step(
    model: nn.Module,
    batch: dict[str, Tensor],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> float:
    images, masks = preprocess(batch)
    images, masks = images.to(DEVICE), masks.to(DEVICE)
    loss = criterion(model(images), masks)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


@step
def evaluate_model(
    model_path: str,
    dataset_chips: str,
    dataset_labels: str,
    chip_size: int = 512,
    num_classes: int = 2,
) -> dict[str, Any]:
    """Pixel accuracy and per-class IoU on the dataset."""
    model = unet(encoder_name=_ENCODER, classes=num_classes).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    model.eval()

    loader = _build_dataset(dataset_chips, dataset_labels, chip_size, length=5)
    total_correct = total_pixels = 0
    intersection = [0] * num_classes
    union = [0] * num_classes

    with torch.no_grad():
        for batch in loader:
            images, masks = preprocess(batch)
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            preds = model(images).argmax(dim=1)
            total_correct += (preds == masks).sum().item()
            total_pixels += masks.numel()
            for c in range(num_classes):
                intersection[c] += ((preds == c) & (masks == c)).sum().item()
                union[c] += ((preds == c) | (masks == c)).sum().item()

    metrics: dict[str, Any] = {
        "accuracy": total_correct / max(total_pixels, 1),
        "mean_iou": sum(intersection[c] / max(union[c], 1) for c in range(num_classes)) / num_classes,
        **{f"iou_class_{c}": intersection[c] / max(union[c], 1) for c in range(num_classes)},
    }
    log_metadata(metadata=metrics)
    return metrics


@step
def run_inference(
    model_weights: str,
    input_images: str,
    chip_size: int,
    num_classes: int,
) -> str:
    """Run inference on PNG images in input dir. Returns predictions dir path."""
    model = unet(encoder_name=_ENCODER, classes=num_classes).to(DEVICE)
    model.load_state_dict(torch.load(model_weights, map_location=DEVICE, weights_only=True))
    model.eval()

    input_dir = Path(input_images)
    output_dir = input_dir.parent / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        patterns = ("*.png", "*.tif", "*.tiff")
        img_paths = sorted(p for pat in patterns for p in input_dir.glob(pat))
        for img_path in img_paths:
            img = np.array(Image.open(img_path).convert("RGB"))
            tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            if tensor.shape[-2:] != (chip_size, chip_size):
                tensor = F.interpolate(tensor, size=(chip_size, chip_size), mode="bilinear", align_corners=False)
            mask = postprocess(model(tensor.to(DEVICE)))[0]
            Image.fromarray(mask).save(output_dir / img_path.name)

    return str(output_dir)


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
    num_classes: Annotated[int, Ge(2), Le(256)],
    optimizer: Literal["Adam", "AdamW", "SGD"] = "AdamW",
    loss: Literal["CrossEntropyLoss", "BCEWithLogitsLoss"] = "CrossEntropyLoss",
) -> None:
    """Full training pipeline: finetune -> evaluate."""
    model_path = train_model(
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
        model_path=model_path,
        dataset_chips=dataset_chips,
        dataset_labels=dataset_labels,
        chip_size=chip_size,
        num_classes=num_classes,
    )


@pipeline
def inference_pipeline(
    model_weights: str,
    input_images: str,
    chip_size: Annotated[
        int, Ge(64), Le(1024)
    ],  # if chip_size is >1024 , we might need to adjust how to handle big images in single chip
    num_classes: Annotated[int, Ge(1), Le(256)],
) -> None:
    """Inference pipeline: predict on input images."""
    run_inference(
        model_weights=model_weights,
        input_images=input_images,
        chip_size=chip_size,
        num_classes=num_classes,
    )
