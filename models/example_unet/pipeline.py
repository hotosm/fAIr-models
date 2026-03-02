"""ZenML pipeline for UNet building segmentation.

Entrypoints referenced by models/example_unet/stac-item.json.
Pretrained weights: OAM-TCD (arxiv.org/abs/2407.11743).
"""

from typing import Annotated, Any, Literal

from annotated_types import Ge, Le
from zenml import log_metadata, pipeline, step

from fair.zenml.steps import load_model

_BUILDING_CLASSES = [{"name": "building", "selector": [{"building": "*"}]}]


def _get_device() -> Literal["mps", "cuda", "cpu"]:
    import torch

    return "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_weights(weight_id: str) -> Any:
    from torchgeo.models import Unet_Weights

    return Unet_Weights[weight_id.rsplit(".", 1)[-1]]


def preprocess(batch: dict[str, Any]) -> tuple[Any, Any]:
    images = batch["image"].float() / 255.0
    masks = batch["mask"].long().squeeze(1)
    return images, masks


def postprocess(logits: Any) -> Any:
    import numpy as np

    return logits.argmax(dim=1).cpu().numpy().astype(np.uint8)


def _build_dataset(chips_path: str, labels_path: str, chip_size: int, length: int, batch_size: int = 4) -> Any:
    """Intersect OAM + OSM GeoDatasets. chip_size in pixels; bounds are slices per torchgeo 0.10.x dev.
    labels_path is the exact GeoJSON file path stored in STAC; OpenStreetMap.paths requires its parent dir.
    Downloads chips and labels to local cache via UPath/fsspec.
    """
    from torch.utils.data import DataLoader
    from torchgeo.datasets import OpenStreetMap, RasterDataset, stack_samples
    from torchgeo.samplers import RandomGeoSampler, Units

    from fair.utils.data import resolve_directory, resolve_path

    local_chips = str(resolve_directory(chips_path, "OAM-*.tif"))
    local_labels = resolve_path(labels_path)

    class _OAMDataset(RasterDataset):  # TODO : After OAM is released , replace this with OAM dataset directly
        filename_glob = "OAM-*.tif"
        filename_regex = r"^OAM-(?P<x>\d+)-(?P<y>\d+)-(?P<z>\d+)\.tif$"
        is_image = True
        separate_files = False

    oam = _OAMDataset(paths=local_chips)
    b = oam.bounds
    bbox = (b[0].start, b[1].start, b[0].stop, b[1].stop)
    osm = OpenStreetMap(bbox=bbox, classes=_BUILDING_CLASSES, paths=str(local_labels.parent), download=False)
    dataset = oam & osm
    sampler = RandomGeoSampler(dataset, size=chip_size, length=length, units=Units.PIXELS)
    return DataLoader(dataset, sampler=sampler, batch_size=batch_size, collate_fn=stack_samples)


def _get_optimizers() -> dict[str, Any]:
    import torch

    return {
        "Adam": torch.optim.Adam,
        "AdamW": torch.optim.AdamW,
        "SGD": torch.optim.SGD,
    }


def _get_losses() -> dict[str, Any]:
    import torch.nn as nn

    return {
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
) -> Any:
    import mlflow
    from torchgeo.models import unet

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
            "num_classes": num_classes,
        }
    )

    device = _get_device()
    model = unet(weights=_resolve_weights(base_model_weights), classes=num_classes).to(device)
    loader = _build_dataset(dataset_chips, dataset_labels, chip_size, length=10, batch_size=batch_size)

    losses = _get_losses()
    optimizers = _get_optimizers()
    criterion = losses[loss]()
    opt = optimizers[optimizer](model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    model.train()
    for epoch in range(epochs):
        total_loss = sum(_train_step(model, batch, criterion, opt, device) for batch in loader)
        avg_loss = total_loss / len(loader)
        mlflow.log_metric("train_loss", avg_loss, step=epoch)  # ty: ignore[possibly-missing-attribute]
        log_metadata(metadata={"loss": avg_loss, "epoch": epoch + 1})

    return model.cpu()


def _train_step(
    model: Any,
    batch: dict[str, Any],
    criterion: Any,
    optimizer: Any,
    device: str,
) -> float:
    images, masks = preprocess(batch)
    images, masks = images.to(device), masks.to(device)
    loss = criterion(model(images), masks)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


@step
def evaluate_model(
    trained_model: Any,
    dataset_chips: str,
    dataset_labels: str,
    chip_size: int = 512,
    num_classes: int = 2,
) -> dict[str, Any]:
    import mlflow
    import torch

    device = _get_device()
    model = trained_model.to(device)
    model.eval()

    loader = _build_dataset(dataset_chips, dataset_labels, chip_size, length=5)
    total_correct = total_pixels = 0
    intersection = [0] * num_classes
    union = [0] * num_classes

    with torch.no_grad():
        for batch in loader:
            images, masks = preprocess(batch)
            images, masks = images.to(device), masks.to(device)
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
    mlflow.log_metrics(metrics)  # ty: ignore[possibly-missing-attribute]
    log_metadata(metadata=metrics)
    return metrics


@step
def load_base_model(
    model_uri: str,
    num_classes: int,
) -> Any:
    from torchgeo.models import unet

    return unet(weights=_resolve_weights(model_uri), classes=num_classes).cpu()


@step
def run_inference(
    model: Any,
    input_images: str,
    chip_size: int,
    num_classes: int,
) -> str:
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image

    from fair.utils.data import resolve_directory

    device = _get_device()
    model = model.to(device)
    model.eval()

    input_dir = resolve_directory(input_images)
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
            mask = postprocess(model(tensor.to(device)))[0]
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
        num_classes=num_classes,
    )


@pipeline
def inference_pipeline(
    model_uri: str,
    input_images: str,
    chip_size: Annotated[
        int, Ge(64), Le(1024)
    ],  # if chip_size is >1024 , we might need to adjust how to handle big images in single chip
    num_classes: Annotated[int, Ge(1), Le(256)],
    zenml_artifact_version_id: str = "",
    use_base_model: bool = False,
) -> None:
    """Inference pipeline: load model then predict. Supports both base and finetuned models."""
    if use_base_model:
        model = load_base_model(model_uri=model_uri, num_classes=num_classes)
    else:
        model = load_model(model_uri=model_uri, zenml_artifact_version_id=zenml_artifact_version_id)
    run_inference(
        model=model,
        input_images=input_images,
        chip_size=chip_size,
        num_classes=num_classes,
    )
