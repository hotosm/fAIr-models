"""ZenML pipeline for UNet building segmentation.

Entrypoints referenced by models/unet_segmentation/stac-item.json.
Pretrained weights: OAM-TCD (arxiv.org/abs/2407.11743).
"""

import tempfile
from pathlib import Path
from typing import Annotated, Any

from zenml import log_metadata, pipeline, step

from fair.zenml.instrumentation import log_evaluation_results, mlflow_training_context
from fair.zenml.steps import load_model

_BUILDING_CLASSES = [{"name": "building", "selector": [{"building": "*"}]}]


def _get_device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _vectorize_segmentation_mask(mask, transform, crs):
    import numpy as np
    import rasterio.features
    from pyproj import Transformer

    mask_uint8 = mask.astype(np.uint8)
    needs_reproject = crs is not None and str(crs) != "EPSG:4326"
    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True) if needs_reproject else None

    features = []
    for geom, value in rasterio.features.shapes(mask_uint8, transform=transform):
        if value == 0:
            continue
        if transformer:
            coords = geom["coordinates"]
            geom["coordinates"] = [[list(transformer.transform(x, y)) for x, y in ring] for ring in coords]
        features.append({"type": "Feature", "properties": {"class": int(value)}, "geometry": geom})
    return features


def _build_feature_collection(features):
    return {"type": "FeatureCollection", "features": features}


def _resolve_weights(weight_id: str) -> Any:
    if "://" in weight_id or "/" in weight_id:
        return None

    from torchgeo.models import Unet_Weights

    return Unet_Weights[weight_id.rsplit(".", 1)[-1]]


def _download_s3(uri: str) -> Path:
    from upath import UPath

    local_path = Path(tempfile.mkdtemp()) / UPath(uri).name
    local_path.write_bytes(UPath(uri).read_bytes())
    return local_path


def resolve_weights(weight_id: str) -> Path:
    import torch

    weights = _resolve_weights(weight_id)
    checkpoint_path = Path(tempfile.mkdtemp()) / "unet_pretrained.pth"
    torch.hub.download_url_to_file(weights.url, str(checkpoint_path))
    return checkpoint_path


def preprocess(batch: dict[str, Any]) -> tuple[Any, Any]:
    images = batch["image"].float() / 255.0
    masks = batch["mask"].long().squeeze(1)
    return images, masks


def postprocess(logits: Any) -> Any:
    import numpy as np

    return logits.argmax(dim=1).cpu().numpy().astype(np.uint8)


def _build_dataset(
    chips_path: str,
    labels_path: str,
    chip_size: int,
    length: int,
    batch_size: int = 4,
    split: str = "train",
    seed: int = 42,
) -> Any:
    """Intersect OAM + OSM GeoDatasets. chip_size in pixels; bounds are slices per torchgeo 0.10.x dev.
    Downloads chips and labels to local cache via UPath/fsspec.
    """
    from torch.utils.data import DataLoader
    from torchgeo.datasets import OpenStreetMap, RasterDataset, stack_samples
    from torchgeo.samplers import GridGeoSampler, RandomGeoSampler, Units

    from fair.utils.data import resolve_directory

    local_chips = str(resolve_directory(chips_path, "OAM-*"))
    local_labels_dir = str(resolve_directory(labels_path, "*.geojson"))

    class _OAMDataset(RasterDataset):  # TODO : After OAM is released , replace this with OAM dataset directly
        filename_glob = "OAM-*.tif"
        filename_regex = r"^OAM-(?P<x>\d+)-(?P<y>\d+)-(?P<z>\d+)\.tif$"
        is_image = True
        separate_files = False

    oam = _OAMDataset(paths=local_chips)
    b = oam.bounds
    bbox = (b[0].start, b[1].start, b[0].stop, b[1].stop)
    osm = OpenStreetMap(bbox=bbox, classes=_BUILDING_CLASSES, paths=str(local_labels_dir), download=False)
    dataset = oam & osm

    if split == "val":
        sampler = GridGeoSampler(dataset, size=chip_size, stride=chip_size, units=Units.PIXELS)
        return DataLoader(dataset, sampler=sampler, batch_size=batch_size, collate_fn=stack_samples)

    import torch

    generator = torch.Generator().manual_seed(seed)
    sampler = RandomGeoSampler(dataset, size=chip_size, length=length, units=Units.PIXELS, generator=generator)
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


def _train_step(
    model: Any,
    batch: dict[str, Any],
    criterion: Any,
    optimizer: Any,
    device: str,
    max_grad_norm: float,
) -> float:
    import torch

    images, masks = preprocess(batch)
    images, masks = images.to(device), masks.to(device)
    loss = criterion(model(images), masks)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
    optimizer.step()
    return loss.item()


@step
def split_dataset(
    dataset_chips: str,
    dataset_labels: str,
    hyperparameters: dict[str, Any],
) -> Annotated[dict[str, Any], "split_info"]:
    val_ratio = hyperparameters.get("val_ratio", 0.2)
    seed = hyperparameters.get("split_seed", 42)
    samples_per_epoch = hyperparameters.get("samples_per_epoch", 50)
    val_samples = max(int(samples_per_epoch * val_ratio), 10)

    split_info = {
        "strategy": "spatial",
        "val_ratio": val_ratio,
        "seed": seed,
        "train_count": samples_per_epoch,
        "val_count": val_samples,
        "description": "Spatial split: RandomGeoSampler for train, GridGeoSampler for val (non-overlapping tiles)",
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
    num_classes: int,
    model_name: str | None = None,
    base_model_id: str | None = None,
    dataset_id: str | None = None,
) -> Annotated[Any, "trained_model"]:
    from torchgeo.models import unet

    epochs = hyperparameters["epochs"]
    batch_size = hyperparameters.get("batch_size", 4)
    learning_rate = hyperparameters.get("learning_rate", 0.0001)
    weight_decay = hyperparameters.get("weight_decay", 0.0001)
    chip_size = hyperparameters.get("chip_size", 256)
    samples_per_epoch = hyperparameters.get("samples_per_epoch", 50)
    optimizer_name = hyperparameters.get("optimizer", "AdamW")
    loss_name = hyperparameters.get("loss", "CrossEntropyLoss")
    max_grad_norm = hyperparameters.get("max_grad_norm", 1.0)
    scheduler_name = hyperparameters.get("scheduler", "cosine")
    freeze_encoder = hyperparameters.get("freeze_encoder", True)
    seed = split_info["seed"]

    with mlflow_training_context(hyperparameters, model_name, base_model_id, dataset_id):
        device = _get_device()
        resolved = _resolve_weights(base_model_weights)
        model = unet(weights=resolved, classes=num_classes)
        if resolved is None:
            import torch

            local_path = _download_s3(base_model_weights)
            state = torch.load(local_path, map_location=device, weights_only=True)
            model.load_state_dict(state, strict=False)
        model.to(device)
        if freeze_encoder:
            encoder = getattr(model, "encoder", None)
            if encoder is not None:
                for param in encoder.parameters():
                    param.requires_grad = False
        loader = _build_dataset(
            dataset_chips,
            dataset_labels,
            chip_size,
            length=samples_per_epoch,
            batch_size=batch_size,
            split="train",
            seed=seed,
        )

        losses = _get_losses()
        optimizers = _get_optimizers()
        criterion = losses[loss_name]()
        trainable = filter(lambda p: p.requires_grad, model.parameters())
        opt = optimizers[optimizer_name](trainable, lr=learning_rate, weight_decay=weight_decay)

        import torch

        scheduler = None
        if scheduler_name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

        model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for batch in loader:
                total_loss += _train_step(model, batch, criterion, opt, device, max_grad_norm)
            if scheduler:
                scheduler.step()
            avg_loss = total_loss / len(loader)
            import mlflow

            mlflow.log_metric("train_loss", avg_loss, step=epoch)  # ty: ignore[possibly-missing-attribute]
            log_metadata(metadata={"loss": avg_loss, "epoch": epoch + 1})

    return model.cpu()


@step
def evaluate_model(
    trained_model: Any,
    dataset_chips: str,
    dataset_labels: str,
    hyperparameters: dict[str, Any],
    split_info: dict[str, Any],
    num_classes: int = 2,
    class_names: list[str] | None = None,
) -> Annotated[dict[str, Any], "metrics"]:
    import torch

    chip_size = hyperparameters.get("chip_size", 256)

    device = _get_device()
    model = trained_model.to(device)
    model.eval()

    loader = _build_dataset(
        dataset_chips,
        dataset_labels,
        chip_size,
        length=0,
        split="val",
        seed=split_info["seed"],
    )
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

    resolved_names = (
        class_names if class_names and len(class_names) == num_classes else [str(c) for c in range(num_classes)]
    )
    per_class_iou = {resolved_names[c]: intersection[c] / max(union[c], 1) for c in range(num_classes)}

    accuracy = total_correct / max(total_pixels, 1)
    mean_iou = sum(per_class_iou.values()) / num_classes
    metrics = {"accuracy": accuracy, "mean_iou": mean_iou, "per_class_iou": per_class_iou}
    log_evaluation_results(metrics)
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
) -> Annotated[dict[str, Any], "predictions"]:
    import rasterio
    import torch
    import torch.nn.functional as F

    from fair.utils.data import resolve_directory

    device = _get_device()
    model = model.to(device)
    model.eval()

    input_dir = resolve_directory(input_images)
    patterns = ("*.png", "*.tif", "*.tiff")
    img_paths = sorted(p for pat in patterns for p in input_dir.glob(pat))
    if not img_paths:
        msg = f"No input images found in {input_dir}"
        raise FileNotFoundError(msg)

    all_features: list[dict[str, Any]] = []
    with torch.no_grad():
        for img_path in img_paths:
            with rasterio.open(img_path) as src:
                img = src.read([1, 2, 3]).transpose(1, 2, 0)
                transform = src.transform
                crs = src.crs

            tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            if tensor.shape[-2:] != (chip_size, chip_size):
                tensor = F.interpolate(tensor, size=(chip_size, chip_size), mode="bilinear", align_corners=False)

            mask = postprocess(model(tensor.to(device)))[0]
            all_features.extend(_vectorize_segmentation_mask(mask, transform, crs))

    return _build_feature_collection(all_features)


@step
def export_onnx(
    trained_model: Any,
    hyperparameters: dict[str, Any],
    num_classes: int = 2,
) -> Annotated[str, "onnx_model"]:
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
    torch.onnx.export(
        model,
        (dummy,),
        path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=18,
    )
    onnx.checker.check_model(path)
    return path


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
        num_classes=num_classes,
    )
    export_onnx(
        trained_model=trained_model,
        hyperparameters=hyperparameters,
        num_classes=num_classes,
    )


@pipeline
def inference_pipeline(
    model_uri: str,
    input_images: str,
    chip_size: int,
    num_classes: int,
    zenml_artifact_version_id: str = "",
    use_base_model: bool = False,
) -> None:
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
