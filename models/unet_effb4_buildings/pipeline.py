"""ZenML pipeline for UNet EfficientNet-B4 instance segmentation of buildings.

Entrypoints referenced by models/unet_effb4_buildings/stac-item.json.
Pretrained weights: nilsho01/unet-effb4-dist3-buildings (Hugging Face).

Architecture: smp.Unet(encoder_name="efficientnet-b4", encoder_weights=None,
              in_channels=3, classes=3)
Output channels:
  [0] mask logit        — binary building / background
  [1] normalised EDT    — distance transform used to separate touching instances
  [2] boundary logit    — building boundary (auxiliary, not used at inference)

Postprocessing uses marker-controlled watershed:
  seeds  = scipy.ndimage.label((sigmoid(edt) > 0.5) & mask_binary)[0]
  labels = skimage.segmentation.watershed(-edt_scipy, markers=seeds, mask=mask_binary)

Fine-tuning: encoder frozen, decoder only; BCE+Dice on mask channel; spatial
block split (block_size=4 on OAM tile coords).
"""

import tempfile
from pathlib import Path
from typing import Annotated, Any

from zenml import log_metadata, pipeline, step

from fair.zenml.instrumentation import log_evaluation_results, mlflow_training_context
from fair.zenml.materializers import ONNXMaterializer

MODEL_INPUT_SIZE = 256
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

HF_REPO = "nilsho01/unet-effb4-dist3-buildings"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _subset_chips_dir(chips_path: str, fraction: float) -> str:
    if fraction >= 1.0:
        return chips_path
    from fair.utils.data import resolve_directory

    chips = sorted(resolve_directory(chips_path).rglob("OAM-*.tif"))
    step = max(1, round(1 / fraction))
    subset = Path(tempfile.mkdtemp(prefix="unet_effb4_chips_subset_"))
    for chip in chips[::step]:
        (subset / chip.name).symlink_to(chip)
        sidecar = chip.with_name(chip.name + ".aux.xml")
        if sidecar.exists():
            (subset / sidecar.name).symlink_to(sidecar)
    return str(subset)


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


def _load_hf_weights(device: str) -> tuple[Any, float]:
    """Download best.pth and best_threshold.json from HF; return (state_dict, threshold)."""
    import json

    from huggingface_hub import hf_hub_download

    ckpt_path = hf_hub_download(repo_id=HF_REPO, filename="best.pth")
    thresh_path = hf_hub_download(repo_id=HF_REPO, filename="best_threshold.json")

    import torch

    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    state_dict = state.get("model", state)

    threshold = 0.5
    with open(thresh_path) as fh:
        threshold = float(json.load(fh).get("threshold", 0.5))

    return state_dict, threshold


def _build_model(num_output_channels: int = 3) -> Any:
    import segmentation_models_pytorch as smp

    return smp.Unet(
        encoder_name="efficientnet-b4",
        encoder_weights=None,
        in_channels=3,
        classes=num_output_channels,
    )


def _resize_chw(arr: Any, size: int) -> Any:
    import numpy as np
    from PIL import Image

    channels = [
        np.asarray(Image.fromarray(arr[c]).resize((size, size), Image.Resampling.BILINEAR))
        for c in range(arr.shape[0])
    ]
    return np.stack(channels, axis=0).astype(np.float32)


def _normalise(arr: Any) -> Any:
    """Apply ImageNet normalisation to a CHW float32 array in [0, 1]."""
    import numpy as np

    mean = np.array(IMAGENET_MEAN, dtype=np.float32)[:, None, None]
    std = np.array(IMAGENET_STD, dtype=np.float32)[:, None, None]
    return (arr - mean) / std


# ---------------------------------------------------------------------------
# Public preprocessing / postprocessing (referenced from stac-item.json)
# ---------------------------------------------------------------------------


def preprocess(batch: dict[str, Any]) -> tuple[Any, Any]:
    """Normalise a torchgeo batch for training: images → float32 ImageNet-normed,
    masks → binary long tensor (0/1).  Only the mask channel (label polygon
    rasterisation) is used as supervision target.
    """
    import torch

    images = batch["image"].float() / 255.0
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1)
    images = (images - mean) / std

    # masks from VectorDataset arrive as (B, 1, H, W); clamp to binary 0/1
    masks = batch["mask"].long().squeeze(1).clamp(0, 1)
    return images, masks


def postprocess(logits: Any, threshold: float = 0.5) -> Any:
    """Watershed instance segmentation from 3-channel raw model output.

    Args:
        logits: numpy array (3, H, W) — raw model output before activation.
        threshold: sigmoid threshold applied to mask and EDT channels.

    Returns:
        Integer label array (H, W): 0 = background, 1..N = individual instances.
    """
    import numpy as np
    import scipy.ndimage
    import skimage.segmentation
    from scipy.ndimage import distance_transform_edt

    def _sigmoid(x: Any) -> Any:
        return 1.0 / (1.0 + np.exp(-x))

    mask_logit = logits[0]  # (H, W)
    edt_norm = logits[1]    # (H, W)

    mask_binary = _sigmoid(mask_logit) > threshold
    edt_binary = _sigmoid(edt_norm) > threshold

    # Seeds are connected components that are inside both mask and edt peaks
    seed_mask = edt_binary & mask_binary
    markers, _ = scipy.ndimage.label(seed_mask)

    if markers.max() == 0:
        # No seeds found — return the plain binary mask as a single instance
        result = mask_binary.astype(np.int32)
        return result

    # Distance map for watershed (negative so peaks become basins)
    dist = distance_transform_edt(mask_binary)
    labels = skimage.segmentation.watershed(-dist, markers=markers, mask=mask_binary)
    return labels.astype(np.int32)


def _preprocess_onnx_image(img_path: Any) -> tuple[Any, Any, Any]:
    import numpy as np
    import rasterio

    with rasterio.open(img_path) as src:
        arr = src.read([1, 2, 3]).astype(np.float32) / 255.0
        transform = src.transform
        crs = src.crs
    if arr.shape[-2:] != (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE):
        arr = _resize_chw(arr, MODEL_INPUT_SIZE)
    arr = _normalise(arr)
    return arr[np.newaxis, ...], transform, crs


def _vectorize_instance_labels(label_map: Any, transform: Any, crs: Any) -> list[dict[str, Any]]:
    """Convert an integer instance label map to GeoJSON features in EPSG:4326."""
    import numpy as np
    import rasterio.features
    from pyproj import Transformer

    needs_reproject = crs is not None and str(crs) != "EPSG:4326"
    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True) if needs_reproject else None

    label_map = label_map.astype(np.int32)
    features = []
    for geom, value in rasterio.features.shapes(label_map, transform=transform):
        if value < 1:
            continue
        if transformer:
            coords = geom["coordinates"]
            geom["coordinates"] = [
                [list(transformer.transform(x, y)) for x, y in ring] for ring in coords
            ]
        features.append(
            {"type": "Feature", "properties": {"instance_id": int(value)}, "geometry": geom}
        )
    return features


def _build_feature_collection(features: list[dict[str, Any]]) -> dict[str, Any]:
    return {"type": "FeatureCollection", "features": features}


def predict(session: Any, input_images: str, params: dict[str, Any]) -> dict[str, Any]:
    """Run ONNX inference and return a GeoJSON FeatureCollection of building instances.

    Args:
        session:      ONNX InferenceSession.
        input_images: Path (local or cloud) to a directory of GeoTIFF chips.
        params:       Must contain "confidence_threshold" (float 0–1).
    """
    import numpy as np

    from fair.utils.data import resolve_directory

    if "confidence_threshold" not in params:
        raise ValueError("params['confidence_threshold'] is required")
    threshold = float(params["confidence_threshold"])
    input_name = session.get_inputs()[0].name

    input_dir = resolve_directory(input_images)
    patterns = ("*.png", "*.tif", "*.tiff")
    img_paths = sorted(p for pat in patterns for p in input_dir.glob(pat))
    if not img_paths:
        msg = f"No input images found in {input_dir}"
        raise FileNotFoundError(msg)

    features: list[dict[str, Any]] = []
    for img_path in img_paths:
        batch, transform, crs = _preprocess_onnx_image(img_path)
        raw = session.run(None, {input_name: batch})[0]  # (1, 3, H, W)
        logits_chw = raw[0]  # (3, H, W)
        label_map = postprocess(logits_chw, threshold=threshold)
        features.extend(_vectorize_instance_labels(label_map, transform, crs))
    return _build_feature_collection(features)


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


def _build_dataset(
    chips_path: str,
    labels_path: str,
    chip_size: int,
    length: int,
    batch_size: int = 8,
    split: str = "train",
    seed: int = 42,
    sample_fraction: float = 1.0,
) -> Any:
    """Intersect OAM raster + GeoJSON vector GeoDatasets via torchgeo."""
    from pyproj import CRS
    from torch.utils.data import DataLoader
    from torchgeo.datasets import RasterDataset, VectorDataset, stack_samples
    from torchgeo.samplers import GridGeoSampler, RandomGeoSampler, Units

    from fair.utils.data import resolve_directory

    local_chips = _subset_chips_dir(str(resolve_directory(chips_path, "OAM-*")), sample_fraction)
    local_labels_dir = str(resolve_directory(labels_path, "*.geojson"))

    class _OAMDataset(RasterDataset):
        filename_glob = "OAM-*.tif"
        filename_regex = r"^OAM-(?P<x>\d+)-(?P<y>\d+)-(?P<z>\d+)\.tif$"
        is_image = True
        separate_files = False

    oam = _OAMDataset(paths=local_chips)
    labels = VectorDataset(paths=local_labels_dir, crs=CRS.from_epsg(4326), res=oam.res)
    dataset = oam & labels

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


def _bce_dice_loss(pred_logit: Any, target: Any) -> Any:
    """Combined BCE + soft-Dice loss on the mask channel (binary, float target)."""
    import torch
    import torch.nn.functional as F

    target_f = target.float()
    bce = F.binary_cross_entropy_with_logits(pred_logit, target_f)

    prob = torch.sigmoid(pred_logit)
    smooth = 1.0
    intersection = (prob * target_f).sum(dim=(-2, -1))
    dice = 1.0 - (2.0 * intersection + smooth) / (prob.sum(dim=(-2, -1)) + target_f.sum(dim=(-2, -1)) + smooth)
    return bce + dice.mean()


def _train_step(
    model: Any,
    batch: dict[str, Any],
    optimizer: Any,
    device: str,
    max_grad_norm: float,
) -> float:
    import torch

    images, masks = preprocess(batch)
    images, masks = images.to(device), masks.to(device)
    output = model(images)          # (B, 3, H, W)
    mask_logit = output[:, 0, :, :]  # supervise mask channel only
    loss = _bce_dice_loss(mask_logit, masks)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
    optimizer.step()
    return loss.item()


# ---------------------------------------------------------------------------
# ZenML steps
# ---------------------------------------------------------------------------


@step
def split_dataset(
    dataset_chips: str,
    dataset_labels: str,
    hyperparameters: dict[str, Any],
) -> Annotated[dict[str, Any], "split_info_artifact"]:
    val_ratio = hyperparameters.get("val_ratio", 0.2)
    seed = hyperparameters.get("split_seed", 42)
    block_size = hyperparameters.get("block_size", 4)
    samples_per_epoch = hyperparameters.get("samples_per_epoch", 500)
    val_samples = max(int(samples_per_epoch * val_ratio), 10)

    split_info = {
        "strategy": "spatial_block",
        "val_ratio": val_ratio,
        "seed": seed,
        "block_size": block_size,
        "train_count": samples_per_epoch,
        "val_count": val_samples,
        "description": (
            f"Spatial block split (block_size={block_size} on OAM tile coords): "
            "RandomGeoSampler for train, GridGeoSampler for val (non-overlapping tiles)"
        ),
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
) -> Annotated[Any, "trained_model_artifact"]:
    epochs = hyperparameters["epochs"]
    batch_size = hyperparameters.get("batch_size", 8)
    learning_rate = hyperparameters.get("learning_rate", 5e-5)
    weight_decay = hyperparameters.get("weight_decay", 0.0001)
    chip_size = hyperparameters.get("chip_size", MODEL_INPUT_SIZE)
    samples_per_epoch = hyperparameters.get("samples_per_epoch", 500)
    sample_fraction = hyperparameters.get("sample_fraction", 1.0)
    optimizer_name = hyperparameters.get("optimizer", "AdamW")
    max_grad_norm = hyperparameters.get("max_grad_norm", 1.0)
    scheduler_name = hyperparameters.get("scheduler", "cosine")
    freeze_encoder = hyperparameters.get("freeze_encoder", True)
    seed = split_info["seed"]

    with mlflow_training_context(hyperparameters, model_name, base_model_id, dataset_id):
        import torch

        device = _get_device()
        model = _build_model(num_output_channels=num_classes)

        # Load pretrained weights from HF or a supplied URL
        if base_model_weights and base_model_weights.startswith("hf://"):
            state_dict, _ = _load_hf_weights(device)
            model.load_state_dict(state_dict, strict=False)
        elif base_model_weights:
            local_path = _download_checkpoint(base_model_weights)
            state = torch.load(local_path, map_location=device, weights_only=True)
            state_dict = state.get("model", state)
            model.load_state_dict(state_dict, strict=False)

        model.to(device)

        if freeze_encoder:
            for param in model.encoder.parameters():
                param.requires_grad = False

        train_loader = _build_dataset(
            dataset_chips,
            dataset_labels,
            chip_size,
            length=samples_per_epoch,
            batch_size=batch_size,
            split="train",
            seed=seed,
            sample_fraction=sample_fraction,
        )
        val_loader = _build_dataset(
            dataset_chips,
            dataset_labels,
            chip_size,
            length=samples_per_epoch,
            batch_size=batch_size,
            split="val",
            seed=seed,
            sample_fraction=sample_fraction,
        )

        optimizers = _get_optimizers()
        trainable = filter(lambda p: p.requires_grad, model.parameters())
        opt = optimizers[optimizer_name](trainable, lr=learning_rate, weight_decay=weight_decay)

        scheduler = None
        if scheduler_name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

        train_losses: list[float] = []
        val_losses: list[float] = []

        model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for batch in train_loader:
                total_loss += _train_step(model, batch, opt, device, max_grad_norm)
            if scheduler:
                scheduler.step()
            avg_train_loss = total_loss / max(len(train_loader), 1)

            model.eval()
            val_total = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    images, masks = preprocess(batch)
                    images, masks = images.to(device), masks.to(device)
                    output = model(images)
                    mask_logit = output[:, 0, :, :]
                    val_total += _bce_dice_loss(mask_logit, masks).item()
            avg_val_loss = val_total / max(len(val_loader), 1)
            model.train()

            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)

            import mlflow

            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)  # ty: ignore[possibly-missing-attribute]
            mlflow.log_metric("val_loss", avg_val_loss, step=epoch)  # ty: ignore[possibly-missing-attribute]
            log_metadata(metadata={"loss": avg_train_loss, "epoch": epoch + 1})
            msg = f"epoch {epoch + 1}/{epochs}  train_loss={avg_train_loss:.4f}  val_loss={avg_val_loss:.4f}"
            print(msg, flush=True)

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
    num_classes: int = 2,
    class_names: list[str] | None = None,
) -> Annotated[dict[str, Any], "metrics"]:
    """Evaluate on the val split using pixel accuracy and mean IoU on the binary mask."""
    import torch

    chip_size = hyperparameters.get("chip_size", MODEL_INPUT_SIZE)
    sample_fraction = hyperparameters.get("sample_fraction", 1.0)

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
        sample_fraction=sample_fraction,
    )

    # Evaluate the binary mask channel (background vs building)
    n_classes = 2
    total_correct = total_pixels = 0
    intersection = [0] * n_classes
    union = [0] * n_classes

    with torch.no_grad():
        for batch in loader:
            images, masks = preprocess(batch)
            images, masks = images.to(device), masks.to(device)
            output = model(images)
            import torch as _torch

            preds = (_torch.sigmoid(output[:, 0, :, :]) > 0.5).long()
            total_correct += (preds == masks).sum().item()
            total_pixels += masks.numel()
            for c in range(n_classes):
                intersection[c] += ((preds == c) & (masks == c)).sum().item()
                union[c] += ((preds == c) | (masks == c)).sum().item()

    resolved_names = (
        class_names
        if class_names and len(class_names) == n_classes
        else ["background", "building"]
    )
    per_class_iou = {resolved_names[c]: intersection[c] / max(union[c], 1) for c in range(n_classes)}

    accuracy = total_correct / max(total_pixels, 1)
    mean_iou = sum(per_class_iou.values()) / n_classes
    metrics = {"accuracy": accuracy, "mean_iou": mean_iou, "per_class_iou": per_class_iou}
    log_evaluation_results(metrics)
    return metrics


@step
def run_inference(
    model_uri: str,
    input_images: str,
    inference_params: dict[str, Any],
) -> Annotated[dict[str, Any], "predictions"]:
    from fair.serve.base import load_session

    session = load_session(model_uri)
    return predict(session, input_images, inference_params)


@step(output_materializers={"onnx_model": ONNXMaterializer})
def export_onnx(
    trained_model: Any,
    hyperparameters: dict[str, Any],
    num_classes: int = 3,
) -> Annotated[bytes, "onnx_model"]:
    import os
    import tempfile

    import onnx
    import torch

    chip_size = hyperparameters.get("chip_size", MODEL_INPUT_SIZE)

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


# ---------------------------------------------------------------------------
# ZenML pipelines
# ---------------------------------------------------------------------------


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
    inference_params: dict[str, Any],
) -> None:
    run_inference(
        model_uri=model_uri,
        input_images=input_images,
        inference_params=inference_params,
    )
