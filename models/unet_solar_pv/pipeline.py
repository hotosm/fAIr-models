"""unet-solar-pv: a U-Net that segments solar PV panels in RGB imagery, finetunable in fAIr."""

import io
import math
import os
import random
import re
import tempfile
from pathlib import Path
from typing import Annotated, Any

from zenml import log_metadata, pipeline, step

from fair.stac.inputs import IMAGE_INPUT
from fair.utils.data import resolve_directory, resolve_path
from fair.zenml.instrumentation import log_evaluation_results, mlflow_training_context
from fair.zenml.materializers import CheckpointBytesMaterializer, ONNXMaterializer
from fair.zenml.metrics import log_loss_history

CHIP_SIZE = 256
ENCODER = "resnet34"
# fAIr names each chip after its web map tile: OAM-<x>-<y>-<zoom>.tif.
OAM_TILE = re.compile(r"OAM-(\d+)-(\d+)-\d+")


def _chip_paths(chips_href: str) -> list[Path]:
    chips = resolve_directory(chips_href)
    return [chips] if chips.is_file() else sorted(chips.rglob("*.tif"))


def preprocess(image_path: Any) -> Any:
    """Read a 256 x 256 RGB chip as a (1, 3, 256, 256) float32 tensor scaled to 0 to 1.

    The ImageNet mean and standard deviation are applied inside the ONNX graph, so only the
    division by 255 happens here.
    """
    import numpy as np
    import rasterio

    with rasterio.open(image_path) as src:
        rgb = src.read([1, 2, 3]).astype(np.float32) / 255.0
    if rgb.shape[1:] != (CHIP_SIZE, CHIP_SIZE):
        msg = f"{image_path}: chip is {rgb.shape[2]}x{rgb.shape[1]}, the model takes {CHIP_SIZE}x{CHIP_SIZE}"
        raise ValueError(msg)
    return rgb[np.newaxis]


def postprocess(logits: Any, threshold: float) -> Any:
    """Turn a (1, H, W) solar panel logit map into a uint8 panel mask."""
    import numpy as np

    logits = np.asarray(logits, dtype=np.float64)
    # One output channel: the sigmoid of the logit is the panel probability.
    panel_probability = 1.0 / (1.0 + np.exp(-logits[0]))
    return (panel_probability >= threshold).astype(np.uint8)


def predict(session: Any, input_images: dict[str, str], params: dict[str, Any]) -> dict[str, Any]:
    import rasterio
    from pyproj import Transformer
    from rasterio.features import shapes
    from shapely.geometry import mapping, shape
    from shapely.ops import transform as shapely_transform

    threshold = float(params.get("confidence_threshold", 0.5))
    input_name = session.get_inputs()[0].name
    chip_paths = _chip_paths(input_images[IMAGE_INPUT])
    if not chip_paths:
        msg = f"No georeferenced (.tif) chips found in {input_images[IMAGE_INPUT]}"
        raise FileNotFoundError(msg)

    features: list[dict[str, Any]] = []
    for chip_path in chip_paths:
        with rasterio.open(chip_path) as src:
            transform = src.transform
            to_wgs84 = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
        logits = session.run(None, {input_name: preprocess(chip_path)})[0][0]
        mask = postprocess(logits, threshold)
        for geom, _ in shapes(mask, mask=mask.astype(bool), transform=transform):
            polygon = shapely_transform(lambda x, y, _z=None, t=to_wgs84: t.transform(x, y), shape(geom))
            features.append(
                {"type": "Feature", "properties": {"class": 1, "label": "solar_panel"}, "geometry": mapping(polygon)}
            )
    return {"type": "FeatureCollection", "features": features}


def _device() -> str:
    if os.environ.get("FAIR_FORCE_CPU", "").lower() in {"1", "true", "yes", "on"}:
        return "cpu"
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _build_model() -> Any:
    """The network inside the ONNX file: ImageNet normalisation, then a ResNet-34 U-Net with one logit."""
    import segmentation_models_pytorch as smp
    import torch

    class SolarNet(torch.nn.Module):
        mean: torch.Tensor
        std: torch.Tensor

        def __init__(self) -> None:
            super().__init__()
            self.unet = smp.Unet(ENCODER, encoder_weights=None, in_channels=3, classes=1)
            self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.unet((x - self.mean) / self.std)

    return SolarNet()


def _load_model(weights: bytes | str) -> Any:
    """The U-Net with weights from a saved state_dict: checkpoint bytes, a local path or a URL."""
    import torch

    if isinstance(weights, str):
        if "://" in weights:
            from upath import UPath

            weights = UPath(weights).read_bytes()
        else:
            weights = Path(weights).read_bytes()
    model = _build_model()
    model.load_state_dict(torch.load(io.BytesIO(weights), map_location="cpu", weights_only=True))
    return model


def _label_geometries(dataset_labels: str) -> list[Any]:
    """Solar panel polygons from the dataset GeoJSON, in EPSG:4326."""
    import json

    from shapely.geometry import shape

    if dataset_labels.endswith((".geojson", ".json")):
        label_file = resolve_path(dataset_labels)
    else:
        files = sorted(resolve_directory(dataset_labels, "*.geojson").rglob("*.geojson"))
        if not files:
            msg = f"No .geojson labels found in {dataset_labels}"
            raise FileNotFoundError(msg)
        label_file = files[0]
    features = json.loads(label_file.read_text())["features"]
    return [shape(f["geometry"]) for f in features if f.get("geometry")]


def _load_chips(chips_href: str, names: list[str], geometries: list[Any]) -> tuple[Any, Any]:
    """Chips as (N, 3, 256, 256) uint8 RGB and their panel masks as (N, 1, 256, 256) float32."""
    import numpy as np
    import rasterio
    from pyproj import Transformer
    from rasterio.features import rasterize
    from rasterio.warp import transform_bounds
    from shapely.geometry import box
    from shapely.ops import transform as shapely_transform

    paths = {p.name: p for p in _chip_paths(chips_href)}
    images = np.zeros((len(names), 3, CHIP_SIZE, CHIP_SIZE), np.uint8)
    masks = np.zeros((len(names), 1, CHIP_SIZE, CHIP_SIZE), np.float32)
    for i, name in enumerate(names):
        images[i] = np.round(preprocess(paths[name])[0] * 255)
        with rasterio.open(paths[name]) as src:
            footprint = box(*transform_bounds(src.crs, "EPSG:4326", *src.bounds))
            to_chip = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
            panels = [
                shapely_transform(lambda x, y, _z=None, t=to_chip: t.transform(x, y), g)
                for g in geometries
                if g.intersects(footprint)
            ]
            if panels:
                masks[i, 0] = rasterize(
                    [(g, 1) for g in panels], out_shape=(CHIP_SIZE, CHIP_SIZE), transform=src.transform
                )
    return images, masks


def _augment(images: Any, masks: Any) -> tuple[Any, Any]:
    """Random quarter turns and flips (nadir imagery has no up), with RGB scaled to 0 to 1."""
    import numpy as np

    x, y = [], []
    for image, mask in zip(images, masks, strict=True):
        turns, flip = np.random.randint(4), np.random.rand() < 0.5
        image, mask = np.rot90(image, turns, axes=(1, 2)), np.rot90(mask, turns, axes=(1, 2))
        if flip:
            image, mask = image[:, :, ::-1], mask[:, :, ::-1]
        x.append(image)
        y.append(mask)
    return np.stack(x).astype(np.float32) / 255.0, np.stack(y).astype(np.float32)


def _panel_probabilities(model: Any, images: Any, batch_size: int = 8) -> Any:
    import numpy as np
    import torch

    device = _device()
    model = model.to(device).eval()
    batches = []
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            x = torch.from_numpy(images[start : start + batch_size].astype(np.float32) / 255.0).to(device)
            batches.append(torch.sigmoid(model(x)).cpu().numpy())
    return np.concatenate(batches) if batches else np.zeros((0, 1, CHIP_SIZE, CHIP_SIZE), np.float32)


def _spatial_split(names: list[str], val_ratio: float, seed: int, block_size: int) -> tuple[list[str], list[str]]:
    """Assign whole blocks of neighbouring chips to train or validation, so validation areas are unseen."""
    blocks: dict[Any, list[str]] = {}
    for name in names:
        tile = OAM_TILE.search(name)
        key = (int(tile[1]) // block_size, int(tile[2]) // block_size) if tile else name
        blocks.setdefault(key, []).append(name)
    if len(blocks) < 2:
        # Every chip sits in one block: hold out single chips instead.
        blocks = {name: [name] for name in names}
    keys = sorted(blocks, key=str)
    random.Random(seed).shuffle(keys)
    target = max(1, round(val_ratio * len(names)))
    val: list[str] = []
    # The last block always trains, so the training set is never empty.
    for key in keys[:-1]:
        if len(val) >= target:
            break
        val.extend(blocks[key])
    held_out = set(val)
    return sorted(n for n in names if n not in held_out), sorted(val)


@step
def split_dataset(
    dataset_chips: dict[str, str],
    dataset_labels: str,
    hyperparameters: dict[str, Any],
) -> Annotated[dict[str, Any], "split_info"]:
    """Spatial block split on OAM tile coordinates: neighbouring chips share roofs and arrays."""
    names = [p.name for p in _chip_paths(dataset_chips[IMAGE_INPUT])]
    if not names:
        msg = f"No .tif chips found in {dataset_chips[IMAGE_INPUT]}"
        raise FileNotFoundError(msg)
    val_ratio = float(hyperparameters.get("val_ratio", 0.2))
    seed = int(hyperparameters.get("split_seed", 42))
    block_size = int(hyperparameters.get("block_size", 4))
    train_names, val_names = _spatial_split(names, val_ratio, seed, block_size)

    info = {
        "strategy": "spatial",
        "val_ratio": val_ratio,
        "seed": seed,
        "block_size": block_size,
        "train_count": len(train_names),
        "val_count": len(val_names),
        "description": (
            f"Chips named OAM-<x>-<y>-<z>.tif are grouped into {block_size} x {block_size} tile blocks and whole "
            "blocks go to train or validation; other chips, or a dataset inside one block, are held out one by one."
        ),
        "train_chip_names": train_names,
        "val_chip_names": val_names,
    }
    log_metadata(metadata={"fair/split": {k: v for k, v in info.items() if not k.endswith("_chip_names")}})
    return info


@step(output_materializers={"trained_model_artifact": CheckpointBytesMaterializer})
def train_model(
    dataset_chips: dict[str, str],
    dataset_labels: str,
    base_model_weights: str,
    hyperparameters: dict[str, Any],
    split_info: dict[str, Any],
    num_classes: int = 2,
    model_name: str | None = None,
    base_model_id: str | None = None,
    dataset_id: str | None = None,
) -> Annotated[bytes, "trained_model_artifact"]:
    """Finetune the base U-Net on the training chips with binary cross-entropy plus Dice loss.

    The network keeps its single panel logit whatever `num_classes` (background and solar_panel) says.
    """
    import mlflow
    import numpy as np
    import segmentation_models_pytorch as smp
    import torch

    epochs = int(hyperparameters.get("epochs", 15))
    batch_size = int(hyperparameters.get("batch_size", 8))
    seed = int(split_info["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    names = split_info["train_chip_names"]
    if not names:
        msg = "The split left no training chips"
        raise ValueError(msg)
    images, masks = _load_chips(dataset_chips[IMAGE_INPUT], names, _label_geometries(dataset_labels))

    device = _device()
    model = _load_model(base_model_weights).to(device)
    if hyperparameters.get("freeze_encoder", False):
        for parameter in model.unet.encoder.parameters():
            parameter.requires_grad = False
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimiser = torch.optim.AdamW(trainable, lr=float(hyperparameters.get("learning_rate", 1e-4)), weight_decay=1e-4)
    scheduler = None
    if hyperparameters.get("scheduler", "cosine") == "cosine":
        total_steps = epochs * math.ceil(len(images) / batch_size)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=total_steps)
    bce = torch.nn.BCEWithLogitsLoss()
    dice = smp.losses.DiceLoss("binary", from_logits=True)
    max_grad_norm = float(hyperparameters.get("max_grad_norm", 1.0))

    losses: list[float] = []
    with mlflow_training_context(hyperparameters, model_name, base_model_id, dataset_id):
        for epoch in range(epochs):
            model.train()
            order = np.random.permutation(len(images))
            total = 0.0
            for start in range(0, len(order), batch_size):
                batch = order[start : start + batch_size]
                x, y = (torch.from_numpy(a).to(device) for a in _augment(images[batch], masks[batch]))
                logits = model(x)
                loss = bce(logits, y) + dice(logits, y)
                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
                optimiser.step()
                if scheduler is not None:
                    scheduler.step()
                total += loss.item() * len(batch)
            losses.append(total / len(images))
            mlflow.log_metric("train_loss", losses[-1], step=epoch)  # ty: ignore[possibly-missing-attribute]
    log_loss_history(losses, [])

    buffer = io.BytesIO()
    torch.save(model.cpu().state_dict(), buffer)
    return buffer.getvalue()


@step
def evaluate_model(
    trained_model: bytes,
    dataset_chips: dict[str, str],
    dataset_labels: str,
    hyperparameters: dict[str, Any],
    split_info: dict[str, Any],
    num_classes: int = 2,
    class_names: list[str] | None = None,
) -> Annotated[dict[str, Any], "metrics"]:
    """Solar panel pixel IoU, precision, recall and F1 on the validation chips."""
    threshold = float(hyperparameters.get("confidence_threshold", 0.5))
    images, masks = _load_chips(
        dataset_chips[IMAGE_INPUT], split_info["val_chip_names"], _label_geometries(dataset_labels)
    )
    panels = _panel_probabilities(_load_model(trained_model), images) >= threshold
    truth = masks.astype(bool)
    tp, fp, fn = int((panels & truth).sum()), int((panels & ~truth).sum()), int((~panels & truth).sum())

    metrics = {
        "iou_solar_panel": tp / max(tp + fp + fn, 1),
        "precision_solar_panel": tp / max(tp + fp, 1),
        "recall_solar_panel": tp / max(tp + fn, 1),
        "f1_solar_panel": 2 * tp / max(2 * tp + fp + fn, 1),
    }
    log_evaluation_results(metrics)
    return metrics


@step(output_materializers={"onnx_model": ONNXMaterializer})
def export_onnx(trained_model: bytes) -> Annotated[bytes, "onnx_model"]:
    """Export with the serving contract: input `image` (N, 3, 256, 256), output `logits` (N, 1, 256, 256)."""
    import onnx
    import torch

    model = _load_model(trained_model).eval()
    example = torch.rand(2, 3, CHIP_SIZE, CHIP_SIZE)
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "model.onnx")
        torch.onnx.export(
            model,
            (example,),
            path,
            input_names=["image"],
            output_names=["logits"],
            opset_version=18,
            dynamo=True,
            verbose=False,
            dynamic_shapes={"x": {0: torch.export.Dim("batch", min=1, max=64)}},
        )
        # The exporter can write weights to a sidecar file; inline them so the bytes stand alone.
        onnx.save(onnx.load(path), path, save_as_external_data=False)
        onnx.checker.check_model(path)
        return Path(path).read_bytes()


@step
def run_inference(
    model_uri: str,
    input_images: dict[str, str],
    inference_params: dict[str, Any],
) -> Annotated[dict[str, Any], "predictions"]:
    from fair.serve.base import load_session

    return predict(load_session(model_uri), input_images, inference_params)


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
    export_onnx(trained_model=trained_model)


@pipeline
def inference_pipeline(
    model_uri: str,
    input_images: dict[str, str],
    inference_params: dict[str, Any],
) -> None:
    run_inference(model_uri=model_uri, input_images=input_images, inference_params=inference_params)
