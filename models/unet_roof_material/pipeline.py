"""unet-roof-material: a U-Net that maps roofs as metal sheet, thatch or other roof in RGB imagery, finetunable
in fAIr."""

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
CLASS_NAMES = ["background", "metal_sheet", "thatch", "other_roof"]
ROOF_CLASSES = [1, 2, 3]
# Label value for roofs whose material is unknown: the loss and the scores skip their pixels.
IGNORE = 255
# OpenStreetMap roof:material values a label can carry instead of an integer `label`.
ROOF_MATERIALS = {"metal_sheet": 1, "thatch": 2, "asbestos": 3, "concrete": 3}
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
    """Turn (4, H, W) class logits, or a batch of them, into a uint8 map of class values.

    Each pixel takes its most likely class; a roof pixel whose class probability is below the
    threshold becomes background.
    """
    import numpy as np

    logits = np.asarray(logits, dtype=np.float64)
    probabilities = np.exp(logits - logits.max(axis=-3, keepdims=True))
    probabilities /= probabilities.sum(axis=-3, keepdims=True)
    classes = probabilities.argmax(axis=-3)
    confidence = probabilities.max(axis=-3)
    classes[(classes > 0) & (confidence < threshold)] = 0
    return classes.astype(np.uint8)


def predict(session: Any, input_images: dict[str, str], params: dict[str, Any]) -> dict[str, Any]:
    import rasterio
    from pyproj import Transformer
    from rasterio.features import shapes
    from shapely.geometry import mapping, shape
    from shapely.ops import transform as shapely_transform

    threshold = float(params.get("confidence_threshold", 0.0))
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
        classes = postprocess(logits, threshold)
        for geom, value in shapes(classes, mask=classes > 0, transform=transform):
            polygon = shapely_transform(lambda x, y, _z=None, t=to_wgs84: t.transform(x, y), shape(geom))
            features.append(
                {
                    "type": "Feature",
                    "properties": {"class": int(value), "label": CLASS_NAMES[int(value)]},
                    "geometry": mapping(polygon),
                }
            )
    return {"type": "FeatureCollection", "features": features}


def _device() -> str:
    if os.environ.get("FAIR_FORCE_CPU", "").lower() in {"1", "true", "yes", "on"}:
        return "cpu"
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _build_model() -> Any:
    """The network inside the ONNX file: ImageNet normalisation, then a ResNet-34 U-Net with one logit per class."""
    import segmentation_models_pytorch as smp
    import torch

    class RoofNet(torch.nn.Module):
        mean: torch.Tensor
        std: torch.Tensor

        def __init__(self) -> None:
            super().__init__()
            self.unet = smp.Unet(ENCODER, encoder_weights=None, in_channels=3, classes=len(CLASS_NAMES))
            self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.unet((x - self.mean) / self.std)

    return RoofNet()


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


def _roof_class(properties: dict[str, Any]) -> int:
    """A label's class: its roof:material tag when the tag is in ROOF_MATERIALS, else its integer `label`
    (1 metal sheet, 2 thatch, 3 other roof), else IGNORE.

    Labels fAIr downloads from OpenStreetMap keep their tags under `tags`.
    """
    tags = properties.get("tags") or {}
    material = properties.get("roof:material") or properties.get("roof_material") or tags.get("roof:material")
    if material in ROOF_MATERIALS:
        return ROOF_MATERIALS[material]
    try:
        label = int(properties.get("label"))  # ty: ignore[invalid-argument-type]
    except (TypeError, ValueError):
        return IGNORE
    return label if label in ROOF_CLASSES else IGNORE


def _roof_labels(dataset_labels: str) -> list[tuple[Any, int]]:
    """(polygon in EPSG:4326, class) for every labelled roof; a roof with no known material gets IGNORE."""
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
    return [(shape(f["geometry"]), _roof_class(f.get("properties") or {})) for f in features if f.get("geometry")]


def _chip_footprints(paths: list[Path]) -> list[Any]:
    import rasterio
    from rasterio.warp import transform_bounds
    from shapely.geometry import box

    footprints = []
    for path in paths:
        with rasterio.open(path) as src:
            footprints.append(box(*transform_bounds(src.crs, "EPSG:4326", *src.bounds)))
    return footprints


def _load_chips(chips_href: str, names: list[str], labels: list[tuple[Any, int]]) -> tuple[Any, Any]:
    """Chips as (N, 3, 256, 256) uint8 RGB and their class masks as (N, 256, 256) uint8.

    Roofs of unknown material are burnt first as IGNORE, so a labelled roof drawn over one wins.
    """
    import numpy as np
    import rasterio
    from pyproj import Transformer
    from rasterio.features import rasterize
    from shapely.ops import transform as shapely_transform
    from shapely.strtree import STRtree

    paths = {p.name: p for p in _chip_paths(chips_href)}
    tree = STRtree([geometry for geometry, _ in labels])
    images = np.zeros((len(names), 3, CHIP_SIZE, CHIP_SIZE), np.uint8)
    masks = np.zeros((len(names), CHIP_SIZE, CHIP_SIZE), np.uint8)
    for i, (name, footprint) in enumerate(zip(names, _chip_footprints([paths[n] for n in names]), strict=True)):
        images[i] = np.round(preprocess(paths[name])[0] * 255)
        hits = [labels[j] for j in tree.query(footprint)]
        if not hits:
            continue
        with rasterio.open(paths[name]) as src:
            to_chip = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
            transform = src.transform
        for unknown in (True, False):
            roofs = [
                (shapely_transform(lambda x, y, _z=None, t=to_chip: t.transform(x, y), geometry), value)
                for geometry, value in hits
                if (value == IGNORE) == unknown
            ]
            if roofs:
                burnt = rasterize(roofs, out_shape=(CHIP_SIZE, CHIP_SIZE), transform=transform, fill=0, dtype="uint8")
                masks[i] = np.where(burnt > 0, burnt, masks[i])
    return images, masks


def _roofs_per_chip(chips_href: str, names: list[str], labels: list[tuple[Any, int]]) -> Any:
    """How many labelled roofs of each roof class touch each chip, as an (N, 3) array."""
    import numpy as np
    from shapely.strtree import STRtree

    paths = {p.name: p for p in _chip_paths(chips_href)}
    tree = STRtree([geometry for geometry, _ in labels])
    counts = np.zeros((len(names), len(ROOF_CLASSES)), np.int64)
    for i, footprint in enumerate(_chip_footprints([paths[n] for n in names])):
        for j in tree.query(footprint):
            if labels[j][1] in ROOF_CLASSES:
                counts[i, labels[j][1] - 1] += 1
    return counts


def _augment(images: Any, masks: Any) -> tuple[Any, Any]:
    """Random quarter turns and flips (nadir imagery has no up) and small brightness and contrast changes.

    RGB comes back scaled to 0 to 1, and each mask turns with its chip.
    """
    import numpy as np

    x, y = [], []
    for image, mask in zip(images, masks, strict=True):
        turns = np.random.randint(4)
        image, mask = np.rot90(image, turns, axes=(1, 2)), np.rot90(mask, turns)
        if np.random.rand() < 0.5:
            image, mask = image[:, :, ::-1], mask[:, ::-1]
        image = image.astype(np.float32) * np.random.uniform(0.8, 1.2) + np.random.uniform(-20, 20)
        x.append(np.clip(image, 0, 255) / 255.0)
        y.append(mask)
    # Turned arrays keep their turned memory layout through np.stack; the losses need plain C order.
    return np.ascontiguousarray(np.stack(x), np.float32), np.ascontiguousarray(np.stack(y), np.int64)


def _class_weights(masks: Any, max_weight: float) -> list[float]:
    """Cross-entropy weight per class: the square root of background share over class share, from 1 to max_weight."""
    import numpy as np

    pixels = np.bincount(masks[masks != IGNORE].ravel(), minlength=len(CLASS_NAMES)).astype(np.float64)
    weights = np.ones(len(CLASS_NAMES))
    present = pixels > 0
    weights[present] = np.sqrt(pixels[0] / pixels[present]).clip(1.0, max_weight) if pixels[0] else 1.0
    return [float(w) for w in weights]


def _predict_classes(model: Any, images: Any, threshold: float, batch_size: int = 8) -> Any:
    """Class value maps (N, 256, 256) for uint8 chips (N, 3, 256, 256)."""
    import numpy as np
    import torch

    device = _device()
    model = model.to(device).eval()
    classes = np.zeros((len(images), CHIP_SIZE, CHIP_SIZE), np.uint8)
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            x = torch.from_numpy(images[start : start + batch_size].astype(np.float32) / 255.0).to(device)
            classes[start : start + batch_size] = postprocess(model(x).cpu().numpy(), threshold)
    return classes


def _spatial_split(
    names: list[str], roofs: Any, val_ratio: float, seed: int, block_size: int, tries: int = 200
) -> tuple[list[str], list[str]]:
    """Assign whole blocks of neighbouring chips to train or validation, so validation areas are unseen.

    Of `tries` seeded shuffles of the blocks, keep the one whose validation set holds the share of each
    roof class closest to `val_ratio`, so a rare material still gets scored.
    """
    import numpy as np

    blocks: dict[Any, list[int]] = {}
    for i, name in enumerate(names):
        tile = OAM_TILE.search(name)
        key = (int(tile[1]) // block_size, int(tile[2]) // block_size) if tile else name
        blocks.setdefault(key, []).append(i)
    if len(blocks) < 2:
        # Every chip sits in one block: hold out single chips instead.
        blocks = {name: [i] for i, name in enumerate(names)}
    keys = sorted(blocks, key=str)
    target = max(1, round(val_ratio * len(names)))
    totals = roofs.sum(axis=0)
    rng = random.Random(seed)
    best: list[int] = []
    best_gap = math.inf
    for _ in range(tries):
        rng.shuffle(keys)
        val: list[int] = []
        # The last block always trains, so the training set is never empty.
        for key in keys[:-1]:
            if len(val) >= target:
                break
            val.extend(blocks[key])
        share = roofs[val].sum(axis=0) / np.maximum(totals, 1)
        gap = float(np.abs(share - val_ratio)[totals > 0].max(initial=0.0))
        if gap < best_gap:
            best, best_gap = val, gap
    held_out = {names[i] for i in best}
    return sorted(n for n in names if n not in held_out), sorted(held_out)


@step
def split_dataset(
    dataset_chips: dict[str, str],
    dataset_labels: str,
    hyperparameters: dict[str, Any],
) -> Annotated[dict[str, Any], "split_info"]:
    """Spatial block split on OAM tile coordinates, balanced so each roof class is held out in about val_ratio."""
    names = [p.name for p in _chip_paths(dataset_chips[IMAGE_INPUT])]
    if not names:
        msg = f"No .tif chips found in {dataset_chips[IMAGE_INPUT]}"
        raise FileNotFoundError(msg)
    val_ratio = float(hyperparameters.get("val_ratio", 0.2))
    seed = int(hyperparameters.get("split_seed", 42))
    block_size = int(hyperparameters.get("block_size", 4))
    roofs = _roofs_per_chip(dataset_chips[IMAGE_INPUT], names, _roof_labels(dataset_labels))
    train_names, val_names = _spatial_split(names, roofs, val_ratio, seed, block_size)

    info = {
        "strategy": "spatial",
        "val_ratio": val_ratio,
        "seed": seed,
        "block_size": block_size,
        "train_count": len(train_names),
        "val_count": len(val_names),
        "description": (
            f"Chips named OAM-<x>-<y>-<z>.tif are grouped into {block_size} x {block_size} tile blocks and whole "
            "blocks go to train or validation, choosing among seeded shuffles the one that holds out each roof "
            "class closest to the validation ratio; other chips, or a dataset inside one block, are held out one "
            "by one."
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
    num_classes: int = len(CLASS_NAMES),
    model_name: str | None = None,
    base_model_id: str | None = None,
    dataset_id: str | None = None,
) -> Annotated[bytes, "trained_model_artifact"]:
    """Finetune the base U-Net on the training chips with class-weighted cross-entropy plus Dice on the roof classes.

    Pixels of roofs with unknown material count in neither loss.
    """
    import mlflow
    import numpy as np
    import segmentation_models_pytorch as smp
    import torch

    if num_classes != len(CLASS_NAMES):
        msg = f"unet-roof-material predicts {len(CLASS_NAMES)} classes {CLASS_NAMES}, got num_classes={num_classes}"
        raise ValueError(msg)
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
    images, masks = _load_chips(dataset_chips[IMAGE_INPUT], names, _roof_labels(dataset_labels))
    weights = _class_weights(masks, float(hyperparameters.get("max_class_weight", 5.0)))
    log_metadata(metadata={"fair/class_weights": dict(zip(CLASS_NAMES, weights, strict=True))})

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
    cross_entropy = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, device=device), ignore_index=IGNORE)
    dice = smp.losses.DiceLoss("multiclass", classes=ROOF_CLASSES, from_logits=True, ignore_index=IGNORE)
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
                loss = cross_entropy(logits, y) + dice(logits, y)
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
    num_classes: int = len(CLASS_NAMES),
    class_names: list[str] | None = None,
) -> Annotated[dict[str, Any], "metrics"]:
    """Per roof class pixel IoU, precision and recall on the validation chips, and their mean IoU.

    Pixels of roofs with unknown material are left out.
    """
    import numpy as np

    threshold = float(hyperparameters.get("confidence_threshold", 0.0))
    images, masks = _load_chips(dataset_chips[IMAGE_INPUT], split_info["val_chip_names"], _roof_labels(dataset_labels))
    predicted = _predict_classes(_load_model(trained_model), images, threshold)
    known = masks != IGNORE
    n = len(CLASS_NAMES)
    # Rows: labelled class, columns: predicted class.
    confusion = np.bincount(masks[known].astype(np.int64) * n + predicted[known], minlength=n * n).reshape(n, n)

    metrics: dict[str, Any] = {}
    for value in ROOF_CLASSES:
        tp = int(confusion[value, value])
        fp, fn = int(confusion[:, value].sum()) - tp, int(confusion[value, :].sum()) - tp
        name = CLASS_NAMES[value]
        metrics[f"iou_{name}"] = tp / max(tp + fp + fn, 1)
        metrics[f"precision_{name}"] = tp / max(tp + fp, 1)
        metrics[f"recall_{name}"] = tp / max(tp + fn, 1)
    metrics["mean_iou_roof"] = float(np.mean([metrics[f"iou_{CLASS_NAMES[v]}"] for v in ROOF_CLASSES]))
    log_evaluation_results(metrics)
    return metrics


@step(output_materializers={"onnx_model": ONNXMaterializer})
def export_onnx(trained_model: bytes) -> Annotated[bytes, "onnx_model"]:
    """Export with the serving contract: input `image` (N, 3, 256, 256), output `logits` (N, 4, 256, 256)."""
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
