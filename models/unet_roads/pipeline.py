"""unet-roads: a U-Net road segmenter that returns road surfaces as polygons.

Training burns road labels into a binary mask (polygons as drawn, centrelines
buffered to a road width) and fits a U-Net on it. Inference mosaics the predicted
road probability of every chip in a request, thresholds it and vectorises the
mask into simplified WGS84 polygons.
"""

import io
import json
import os
import re
from pathlib import Path
from typing import Annotated, Any

from zenml import log_metadata, pipeline, step

from fair.stac.inputs import IMAGE_INPUT
from fair.utils.data import resolve_directory, resolve_path
from fair.zenml.instrumentation import log_evaluation_results, mlflow_training_context
from fair.zenml.materializers import CheckpointBytesMaterializer, ONNXMaterializer
from fair.zenml.metrics import log_loss_history

MODEL_NAME = "unet-roads"
CLASS_NAMES = ("background", "road")
MODEL_INPUT_SIZE = 256
DEFAULT_ENCODER = "resnet34"
CHIP_NAME = re.compile(r"OAM-(\d+)-(\d+)-(\d+)")

DEFAULT_INFERENCE_PARAMS: dict[str, Any] = {
    "confidence_threshold": 0.5,
    "min_area_m2": 30.0,
    "simplify_m": 0.5,
}


def _chip_paths(chips_href: str) -> list[Path]:
    chips = resolve_directory(chips_href, "*.tif*")
    return [chips] if chips.is_file() else sorted(chips.rglob("*.tif"))


def _label_geometries(dataset_labels: str) -> Any:
    """Road geometries from the dataset GeoJSON (assumed EPSG:4326), spatially indexed."""
    from shapely import STRtree
    from shapely.geometry import shape

    labels = resolve_directory(dataset_labels)
    label_file = labels if labels.is_file() else sorted(labels.rglob("*.geojson"))[0]
    data = json.loads(label_file.read_text())
    return STRtree([shape(f["geometry"]) for f in data.get("features", []) if f.get("geometry")])


def _metres_per_pixel(src: Any) -> float:
    """Ground width of one chip pixel, measured geodesically along the chip's top edge."""
    from pyproj import Geod, Transformer

    to_wgs84 = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
    (lon0, lon1), (lat0, lat1) = to_wgs84.transform([src.bounds.left, src.bounds.right], [src.bounds.top] * 2)
    _, _, distance = Geod(ellps="WGS84").inv(lon0, lat0, lon1, lat1)
    return distance / src.width


def _model_grid(src: Any) -> Any:
    """Geotransform of the chip resampled to the model's MODEL_INPUT_SIZE square grid."""
    from rasterio.transform import Affine

    return src.transform * Affine.scale(src.width / MODEL_INPUT_SIZE, src.height / MODEL_INPUT_SIZE)


def _road_mask(src: Any, roads: Any, road_width_m: float) -> Any:
    """Burn the indexed road labels near the chip into a model-grid uint8 mask.

    Polygon labels are burned as drawn. Line and point labels carry no width, so they
    are buffered to `road_width_m` first.
    """
    import numpy as np
    from pyproj import Transformer
    from rasterio.features import rasterize
    from rasterio.warp import transform_bounds
    from shapely.geometry import box
    from shapely.ops import transform as shapely_transform

    size = (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)
    # Widen the chip's lon/lat box by the road width (1e-5 degrees is about 1 m) so roads just outside still bleed in.
    margin = road_width_m * 1e-5
    west, south, east, north = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
    labels = roads.geometries.take(roads.query(box(west - margin, south - margin, east + margin, north + margin)))
    if not len(labels):
        return np.zeros(size, dtype=np.uint8)
    to_chip = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
    # Buffer in chip CRS units so the same code serves EPSG:4326 and EPSG:3857 chips.
    half_width = road_width_m / 2 / _metres_per_pixel(src) * abs(src.transform.a)
    shapes = []
    for label in labels:
        geometry = shapely_transform(lambda x, y, _z=None, t=to_chip: t.transform(x, y), label)
        if not geometry.geom_type.endswith("Polygon"):
            geometry = geometry.buffer(half_width)
        if not geometry.is_empty:
            shapes.append((geometry, 1))
    if not shapes:
        return np.zeros(size, dtype=np.uint8)
    return rasterize(shapes, out_shape=size, transform=_model_grid(src), dtype="uint8")


def _build_model(encoder_name: str, encoder_weights: str | None) -> Any:
    import segmentation_models_pytorch as smp

    return smp.Unet(encoder_name=encoder_name, encoder_weights=encoder_weights, in_channels=3, classes=1)


def _load_model(checkpoint: bytes) -> tuple[Any, str]:
    """Rebuild the U-Net from checkpoint bytes; returns the model and its encoder name."""
    import torch

    state = torch.load(io.BytesIO(checkpoint), map_location="cpu", weights_only=False)
    model = _build_model(state["encoder_name"], None)
    model.load_state_dict(state["state_dict"])
    return model.eval(), state["encoder_name"]


def _read_weights(base_model_weights: str) -> bytes | None:
    """Checkpoint bytes from a local path or a remote file (https or s3); None when no base weights are given."""
    if not base_model_weights:
        return None
    return resolve_path(base_model_weights).read_bytes()


def preprocess(image_path: Any) -> Any:
    """Read an RGB chip as a (1, 3, 256, 256) float32 tensor scaled to 0 to 1."""
    import numpy as np
    import rasterio

    with rasterio.open(image_path) as src:
        rgb = src.read([1, 2, 3], out_shape=(3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)).astype(np.float32) / 255.0
    return rgb[np.newaxis]


def postprocess(logits: Any, threshold: float) -> Any:
    """Turn (1, H, W) road logits into a uint8 road mask."""
    import numpy as np

    logits = np.asarray(logits, dtype=np.float64)
    probability = 1.0 / (1.0 + np.exp(-logits.reshape(logits.shape[-2:])))
    return (probability >= threshold).astype(np.uint8)


# Probability mosaics above this many pixels are vectorised chip by chip instead (memory guard).
MAX_MOSAIC_PIXELS = 150_000_000


def _mosaic_groups(chip_paths: list[Path]) -> list[dict[str, Any]]:
    """Group chips that share a CRS and pixel size, and place each on a common pixel grid.

    Tiles downloaded for one request are aligned, so a group's probability maps can be
    pasted into one mosaic and vectorised together; a road then comes out as one polygon
    across chip edges instead of being cut at every tile boundary.
    """
    import rasterio
    from rasterio.transform import Affine

    groups: dict[tuple[str, float, float], dict[str, Any]] = {}
    for chip_path in chip_paths:
        with rasterio.open(chip_path) as src:
            t = src.transform
            key = (str(src.crs), round(t.a, 12), round(t.e, 12))
            group = groups.setdefault(key, {"crs": src.crs, "res": (t.a, t.e), "chips": []})
            group["chips"].append(
                {"path": chip_path, "left": t.c, "top": t.f, "width": src.width, "height": src.height}
            )
    for group in groups.values():
        a, e = group["res"]
        left = min(c["left"] for c in group["chips"])
        top = max(c["top"] for c in group["chips"])
        for c in group["chips"]:
            c["col"] = round((c["left"] - left) / a)
            c["row"] = round((c["top"] - top) / e)
        group["width"] = max(c["col"] + c["width"] for c in group["chips"])
        group["height"] = max(c["row"] + c["height"] for c in group["chips"])
        group["transform"] = Affine(a, 0.0, left, 0.0, e, top)
    return list(groups.values())


def _polygon_features(probability: Any, transform: Any, crs: Any, params: dict[str, Any]) -> list[dict[str, Any]]:
    """Threshold one probability raster and vectorise the road mask into WGS84 polygon features."""
    import numpy as np
    from pyproj import Geod, Transformer
    from rasterio.features import shapes
    from shapely.geometry import mapping, shape
    from shapely.ops import transform as shapely_transform

    mask = (np.asarray(probability) >= float(params["confidence_threshold"])).astype(np.uint8)
    to_wgs84 = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    geod = Geod(ellps="WGS84")
    width = mask.shape[1]
    (lon0, lon1), (lat0, lat1) = to_wgs84.transform([transform.c, transform.c + transform.a * width], [transform.f] * 2)
    pixel_m = geod.inv(lon0, lat0, lon1, lat1)[2] / width
    # Simplify in raster CRS units so the tolerance means the same on EPSG:4326 and EPSG:3857 chips.
    tolerance = float(params["simplify_m"]) / pixel_m * abs(transform.a)
    features = []
    for geometry, _ in shapes(mask, mask=mask.astype(bool), transform=transform):
        polygon = shape(geometry).simplify(tolerance, preserve_topology=True)
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        if polygon.is_empty:
            continue
        geo_polygon = shapely_transform(lambda x, y, _z=None, t=to_wgs84: t.transform(x, y), polygon)
        if abs(geod.geometry_area_perimeter(geo_polygon)[0]) < float(params["min_area_m2"]):
            continue
        features.append(
            {"type": "Feature", "properties": {"class": 1, "label": "road"}, "geometry": mapping(geo_polygon)}
        )
    return features


def predict(session: Any, input_images: dict[str, str], params: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    import rasterio
    from skimage.transform import resize

    params = {**DEFAULT_INFERENCE_PARAMS, **(params or {})}
    input_name = session.get_inputs()[0].name

    chip_paths = _chip_paths(input_images[IMAGE_INPUT])
    if not chip_paths:
        msg = f"No georeferenced (.tif) chips found in {input_images[IMAGE_INPUT]}"
        raise FileNotFoundError(msg)

    def chip_probability(chip: dict[str, Any]) -> Any:
        logits = np.asarray(session.run(None, {input_name: preprocess(chip["path"])})[0])[0, 0]
        probability = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
        if probability.shape != (chip["height"], chip["width"]):
            probability = resize(probability, (chip["height"], chip["width"]), order=1, anti_aliasing=False)
        return probability.astype(np.float32)

    features: list[dict[str, Any]] = []
    for group in _mosaic_groups(chip_paths):
        if group["width"] * group["height"] <= MAX_MOSAIC_PIXELS:
            mosaic = np.zeros((group["height"], group["width"]), dtype=np.float32)
            for chip in group["chips"]:
                r, c = chip["row"], chip["col"]
                mosaic[r : r + chip["height"], c : c + chip["width"]] = chip_probability(chip)
            features.extend(_polygon_features(mosaic, group["transform"], group["crs"], params))
        else:
            for chip in group["chips"]:
                with rasterio.open(chip["path"]) as src:
                    features.extend(_polygon_features(chip_probability(chip), src.transform, src.crs, params))
    return {"type": "FeatureCollection", "features": features}


def _spatial_split(names: list[str], val_ratio: float, seed: int, block_size: int) -> tuple[list[str], list[str]]:
    """Hold out whole blocks of neighbouring tiles; chips with unparsable names form their own block."""
    import random

    blocks: dict[Any, list[str]] = {}
    for name in names:
        match = CHIP_NAME.search(name)
        key = (int(match[1]) // block_size, int(match[2]) // block_size) if match else name
        blocks.setdefault(key, []).append(name)

    keys = sorted(blocks, key=str)
    random.Random(seed).shuffle(keys)
    val_target = max(1, round(len(names) * val_ratio))
    val: list[str] = []
    for key in keys:
        # Stop before a block would leave nothing to train on.
        if len(val) >= val_target or len(val) + len(blocks[key]) >= len(names):
            break
        val.extend(blocks[key])
    if not val and block_size > 1:
        # Too few blocks to hold one out (small AOIs): fall back to holding out single chips.
        return _spatial_split(names, val_ratio, seed, 1)
    val_set = set(val)
    return sorted(n for n in names if n not in val_set), sorted(val)


def _load_xy(chip_paths: list[Path], roads: Any, road_width_m: float) -> tuple[Any, Any]:
    """Stack chips into (N, 3, 256, 256) float32 images and (N, 1, 256, 256) float32 road masks."""
    import numpy as np
    import rasterio

    images, masks = [], []
    for chip_path in chip_paths:
        images.append(preprocess(chip_path)[0])
        with rasterio.open(chip_path) as src:
            masks.append(_road_mask(src, roads, road_width_m)[np.newaxis].astype(np.float32))
    if not images:
        size = (0, 1, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)
        return np.zeros((0, 3, *size[2:]), dtype=np.float32), np.zeros(size, dtype=np.float32)
    return np.stack(images), np.stack(masks)


def _relaxed_scores(pred: Any, truth: Any, tolerance_px: int) -> tuple[float, float]:
    """Precision and recall of the road masks' skeletons, counting matches within `tolerance_px` pixels.

    Comparing centrelines rather than pixels keeps the score independent of road width.
    """
    import numpy as np
    from scipy.ndimage import binary_dilation
    from skimage.morphology import skeletonize

    pred_sk, truth_sk = skeletonize(pred), skeletonize(truth)
    footprint = np.ones((2 * tolerance_px + 1, 2 * tolerance_px + 1), dtype=bool)
    matched_pred = (pred_sk & binary_dilation(truth_sk, footprint)).sum()
    matched_truth = (truth_sk & binary_dilation(pred_sk, footprint)).sum()
    precision = matched_pred / pred_sk.sum() if pred_sk.any() else 0.0
    recall = matched_truth / truth_sk.sum() if truth_sk.any() else 0.0
    return float(precision), float(recall)


@step
def split_dataset(
    dataset_chips: dict[str, str],
    dataset_labels: str,
    hyperparameters: dict[str, Any],
) -> Annotated[dict[str, Any], "split_info"]:
    """Hold out spatial blocks of neighbouring OAM tiles so a road never straddles train and val."""
    names = [p.name for p in _chip_paths(dataset_chips[IMAGE_INPUT])]
    val_ratio = hyperparameters.get("val_ratio", 0.2)
    seed = hyperparameters.get("split_seed", 42)
    block_size = hyperparameters.get("block_size", 4)
    train_names, val_names = _spatial_split(names, val_ratio, seed, block_size)
    info = {
        "strategy": "spatial",
        "val_ratio": val_ratio,
        "seed": seed,
        "block_size": block_size,
        "train_count": len(train_names),
        "val_count": len(val_names),
        "train_chip_names": train_names,
        "val_chip_names": val_names,
        "description": f"Blocks of {block_size}x{block_size} neighbouring tiles are held out together.",
    }
    log_metadata(metadata={"fair/split": info})
    return info


def augment_batch(xb: Any, yb: Any, rng: Any, photometric: bool = True) -> tuple[Any, Any]:
    """Random flips, right-angle turns and, with `photometric`, zoom and colour jitter on a (N, 3, H, W) batch.

    Roads have no preferred orientation, so geometry is always augmented. The colour jitter
    (brightness, contrast, saturation, gamma, channel shift, noise) and zoom make the model
    learn road *shape* rather than the colours of one imagery source, which matters when a
    satellite-trained model is applied to drone imagery.
    """
    import torch
    import torch.nn.functional as functional

    if rng.random() < 0.5:
        xb, yb = xb.flip(-1), yb.flip(-1)
    turns = int(rng.integers(4))
    xb, yb = xb.rot90(turns, (-2, -1)).contiguous(), yb.rot90(turns, (-2, -1)).contiguous()
    if not photometric:
        return xb, yb
    n, _, h, w = xb.shape
    if rng.random() < 0.5:
        size = int(h * rng.uniform(0.7, 1.0))
        top, left = int(rng.integers(h - size + 1)), int(rng.integers(w - size + 1))
        window = (slice(None), slice(None), slice(top, top + size), slice(left, left + size))
        xb = functional.interpolate(xb[window], size=(h, w), mode="bilinear", align_corners=False)
        yb = (functional.interpolate(yb[window], size=(h, w), mode="nearest") > 0.5).float()
    device = xb.device
    brightness = torch.empty(n, 1, 1, 1, device=device).uniform_(-0.15, 0.15)
    contrast = torch.empty(n, 1, 1, 1, device=device).uniform_(0.7, 1.3)
    saturation = torch.empty(n, 1, 1, 1, device=device).uniform_(0.6, 1.4)
    gamma = torch.empty(n, 1, 1, 1, device=device).uniform_(0.7, 1.4)
    shift = torch.empty(n, 3, 1, 1, device=device).uniform_(-0.06, 0.06)
    gray = xb.mean(1, keepdim=True)
    xb = gray + (xb - gray) * saturation
    mean = xb.mean((1, 2, 3), keepdim=True)
    xb = ((xb - mean) * contrast + mean + brightness + shift).clamp(0, 1).pow(gamma)
    if rng.random() < 0.3:
        xb = xb + torch.randn_like(xb) * 0.02
    return xb.clamp(0, 1), yb


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
    """Fine-tune the U-Net on the train split with BCE plus Dice loss and geometric and colour augmentation."""
    import numpy as np
    import segmentation_models_pytorch as smp
    import torch

    seed = int(split_info["seed"])
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    train = set(split_info["train_chip_names"])
    chips = [p for p in _chip_paths(dataset_chips[IMAGE_INPUT]) if p.name in train]
    x, y = _load_xy(chips, _label_geometries(dataset_labels), hyperparameters.get("road_width_m", 6.0))

    weights = _read_weights(base_model_weights)
    if weights is not None:
        model, encoder_name = _load_model(weights)
    else:
        encoder_name = hyperparameters.get("encoder_name", DEFAULT_ENCODER)
        model = _build_model(encoder_name, hyperparameters.get("encoder_weights", "imagenet"))

    use_cuda = torch.cuda.is_available() and os.environ.get("FAIR_FORCE_CPU") != "1"
    device = "cuda" if use_cuda else "cpu"
    model = model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=hyperparameters.get("learning_rate", 1e-4))
    bce, dice = torch.nn.BCEWithLogitsLoss(), smp.losses.DiceLoss(mode="binary")
    batch_size = int(hyperparameters.get("batch_size", 8))

    train_losses: list[float] = []
    with mlflow_training_context(hyperparameters, model_name, base_model_id, dataset_id):
        for _ in range(int(hyperparameters.get("epochs", 30))):
            order, epoch_loss = rng.permutation(len(x)), 0.0
            for start in range(0, len(x), batch_size):
                idx = order[start : start + batch_size]
                xb, yb = torch.from_numpy(x[idx]).to(device), torch.from_numpy(y[idx]).to(device)
                xb, yb = augment_batch(xb, yb, rng, photometric=bool(hyperparameters.get("photometric_augment", True)))
                logits = model(xb)
                loss = bce(logits, yb) + dice(logits, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(idx)
            train_losses.append(epoch_loss / max(len(x), 1))
    log_loss_history(train_losses, [])

    buffer = io.BytesIO()
    torch.save({"encoder_name": encoder_name, "state_dict": model.cpu().state_dict()}, buffer)
    return buffer.getvalue()


@step
def evaluate_model(
    trained_model: bytes,
    dataset_chips: dict[str, str],
    dataset_labels: str,
    hyperparameters: dict[str, Any],
    split_info: dict[str, Any],
    class_names: list[str] | None = None,
) -> Annotated[dict[str, Any], "metrics"]:
    """Report road pixel IoU and relaxed centreline precision, recall and F1 on the val split."""
    import numpy as np
    import torch

    model, _ = _load_model(trained_model)
    val = set(split_info["val_chip_names"])
    chips = [p for p in _chip_paths(dataset_chips[IMAGE_INPUT]) if p.name in val]
    x, y = _load_xy(chips, _label_geometries(dataset_labels), hyperparameters.get("road_width_m", 6.0))
    threshold = hyperparameters.get("confidence_threshold", DEFAULT_INFERENCE_PARAMS["confidence_threshold"])
    tolerance = int(hyperparameters.get("relaxed_tolerance_px", 3))

    batch_size = int(hyperparameters.get("batch_size", 8))
    with torch.no_grad():
        # Batched so a real validation split (hundreds of chips) does not go through the net at once.
        probs = (
            np.concatenate(
                [
                    torch.sigmoid(model(torch.from_numpy(x[i : i + batch_size]))).numpy()[:, 0]
                    for i in range(0, len(x), batch_size)
                ]
            )
            if len(x)
            else np.zeros((0, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), dtype=np.float32)
        )
    pred, truth = probs >= threshold, y[:, 0].astype(bool)

    scores = [_relaxed_scores(p, t, tolerance) for p, t in zip(pred, truth, strict=True)]
    precision = float(np.mean([s[0] for s in scores])) if scores else 0.0
    recall = float(np.mean([s[1] for s in scores])) if scores else 0.0
    union = (pred | truth).sum()
    metrics = {
        "iou_road": float((pred & truth).sum() / union) if union else 0.0,
        "relaxed_precision": precision,
        "relaxed_recall": recall,
        "relaxed_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
    }
    log_evaluation_results(metrics)
    return metrics


@step(output_materializers={"onnx_model": ONNXMaterializer})
def export_onnx(trained_model: bytes) -> Annotated[bytes, "onnx_model"]:
    """Export the U-Net to single-file ONNX bytes."""
    import tempfile

    import onnx
    import torch

    model, _ = _load_model(trained_model)
    # The serving runtime feeds one chip at a time, so export at batch 1.
    dummy = torch.randn(1, 3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = str(Path(tmpdir) / "model.onnx")
        torch.onnx.export(model, (dummy,), path, input_names=["image"], output_names=["logits"], dynamo=True)
        # Dynamo writes weights to a sidecar `.onnx.data`; inline so the returned bytes are self-contained.
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
    )
    export_onnx(trained_model=trained_model)


@pipeline
def inference_pipeline(
    model_uri: str,
    input_images: dict[str, str],
    inference_params: dict[str, Any],
) -> None:
    run_inference(model_uri=model_uri, input_images=input_images, inference_params=inference_params)
