"""unet-drainage-channels: a U-Net that traces open drainage channels in RGB drone imagery as lines, finetunable
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
CLASS_NAMES = ["background", "drain"]
# Label value for pixels that count in neither the loss nor the pixel scores.
IGNORE = 255
# OpenStreetMap waterway values the model maps; a label with another waterway value is left out.
DRAIN_WATERWAYS = {"drain", "ditch"}
# A drain is drawn at its `width` tag, or at DEFAULT_WIDTH_M without one.
DEFAULT_WIDTH_M = 1.0
# Ignored on each side of every drain, so a line mapped a little off the channel does not teach the model
# that the channel is background.
MARGIN_M = 1.0
# Predictions this close to the no-data edge of the image are dropped.
EDGE_M = 2.0
# Predicted pieces smaller than this are dropped before lines are traced.
MIN_PIECE_M2 = 1.0
# Traced lines are simplified to within this distance.
SIMPLIFY_M = 0.3
# Distances at which predicted and labelled lines are compared.
TOLERANCES_M = (1, 2, 4)
# Training tries a label every SLIDE_STEP pixels, and each pixel of shift costs SLIDE_PRIOR, so a tie keeps
# the label where it was mapped.
SLIDE_STEP = 2
SLIDE_PRIOR = 0.002
# The mask is extended by this many pixels before thinning, so a line runs to the edge of its chip.
SKELETON_PAD = 32
# fAIr names each chip after its web map tile: OAM-<x>-<y>-<zoom>.tif.
OAM_TILE = re.compile(r"OAM-(\d+)-(\d+)-\d+")
WIDTH = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s*m?\s*$")


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


def _probability(logits: Any) -> Any:
    import numpy as np

    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(logits, dtype=np.float64), -50.0, 50.0)))


def postprocess(logits: Any, threshold: float) -> Any:
    """Turn (1, H, W) drain logits, or a batch of them, into a uint8 mask that is 1 where the drain
    probability is above the threshold."""
    import numpy as np

    return (_probability(logits)[..., 0, :, :] > threshold).astype(np.uint8)


def _metres_per_pixel(src: Any) -> float:
    """Ground width of one pixel of an open chip, whatever its coordinate system."""
    from pyproj import Geod
    from rasterio.warp import transform_bounds

    west, south, east, north = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
    middle = (south + north) / 2
    return Geod(ellps="WGS84").inv(west, middle, east, middle)[2] / src.width


def _no_data(rgb: Any) -> Any:
    """Pixels outside the drone image: black in all three bands of a (3, H, W) chip scaled 0 to 255."""
    import numpy as np

    return np.asarray(rgb).max(axis=0) <= 3


def _edge(no_data: Any, gsd: float) -> Any:
    """Pixels outside the drone image or within EDGE_M of its edge."""
    from scipy import ndimage

    return ndimage.distance_transform_edt(~no_data) <= EDGE_M / gsd if no_data.any() else no_data


def _clean(mask: Any, no_data: Any, gsd: float) -> Any:
    """A drain mask without its pixels at the edge of the image and without pieces under MIN_PIECE_M2.

    A model can score well on line length by scattering dots near the labels; dropping tiny pieces before
    anything is traced or scored removes that way out.
    """
    import numpy as np
    from scipy import ndimage

    mask = np.asarray(mask, bool) & ~_edge(no_data, gsd)
    # Pixels that touch at a corner belong to the same piece.
    pieces, count = ndimage.label(mask, structure=np.ones((3, 3), bool))
    if not count:
        return mask
    keep = np.bincount(pieces.ravel()) >= MIN_PIECE_M2 / gsd**2
    keep[0] = False
    return keep[pieces]


def _centrelines(mask: Any, gsd: float, min_length_m: float) -> list[Any]:
    """Centrelines of a drain mask as shapely lines in pixel coordinates, one vertex per pixel centre.

    The mask is thinned to lines one pixel wide, neighbouring pixels are joined, and the joins are merged
    into one line between each pair of ends or junctions. A loose end shorter than `min_length_m` is a
    side branch of the thinning or a speck, and is dropped.
    """
    import numpy as np
    import shapely
    from skimage.morphology import skeletonize

    pad = SKELETON_PAD
    s = skeletonize(np.pad(np.asarray(mask, bool), pad, mode="edge"))[pad:-pad, pad:-pad]
    # Pixel pairs to join: right, down, and the two diagonals where no straight step already links the pair.
    joins = (
        (s[:, :-1] & s[:, 1:], (1, 0)),
        (s[:-1, :] & s[1:, :], (0, 1)),
        (s[:-1, :-1] & s[1:, 1:] & ~s[:-1, 1:] & ~s[1:, :-1], (1, 1)),
        (s[:-1, 1:] & s[1:, :-1] & ~s[:-1, :-1] & ~s[1:, 1:], (-1, 1)),
    )
    segments = []
    for linked, (dx, dy) in joins:
        rows, cols = np.nonzero(linked)
        start = np.stack([cols + (1.5 if dx < 0 else 0.5), rows + 0.5], axis=1)
        segments.append(np.stack([start, start + np.array([dx, dy])], axis=1))
    segments = np.concatenate(segments)
    if not len(segments):
        return []

    def merged(parts: Any) -> list[Any]:
        return list(shapely.get_parts(shapely.line_merge(shapely.multilinestrings(parts))))

    lines = merged(shapely.linestrings(segments))
    for _ in range(3):
        ends: dict[tuple[float, ...], int] = {}
        for line in lines:
            for end in (line.coords[0], line.coords[-1]):
                ends[end] = ends.get(end, 0) + 1
        kept = [
            line
            for line in lines
            if line.length * gsd >= min_length_m or (ends[line.coords[0]] > 1 and ends[line.coords[-1]] > 1)
        ]
        if len(kept) == len(lines):
            break
        lines = merged(kept) if kept else []
    return lines


def predict(session: Any, input_images: dict[str, str], params: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    import rasterio
    from pyproj import Transformer

    threshold = float(params.get("confidence_threshold", 0.3))
    min_length_m = float(params.get("min_length_m", 2.0))
    input_name = session.get_inputs()[0].name
    chip_paths = _chip_paths(input_images[IMAGE_INPUT])
    if not chip_paths:
        msg = f"No georeferenced (.tif) chips found in {input_images[IMAGE_INPUT]}"
        raise FileNotFoundError(msg)

    features: list[dict[str, Any]] = []
    for chip_path in chip_paths:
        with rasterio.open(chip_path) as src:
            transform, gsd = src.transform, _metres_per_pixel(src)
            to_wgs84 = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
        image = preprocess(chip_path)
        logits = session.run(None, {input_name: image})[0][0]
        probability = _probability(logits)[0]
        mask = _clean(postprocess(logits, threshold), _no_data(image[0] * 255.0), gsd)
        for line in _centrelines(mask, gsd, min_length_m):
            cols, rows = np.asarray(line.coords).astype(int).T
            x, y = transform * np.asarray(line.simplify(SIMPLIFY_M / gsd).coords).T
            lon, lat = to_wgs84.transform(x, y)
            features.append(
                {
                    "type": "Feature",
                    "properties": {
                        "class": 1,
                        "label": CLASS_NAMES[1],
                        "score": round(float(probability[rows, cols].mean()), 3),
                    },
                    "geometry": {
                        "type": "LineString",
                        "coordinates": [[float(a), float(b)] for a, b in zip(lon, lat, strict=True)],
                    },
                }
            )
    return {"type": "FeatureCollection", "features": features}


def _device() -> str:
    if os.environ.get("FAIR_FORCE_CPU", "").lower() in {"1", "true", "yes", "on"}:
        return "cpu"
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _build_model() -> Any:
    """The network inside the ONNX file: ImageNet normalisation, then a ResNet-34 U-Net with one drain logit."""
    import segmentation_models_pytorch as smp
    import torch

    class DrainNet(torch.nn.Module):
        mean: torch.Tensor
        std: torch.Tensor

        def __init__(self) -> None:
            super().__init__()
            self.unet = smp.Unet(ENCODER, encoder_weights=None, in_channels=3, classes=1)
            self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.unet((x - self.mean) / self.std)

    return DrainNet()


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


def _width_m(tags: dict[str, Any]) -> float:
    """A drain's `width` tag in metres, kept between 0.2 and 10 m, or DEFAULT_WIDTH_M when it has none."""
    match = WIDTH.match(str(tags.get("width", "")))
    return min(max(float(match[1]), 0.2), 10.0) if match else DEFAULT_WIDTH_M


def _is_covered(tags: dict[str, Any]) -> bool:
    """Culverts and covered or underground drains, which cannot be seen from above."""
    return (
        str(tags.get("tunnel")) in {"yes", "culvert"}
        or str(tags.get("covered")) == "yes"
        or str(tags.get("layer", "0")).startswith("-")
    )


def _drain_labels(dataset_labels: str) -> list[tuple[Any, float, bool]]:
    """(line in EPSG:4326, width in metres, covered) for every labelled drain.

    Labels fAIr downloads from OpenStreetMap keep their tags under `tags`.
    """
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
    drains = []
    for feature in json.loads(label_file.read_text(encoding="utf-8"))["features"]:
        geometry = feature.get("geometry") or {}
        properties = feature.get("properties") or {}
        tags = {**properties, **(properties.get("tags") or {})}
        if geometry.get("type") not in {"LineString", "MultiLineString"}:
            continue
        if tags.get("waterway", "drain") not in DRAIN_WATERWAYS:
            continue
        drains.append((shape(geometry), _width_m(tags), _is_covered(tags)))
    if not drains:
        msg = f"No drain lines found in {label_file}: the labels must be LineString features"
        raise ValueError(msg)
    return drains


def _chip_footprints(paths: list[Path]) -> list[Any]:
    import rasterio
    from rasterio.warp import transform_bounds
    from shapely.geometry import box

    footprints = []
    for path in paths:
        with rasterio.open(path) as src:
            footprints.append(box(*transform_bounds(src.crs, "EPSG:4326", *src.bounds)))
    return footprints


def _load_chips(
    chips_href: str, names: list[str], labels: list[tuple[Any, float, bool]]
) -> tuple[Any, Any, Any, Any, Any]:
    """Chips as (N, 3, 256, 256) uint8 RGB and, for each chip:

    - the label mask (N, 256, 256): 1 on open drains at their width, IGNORE on a margin of MARGIN_M beside
      every drain, on covered drains and at the edge of the image, 0 elsewhere;
    - the centreline pixels of the open drains;
    - the pixels where a predicted line is neither right nor wrong: covered drains and the edge of the image;
    - the ground size of a pixel in metres.
    """
    import numpy as np
    import rasterio
    from pyproj import Transformer
    from rasterio.features import rasterize
    from shapely.ops import transform as shapely_transform
    from shapely.strtree import STRtree

    paths = {p.name: p for p in _chip_paths(chips_href)}
    tree = STRtree([geometry for geometry, _, _ in labels])
    shape = (CHIP_SIZE, CHIP_SIZE)
    images = np.zeros((len(names), 3, *shape), np.uint8)
    masks = np.zeros((len(names), *shape), np.uint8)
    centres = np.zeros((len(names), *shape), bool)
    skips = np.zeros((len(names), *shape), bool)
    gsds = np.zeros(len(names))
    for i, (name, footprint) in enumerate(zip(names, _chip_footprints([paths[n] for n in names]), strict=True)):
        images[i] = np.round(preprocess(paths[name])[0] * 255)
        with rasterio.open(paths[name]) as src:
            to_chip = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
            transform, gsds[i] = src.transform, _metres_per_pixel(src)
        # Chip coordinate units per metre on the ground.
        scale = abs(transform.a) / gsds[i]
        drains = [
            (shapely_transform(lambda x, y, _z=None, t=to_chip: t.transform(x, y), labels[j][0]), *labels[j][1:])
            for j in tree.query(footprint)
        ]

        def burn(shapes: list[Any], all_touched: bool = False, transform: Any = transform) -> Any:
            if not shapes:
                return np.zeros(shape, bool)
            return rasterize(shapes, out_shape=shape, transform=transform, all_touched=all_touched).astype(bool)

        def band(line: Any, width_m: float, scale: float = scale) -> Any:
            return line.buffer(width_m / 2 * scale, cap_style="flat")

        # Later layers win: an open drain overrides the margin of a neighbour.
        masks[i][burn([band(line, width + 2 * MARGIN_M) for line, width, _ in drains])] = IGNORE
        masks[i][burn([band(line, width) for line, width, covered in drains if covered])] = IGNORE
        masks[i][burn([band(line, width) for line, width, covered in drains if not covered])] = 1
        centres[i] = burn([line for line, _, covered in drains if not covered], all_touched=True)
        skips[i] = burn([band(line, width + 2 * MARGIN_M) for line, width, covered in drains if covered])
        edge = _edge(_no_data(images[i]), gsds[i])
        masks[i][edge] = IGNORE
        centres[i] &= ~edge
        skips[i] |= edge
    return images, masks, centres, skips, gsds


def _drain_length_per_chip(chips_href: str, names: list[str], labels: list[tuple[Any, float, bool]]) -> Any:
    """Length of open drain inside each chip, as an (N, 1) array in degrees: only its share per chip matters."""
    import numpy as np
    from shapely.strtree import STRtree

    paths = {p.name: p for p in _chip_paths(chips_href)}
    tree = STRtree([geometry for geometry, _, _ in labels])
    lengths = np.zeros((len(names), 1))
    for i, footprint in enumerate(_chip_footprints([paths[n] for n in names])):
        for j in tree.query(footprint):
            if not labels[j][2]:
                lengths[i, 0] += labels[j][0].intersection(footprint).length
    return lengths


def _augment(images: Any, masks: Any) -> tuple[Any, Any]:
    """Random quarter turns and flips (nadir imagery has no up) and small brightness, contrast, colour and
    saturation changes.

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
        image = image * np.random.uniform(0.9, 1.1, (3, 1, 1)).astype(np.float32)
        grey = image.mean(axis=0, keepdims=True)
        image = grey + (image - grey) * np.random.uniform(0.7, 1.3)
        x.append(np.clip(image, 0, 255) / 255.0)
        y.append(mask)
    # Turned arrays keep their turned memory layout through np.stack; the losses need plain C order.
    return np.ascontiguousarray(np.stack(x), np.float32), np.ascontiguousarray(np.stack(y), np.int64)


def _drain_weight(masks: Any, max_weight: float) -> float:
    """How much more a drain pixel counts than a background pixel in the cross-entropy: the square root of
    background share over drain share, from 1 to max_weight."""
    background, drain = int((masks == 0).sum()), int((masks == 1).sum())
    return float(min(max(math.sqrt(background / max(drain, 1)), 1.0), max_weight))


def _loss(logits: Any, target: Any, drain_weight: float) -> Any:
    """Weighted binary cross-entropy plus soft Dice on the labelled pixels; IGNORE pixels count in neither."""
    import torch
    import torch.nn.functional as F

    logit = logits[:, 0]
    valid = (target != IGNORE).float()
    drain = (target == 1).float()
    weight = torch.tensor(drain_weight, device=logit.device)
    cross_entropy = F.binary_cross_entropy_with_logits(logit, drain, pos_weight=weight, reduction="none")
    cross_entropy = (cross_entropy * valid).sum() / valid.sum().clamp(min=1)
    probability = torch.sigmoid(logit) * valid
    dice = 1 - (2 * (probability * drain).sum() + 1) / (probability.sum() + drain.sum() + 1)
    return cross_entropy + dice


def _slide_radius(gsd: float, tolerance_m: float) -> int:
    """The furthest a label may slide, in pixels: a multiple of SLIDE_STEP within the tolerance and the chip."""
    return min(SLIDE_STEP * (int(tolerance_m / gsd) // SLIDE_STEP), CHIP_SIZE // 4)


def _slide_labels(logits: Any, target: Any, radius: int, drain_weight: float) -> tuple[Any, list[tuple[int, int]]]:
    """Each chip's label mask moved by up to `radius` pixels to where the model's own prediction disagrees
    with it least, and the (down, right) shift chosen for each chip.

    Drains in OpenStreetMap are often mapped a metre or two off the channel in the imagery. Training against
    the label where it fits the image best, within a small window, stops those offsets from blurring the
    model. Every position is judged on the same interior pixels, so none gains by pushing a label off the
    chip. It tolerates misplaced labels; it does not measure how far they are off.
    """
    import torch
    import torch.nn.functional as F

    logit = logits[:, 0].float()
    cost_drain, cost_background = drain_weight * F.softplus(-logit), F.softplus(logit)
    height, width = target.shape[1:]
    padded = F.pad(target, (radius, radius, radius, radius), value=IGNORE)
    inner = torch.zeros((height, width), dtype=torch.bool, device=target.device)
    inner[radius : height - radius, radius : width - radius] = True
    steps = range(-radius, radius + 1, SLIDE_STEP)
    shifts = [(down, right) for down in steps for right in steps]

    def at(down: int, right: int) -> Any:
        return padded[:, radius - down : radius - down + height, radius - right : radius - right + width]

    costs = []
    for down, right in shifts:
        moved = at(down, right)
        drain, background = (moved == 1) & inner, (moved == 0) & inner
        costs.append(
            (cost_drain * drain).sum((1, 2)) / drain.sum((1, 2)).clamp(min=1)
            + (cost_background * background).sum((1, 2)) / background.sum((1, 2)).clamp(min=1)
            + SLIDE_PRIOR * (abs(down) + abs(right))
        )
    chosen = [shifts[j] for j in torch.stack(costs).argmin(0).tolist()]
    return torch.stack([at(*shift)[i] for i, shift in enumerate(chosen)]), chosen


def _predict_drains(
    model: Any, images: Any, gsds: Any, threshold: float, min_length_m: float, batch_size: int = 8
) -> tuple[Any, Any]:
    """For uint8 chips (N, 3, 256, 256): the cleaned drain masks and the pixels of the lines traced from
    them, both (N, 256, 256) bool."""
    import numpy as np
    import torch

    device = _device()
    model = model.to(device).eval()
    drains = np.zeros((len(images), CHIP_SIZE, CHIP_SIZE), bool)
    traced = np.zeros_like(drains)
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            x = torch.from_numpy(images[start : start + batch_size].astype(np.float32) / 255.0).to(device)
            for i, mask in enumerate(postprocess(model(x).cpu().numpy(), threshold), start):
                drains[i] = _clean(mask, _no_data(images[i]), gsds[i])
                for line in _centrelines(drains[i], gsds[i], min_length_m):
                    cols, rows = np.asarray(line.coords).astype(int).T
                    traced[i, rows, cols] = True
    return drains, traced


def _scores(drains: Any, traced: Any, masks: Any, centres: Any, skips: Any, gsds: Any) -> dict[str, float]:
    """Pixel scores of the predicted drain masks and length scores of the predicted lines.

    Completeness is the share of labelled drain centreline with a predicted line within the tolerance.
    Correctness is the share of predicted line with a labelled centreline within the tolerance; predicted
    line over a covered drain or at the edge of the image counts as neither right nor wrong.
    """
    import numpy as np
    from scipy import ndimage

    tally = dict.fromkeys(("tp", "fp", "fn", "labelled"), 0)
    tally |= {f"{key}_{tolerance}m": 0 for key in ("found", "traced", "correct") for tolerance in TOLERANCES_M}
    for drain, line, mask, centre, skip, gsd in zip(drains, traced, masks, centres, skips, gsds, strict=True):
        tally["tp"] += int((drain & (mask == 1)).sum())
        tally["fp"] += int((drain & (mask == 0)).sum())
        tally["fn"] += int((~drain & (mask == 1)).sum())
        tally["labelled"] += int(centre.sum())
        to_line = ndimage.distance_transform_edt(~line) if line.any() else np.full(line.shape, np.inf)
        to_centre = ndimage.distance_transform_edt(~centre) if centre.any() else np.full(line.shape, np.inf)
        for tolerance in TOLERANCES_M:
            near = to_centre <= tolerance / gsd
            counted = line & (near | ~skip)
            tally[f"found_{tolerance}m"] += int((centre & (to_line <= tolerance / gsd)).sum())
            tally[f"traced_{tolerance}m"] += int(counted.sum())
            tally[f"correct_{tolerance}m"] += int((counted & near).sum())

    metrics = {
        "pixel_iou": tally["tp"] / max(tally["tp"] + tally["fp"] + tally["fn"], 1),
        "pixel_precision": tally["tp"] / max(tally["tp"] + tally["fp"], 1),
        "pixel_recall": tally["tp"] / max(tally["tp"] + tally["fn"], 1),
    }
    for tolerance in TOLERANCES_M:
        completeness = tally[f"found_{tolerance}m"] / max(tally["labelled"], 1)
        correctness = tally[f"correct_{tolerance}m"] / max(tally[f"traced_{tolerance}m"], 1)
        metrics[f"completeness_{tolerance}m"] = completeness
        metrics[f"correctness_{tolerance}m"] = correctness
        metrics[f"f1_{tolerance}m"] = 2 * completeness * correctness / max(completeness + correctness, 1e-9)
    return metrics


def _spatial_split(
    names: list[str], drains: Any, val_ratio: float, seed: int, block_size: int, tries: int = 200
) -> tuple[list[str], list[str]]:
    """Assign whole blocks of neighbouring chips to train or validation, so validation areas are unseen.

    Of `tries` seeded shuffles of the blocks, keep the one whose validation set holds the share of drain
    length closest to `val_ratio`.
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
    totals = drains.sum(axis=0)
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
        share = drains[val].sum(axis=0) / np.maximum(totals, 1e-12)
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
    """Spatial block split on OAM tile coordinates, balanced so about val_ratio of the drain length is held out."""
    names = [p.name for p in _chip_paths(dataset_chips[IMAGE_INPUT])]
    if not names:
        msg = f"No .tif chips found in {dataset_chips[IMAGE_INPUT]}"
        raise FileNotFoundError(msg)
    val_ratio = float(hyperparameters.get("val_ratio", 0.2))
    seed = int(hyperparameters.get("split_seed", 42))
    block_size = int(hyperparameters.get("block_size", 8))
    drains = _drain_length_per_chip(dataset_chips[IMAGE_INPUT], names, _drain_labels(dataset_labels))
    train_names, val_names = _spatial_split(names, drains, val_ratio, seed, block_size)

    info = {
        "strategy": "spatial",
        "val_ratio": val_ratio,
        "seed": seed,
        "block_size": block_size,
        "train_count": len(train_names),
        "val_count": len(val_names),
        "description": (
            f"Chips named OAM-<x>-<y>-<z>.tif are grouped into {block_size} x {block_size} tile blocks and whole "
            "blocks go to train or validation, choosing among seeded shuffles the one that holds out the share "
            "of drain length closest to the validation ratio; other chips, or a dataset inside one block, are "
            "held out one by one."
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
    """Finetune the base U-Net on the training chips with weighted cross-entropy plus Dice.

    After the warm-up epochs each chip's label is first slid, within slide_tolerance_m, to where it fits the
    model's prediction best (see _slide_labels). Margins, covered drains and image edges count in no loss.
    """
    import mlflow
    import numpy as np
    import torch

    if num_classes != len(CLASS_NAMES):
        msg = f"unet-drainage-channels predicts {len(CLASS_NAMES)} classes {CLASS_NAMES}, got num_classes={num_classes}"
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
    images, masks, _, _, gsds = _load_chips(dataset_chips[IMAGE_INPUT], names, _drain_labels(dataset_labels))
    drain_weight = _drain_weight(masks, float(hyperparameters.get("max_drain_weight", 10.0)))
    gsd = float(np.median(gsds))
    radius = _slide_radius(gsd, float(hyperparameters.get("slide_tolerance_m", 2.0)))
    warmup = int(hyperparameters.get("slide_warmup_epochs", 1))

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
    max_grad_norm = float(hyperparameters.get("max_grad_norm", 1.0))

    losses: list[float] = []
    slid: list[float] = []
    with mlflow_training_context(hyperparameters, model_name, base_model_id, dataset_id):
        for epoch in range(epochs):
            model.train()
            order = np.random.permutation(len(images))
            total = 0.0
            for start in range(0, len(order), batch_size):
                batch = order[start : start + batch_size]
                x, y = (torch.from_numpy(a).to(device) for a in _augment(images[batch], masks[batch]))
                logits = model(x)
                if radius and epoch >= warmup:
                    y, shifts = _slide_labels(logits.detach(), y, radius, drain_weight)
                    slid += [math.hypot(*shift) * gsd for shift in shifts]
                loss = _loss(logits, y, drain_weight)
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
    log_metadata(
        metadata={
            "fair/drain_weight": drain_weight,
            "fair/slide": {"limit_m": radius * gsd, "mean_shift_m": float(np.mean(slid)) if slid else 0.0},
        }
    )

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
    """Score the predicted lines and masks on the validation chips, against the labels as they are mapped.

    The headline scores are on line length (see _scores): a drain is a thin feature, and a pixel score
    punishes a line for being half a metre off.
    """
    threshold = float(hyperparameters.get("confidence_threshold", 0.3))
    min_length_m = float(hyperparameters.get("min_length_m", 2.0))
    images, masks, centres, skips, gsds = _load_chips(
        dataset_chips[IMAGE_INPUT], split_info["val_chip_names"], _drain_labels(dataset_labels)
    )
    drains, traced = _predict_drains(_load_model(trained_model), images, gsds, threshold, min_length_m)
    metrics = _scores(drains, traced, masks, centres, skips, gsds)
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
