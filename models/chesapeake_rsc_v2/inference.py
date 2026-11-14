"""Blend overlapping ONNX predictions on a common road-model grid."""

from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.coords import BoundingBox
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import Affine, from_bounds

from .imagery import chip_paths, padded_size, read_chip, target_shape, validate_rgb

PATCH_SIZE = 512
STRIDE = PATCH_SIZE // 2
MAX_MOSAIC_PIXELS = 16_000_000


def _fill_batch(batch: np.ndarray, rgb: np.ndarray, scale: float) -> None:
    _, height, width = rgb.shape
    pixels = batch[0]
    np.divide(rgb, scale, out=pixels[:, :height, :width], dtype=np.float32)
    pixels[:, :height, width:] = pixels[:, :height, width - 1 : width]
    pixels[:, height:, :] = pixels[:, height - 1 : height, :]


def preprocess(image_path: Any) -> np.ndarray:
    """Return a padded NCHW batch at approximately 0.6 m GSD."""
    with rasterio.open(image_path) as src:
        rgb, _, _ = read_chip(src)
    _, height, width = rgb.shape
    batch = np.empty((1, 3, padded_size(height), padded_size(width)), dtype=np.float32)
    _fill_batch(batch, rgb, 1)
    return batch


def postprocess(logits: Any) -> np.ndarray:
    """Convert two-class logits to road probabilities."""
    logits = np.asarray(logits)
    probability = logits[:, 0] - logits[:, 1]
    np.clip(probability, -80, 80, out=probability)
    np.exp(probability, out=probability)
    probability += 1
    np.reciprocal(probability, out=probability)
    return probability


def _mosaic(paths: list[Path]) -> tuple[np.ndarray, np.ndarray, Affine, CRS]:
    bounds = []
    with rasterio.open(paths[0]) as first:
        crs = first.crs
    for path in paths:
        with rasterio.open(path) as src:
            validate_rgb(src)
            t = src.transform
            if src.crs != crs or t.b or t.d or t.a <= 0 or t.e >= 0:
                raise ValueError("Chips must share a CRS and be north-up")
            bounds.append(src.bounds)
    extent = BoundingBox(
        min(b.left for b in bounds),
        min(b.bottom for b in bounds),
        max(b.right for b in bounds),
        max(b.top for b in bounds),
    )
    height, width = target_shape(extent, crs)
    if height * width > MAX_MOSAIC_PIXELS:
        raise ValueError("Prediction area exceeds 16 million pixels at 0.6 m GSD. Split it into smaller areas.")
    grid = from_bounds(*extent, width, height)
    inverse = ~grid
    rgb = np.zeros((3, height, width), dtype=np.uint8)
    valid = np.zeros((height, width), dtype=bool)
    for path, b in zip(paths, bounds, strict=True):
        # Snap shared bounds to the same pixel edge, even when geographic tiles vary in height.
        left, top = map(round, inverse * (b.left, b.top))
        right, bottom = map(round, inverse * (b.right, b.bottom))
        h, w = bottom - top, right - left
        if h <= 0 or w <= 0:
            continue
        with rasterio.open(path) as src:
            image = src.read([1, 2, 3], out_shape=(3, h, w), resampling=Resampling.bilinear)
            mask = src.read_masks([1, 2, 3], out_shape=(3, h, w)).all(axis=0)
        np.copyto(rgb[:, top:bottom, left:right], image, where=mask[None])
        valid[top:bottom, left:right] |= mask
    return rgb, valid, grid, crs


def _patch_starts(extent: int) -> list[int]:
    last = max(0, extent - PATCH_SIZE)
    return sorted({*range(0, last + 1, STRIDE), last})


def _blend_probabilities(session: Any, rgb: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Blend overlapping road probabilities with positive Hann weights."""
    height, width = valid.shape
    h, w = min(height, PATCH_SIZE), min(width, PATCH_SIZE)
    weights = np.maximum(np.outer(np.hanning(h), np.hanning(w)), 1e-3).astype(np.float32)
    score = np.zeros((height, width), dtype=np.float32)
    weight_sum = np.zeros_like(score)
    batch = np.empty((1, 3, padded_size(h), padded_size(w)), dtype=np.float32)
    window_weights = np.empty_like(weights)
    input_name = session.get_inputs()[0].name
    column_starts = _patch_starts(width)
    for y in _patch_starts(height):
        for x in column_starts:
            window = np.s_[y : y + h, x : x + w]
            mask = valid[window]
            if not mask.any():
                continue
            _fill_batch(batch, rgb[:, window[0], window[1]], 255)
            logits = session.run(None, {input_name: batch})[0][:, :, :h, :w]
            if not np.isfinite(logits).all():
                raise ValueError("Model produced non-finite road logits")
            np.multiply(weights, mask, out=window_weights)
            probability = postprocess(logits)[0]
            probability *= window_weights
            score[window] += probability
            weight_sum[window] += window_weights
    np.maximum(weight_sum, 1e-3, out=weight_sum)
    score /= weight_sum
    return score


def predict(session: Any, input_images: dict[str, str], params: dict[str, Any]) -> dict[str, Any]:
    """Mosaic chips, blend road probabilities, and trace centerlines."""
    from .centerlines import probability_ridges, trace

    threshold = float(params.get("confidence_threshold", 0.08))
    if not 0 <= threshold <= 1:
        raise ValueError("confidence_threshold must be between zero and one")
    rgb, valid, grid, crs = _mosaic(chip_paths(input_images["image"]))
    probability = _blend_probabilities(session, rgb, valid)
    del rgb
    road = probability_ridges(probability, valid, threshold)
    return {
        "type": "FeatureCollection",
        "features": trace(road, grid, crs, probability),
    }
