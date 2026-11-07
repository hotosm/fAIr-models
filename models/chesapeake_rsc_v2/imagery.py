"""Read fAIr chips at approximately the model's native ground resolution."""

from pathlib import Path

import numpy as np
import rasterio
from rasterio.coords import BoundingBox
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import transform

from fair.utils.data import resolve_directory, resolve_path

GSD = 0.6


def chip_paths(href: str) -> list[Path]:
    path = resolve_path(href) if href.lower().endswith((".tif", ".tiff")) else resolve_directory(href, "*.tif*")
    paths = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.suffix.lower() in {".tif", ".tiff"})
    if not paths:
        raise FileNotFoundError(f"No GeoTIFF chips at {href}")
    return paths


def local_metric_crs(bounds: BoundingBox, crs: CRS) -> CRS:
    """Center a metric projection on the imagery bounds."""
    west, south, east, north = bounds
    lon, lat = transform(crs, "EPSG:4326", [(west + east) / 2], [(south + north) / 2])
    return CRS.from_dict(proj="aeqd", lon_0=lon[0], lat_0=lat[0], datum="WGS84", units="m")


def target_shape(bounds: BoundingBox, crs: CRS) -> tuple[int, int]:
    """Size an axis-aligned grid at approximately 0.6 m at its center."""
    west, south, east, north = bounds
    x, y = (west + east) / 2, (south + north) / 2
    xs, ys = transform(crs, local_metric_crs(bounds, crs), [west, east, x, x], [y, y, south, north])
    width_m = np.hypot(xs[1] - xs[0], ys[1] - ys[0])
    height_m = np.hypot(xs[3] - xs[2], ys[3] - ys[2])
    return max(1, round(height_m / GSD)), max(1, round(width_m / GSD))


def validate_rgb(src: rasterio.DatasetReader) -> None:
    if src.crs is None or src.count < 3 or set(src.dtypes[:3]) != {"uint8"}:
        raise ValueError(f"Expected a georeferenced uint8 RGB chip: {src.name}")


def read_chip(src: rasterio.DatasetReader) -> tuple[np.ndarray, np.ndarray, Affine]:
    """Return normalized RGB, valid pixels, and the transform for the resized grid."""
    validate_rgb(src)
    height, width = target_shape(src.bounds, src.crs)
    if max(height, width) > 2048:
        raise ValueError(f"Chip exceeds 2048 pixels at {GSD} m GSD. Request smaller tiles or a higher zoom.")
    transform = src.transform * Affine.scale(src.width / width, src.height / height)
    rgb = src.read([1, 2, 3], out_shape=(3, height, width), resampling=Resampling.bilinear)
    valid = (src.read_masks([1, 2, 3], out_shape=(3, height, width)) != 0).all(axis=0)
    return rgb.astype(np.float32) / 255.0, valid, transform


def padded_size(length: int) -> int:
    return max(64, (length + 31) // 32 * 32)
