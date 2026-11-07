"""Trace a skeleton on the combined chip grid so roads can cross seams."""

from collections import Counter
from collections.abc import Iterator

import numpy as np
from rasterio.crs import CRS
from rasterio.transform import Affine
from rasterio.warp import transform_geom
from scipy import ndimage as ndi
from shapely import LineString, MultiLineString, line_merge
from skimage.morphology import skeletonize

Pixel = tuple[int, int]


def probability_ridges(probability: np.ndarray, valid: np.ndarray, minimum: float) -> np.ndarray:
    """Find cross-road probability peaks with bounded scratch space."""
    ridges = np.empty(valid.shape, dtype=bool)
    height, width = valid.shape
    radius = 6
    # Gaussian support, one interpolation pixel, and two pixels for closing.
    halo = radius + 3
    for start in range(0, height, 256):
        stop = min(start + 256, height)
        first, last = max(0, start - halo), min(height, stop + halo)
        patch = probability[first:last]
        smoothed = ndi.gaussian_filter(patch, 1.4, radius=radius)
        normal_angle = ndi.gaussian_filter(patch, 1.4, order=(2, 0), radius=radius)
        scratch = ndi.gaussian_filter(patch, 1.4, order=(0, 2), radius=radius)
        normal_angle -= scratch
        ndi.gaussian_filter(patch, 1.4, order=(1, 1), radius=radius, output=scratch)
        scratch *= 2
        np.arctan2(scratch, normal_angle, out=normal_angle)
        normal_angle *= 0.5

        rows = np.arange(last - first, dtype=np.float32)[:, None]
        cols = np.arange(width, dtype=np.float32)[None, :]
        coordinates = np.empty((2, *patch.shape), dtype=np.float32)
        np.sin(normal_angle, out=coordinates[0])
        np.negative(coordinates[0], out=coordinates[0])
        np.cos(normal_angle, out=coordinates[1])
        coordinates[0] += rows
        coordinates[1] += cols
        ndi.map_coordinates(smoothed, coordinates, order=1, mode="nearest", output=scratch)
        peaks = smoothed >= scratch
        np.subtract(2 * rows, coordinates[0], out=coordinates[0])
        np.subtract(2 * cols, coordinates[1], out=coordinates[1])
        ndi.map_coordinates(smoothed, coordinates, order=1, mode="nearest", output=scratch)
        peaks &= smoothed >= scratch
        peaks &= smoothed >= minimum
        peaks &= valid[first:last]
        closed = ndi.binary_closing(peaks, structure=np.ones((3, 3), dtype=bool))
        np.logical_and(closed[start - first : stop - first], valid[start:stop], out=ridges[start:stop])
    return ridges


def _edges(skeleton: np.ndarray) -> Iterator[tuple[Pixel, Pixel]]:
    rows, cols = skeleton.shape
    for row, col in zip(*skeleton.nonzero(), strict=True):
        row, col = int(row), int(col)
        for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
            r, c = row + dr, col + dc
            if r >= rows or not 0 <= c < cols or not skeleton[r, c]:
                continue
            # At corners keep the orthogonal route and avoid triangles.
            if dr and dc and (skeleton[row, c] or skeleton[r, col]):
                continue
            yield (col, row), (c, r)


def trace(mask: np.ndarray, grid: Affine, crs: CRS) -> list[dict]:
    """Return WGS84 road features, pruning short terminal spurs at junctions."""
    edges = list(_edges(skeletonize(mask)))
    if not edges:
        return []
    merged = line_merge(MultiLineString(edges))
    lines = [merged] if isinstance(merged, LineString) else list(merged.geoms)
    degree = Counter(point for line in lines for point in (line.coords[0], line.coords[-1]))
    features = []
    for line in lines:
        start, end = line.coords[0], line.coords[-1]
        degrees = sorted((degree[start], degree[end]))
        if line.length < 4 and degrees[0] == 1 and degrees[1] >= 3:
            continue
        coords = [grid * (x + 0.5, y + 0.5) for x, y in line.simplify(0.5).coords]
        features.append(
            {
                "type": "Feature",
                "geometry": transform_geom(crs, "EPSG:4326", {"type": "LineString", "coordinates": coords}),
                "properties": {"class": 1, "label": "road"},
            }
        )
    return features
