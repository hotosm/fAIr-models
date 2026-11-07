"""Spatial holdout, road-label rasterization, and checkpoint helpers."""

import io
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import segmentation_models_pytorch as smp
import torch
from pyproj import Transformer
from rasterio.features import rasterize
from shapely.geometry import shape
from shapely.ops import transform as transform_geometry
from torch.nn.functional import pad
from torch.utils.data import Dataset

from fair.utils.data import resolve_directory, resolve_path

from .imagery import chip_paths, padded_size, read_chip

IGNORE = 255
TILE = re.compile(r"OAM-(\d+)-(\d+)-(\d+)\.(?:tif|tiff)", re.IGNORECASE)


def spatial_split(paths: list[Path], ratio: float, seed: int, block_size: int) -> dict[str, Any]:
    """Hold out complete spatial blocks on a single zoom grid."""
    blocks: dict[tuple[int, int], list[str]] = defaultdict(list)
    zooms = set()
    for path in paths:
        match = TILE.fullmatch(path.name)
        if match is None:
            raise ValueError(f"Expected OAM-x-y-z.tif for the spatial split: {path.name}")
        x, y, zoom = map(int, match.groups())
        zooms.add(zoom)
        blocks[x // block_size, y // block_size].append(path.name)
    if len(zooms) != 1 or len(blocks) < 2 or len({p.name for p in paths}) != len(paths):
        raise ValueError("Spatial holdout needs distinct chips at one zoom spanning at least two blocks")
    keys = sorted(blocks)
    random.Random(seed).shuffle(keys)
    val = []
    while len(keys) > 1 and len(val) < len(paths) * ratio:
        val.extend(blocks[keys.pop()])
    train = [name for key in keys for name in blocks[key]]
    return {
        "strategy": "spatial",
        "description": "Whole OAM tile blocks held out",
        "seed": seed,
        "block_size": block_size,
        "val_ratio": len(val) / len(paths),
        "train_count": len(train),
        "val_count": len(val),
        "train_chip_names": train,
        "val_chip_names": val,
    }


class RoadChips(Dataset):
    def __init__(self, href: str, labels: str, names: list[str], road_width_m: float) -> None:
        selected = set(names)
        self.paths = [p for p in chip_paths(href) if p.name in selected]
        if len(self.paths) != len(selected):
            raise ValueError("Chip inventory changed after the split")
        path = resolve_path(labels) if labels.endswith(".geojson") else resolve_directory(labels, "*.geojson")
        files = [path] if path.is_file() else sorted(path.rglob("*.geojson"))
        if not files:
            raise FileNotFoundError(f"No GeoJSON labels at {labels}")
        self.geometries = []
        self.road_width_m = road_width_m
        for file in files:
            for feature in json.loads(file.read_text())["features"]:
                if not feature.get("geometry"):
                    continue
                geometry = shape(feature["geometry"])
                if geometry.geom_type not in {"LineString", "MultiLineString", "Polygon", "MultiPolygon"}:
                    raise ValueError(f"Unsupported road label geometry: {geometry.geom_type}")
                if not geometry.is_empty:
                    self.geometries.append(geometry)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        with rasterio.open(self.paths[index]) as src:
            rgb, valid, transform = read_chip(src)
            to_lonlat = Transformer.from_crs(src.crs, 4326, always_xy=True)
            lon, lat = to_lonlat.transform(*(src.transform * (src.width / 2, src.height / 2)))
            # Buffer centerline labels by a physical road width on a local metric grid.
            utm = (32600 if lat >= 0 else 32700) + min(60, max(1, int((lon + 180) // 6) + 1))
            to_utm = Transformer.from_crs(4326, utm, always_xy=True)
            from_utm = Transformer.from_crs(utm, src.crs, always_xy=True)
            from_lonlat = Transformer.from_crs(4326, src.crs, always_xy=True)
            geoms = []
            for geom in self.geometries:
                if "LineString" in geom.geom_type:
                    buffered = transform_geometry(to_utm.transform, geom).buffer(self.road_width_m / 2)
                    geoms.append(transform_geometry(from_utm.transform, buffered))
                else:
                    geoms.append(transform_geometry(from_lonlat.transform, geom))
            target = (
                rasterize(geoms, out_shape=valid.shape, transform=transform, dtype="uint8")
                if geoms
                else np.zeros(valid.shape, dtype=np.uint8)
            )
        target[~valid] = IGNORE
        return torch.from_numpy(rgb), torch.from_numpy(target).long()


def collate(samples: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    height = padded_size(max(mask.shape[0] for _, mask in samples))
    width = padded_size(max(mask.shape[1] for _, mask in samples))
    images, masks = [], []
    for rgb, target in samples:
        h, w = target.shape
        padding = (0, width - w, 0, height - h)
        images.append(pad(rgb, padding))
        masks.append(pad(target, padding, value=IGNORE))
    return torch.stack(images), torch.stack(masks)


def load_model(checkpoint: Path | io.BytesIO) -> torch.nn.Module:
    model = smp.Unet(encoder_name="resnet18", encoder_weights=None, in_channels=3, classes=2)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    return model
