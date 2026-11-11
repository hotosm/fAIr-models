"""Toy chips, labels and models with the U-Net's input and output contract."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds

CHIP_PIXELS = 256
STEP_DEG = 0.001
WEST, SOUTH = 7.432, 10.469
EAST, NORTH = WEST + STEP_DEG, SOUTH + STEP_DEG
CHIPS_PER_SIDE = 2
# Web map tile numbers of the first toy chip, so chip names follow fAIr's OAM-<x>-<y>-<z>.tif pattern.
TILE_X, TILE_Y, ZOOM = 278600, 248600, 19


def write_chip(
    path: Path, pixels: int = CHIP_PIXELS, bounds: tuple[float, float, float, float] = (WEST, SOUTH, EAST, NORTH)
) -> None:
    """An RGB chip whose left half is blue (solar panel) and right half dark (background)."""
    west, south, east, north = bounds
    rgb = np.full((3, pixels, pixels), 32, dtype=np.uint8)
    rgb[2, :, : pixels // 2] = 224
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=pixels,
        height=pixels,
        count=3,
        dtype="uint8",
        crs=CRS.from_epsg(4326),
        transform=from_bounds(west, south, east, north, pixels, pixels),
    ) as dst:
        dst.write(rgb)


def create_toy_data(root: Path) -> dict[str, Path]:
    """Four chips in a 2 x 2 grid, each blue on its left half, with panel labels over the blue halves."""
    chips_dir = root / "chips"
    chips_dir.mkdir(parents=True)
    panels = []
    for row in range(CHIPS_PER_SIDE):
        for col in range(CHIPS_PER_SIDE):
            west, north = WEST + col * STEP_DEG, NORTH - row * STEP_DEG
            south, mid = north - STEP_DEG, west + STEP_DEG / 2
            write_chip(
                chips_dir / f"OAM-{TILE_X + col}-{TILE_Y + row}-{ZOOM}.tif",
                bounds=(west, south, west + STEP_DEG, north),
            )
            panels.append(
                {
                    "type": "Feature",
                    "properties": {"class": "solar_panel"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[west, south], [mid, south], [mid, north], [west, north], [west, south]]],
                    },
                }
            )

    labels_dir = root / "labels"
    labels_dir.mkdir()
    (labels_dir / "labels.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": panels}))

    stac_path = root / "dataset-stac-item.json"
    stac_path.write_text(json.dumps(_build_dataset_stac_item(chips_dir, labels_dir), indent=2))
    return {"chips": chips_dir, "labels": labels_dir, "dataset_stac_item": stac_path}


@pytest.fixture(scope="session")
def generate_toy_dataset(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    return create_toy_data(tmp_path_factory.mktemp("toy_unet_solar_pv"))


@pytest.fixture(scope="session")
def base_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A randomly initialised state_dict in the base checkpoint's format, so the tests need no download."""
    import torch

    from models.unet_solar_pv.pipeline import _build_model

    path = tmp_path_factory.mktemp("checkpoint") / "base.pt"
    torch.manual_seed(0)
    torch.save(_build_model().state_dict(), path)
    return path


def _build_dataset_stac_item(chips_dir: Path, labels_dir: Path) -> dict[str, Any]:
    east, south = WEST + CHIPS_PER_SIDE * STEP_DEG, NORTH - CHIPS_PER_SIDE * STEP_DEG
    return {
        "type": "Feature",
        "stac_version": "1.1.0",
        "stac_extensions": ["https://stac-extensions.github.io/label/v1.0.1/schema.json"],
        "id": "toy-unet-solar-pv",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[WEST, south], [east, south], [east, NORTH], [WEST, NORTH], [WEST, south]]],
        },
        "bbox": [WEST, south, east, NORTH],
        "properties": {
            "datetime": "2026-11-02T00:00:00Z",
            "description": "Toy solar panel segmentation dataset",
            "label:type": "vector",
            "label:tasks": ["segmentation"],
            "label:classes": [{"name": "solar_panel", "classes": ["yes"]}],
            "label:description": "Solar panel outlines",
            "keywords": ["solar"],
            "fair:user_id": "test",
            "version": "1",
            "deprecated": False,
            "license": "CC-BY-4.0",
            "providers": [{"name": "HOTOSM", "roles": ["producer"], "url": "https://www.hotosm.org"}],
        },
        "assets": {
            "chips-image": {"href": str(chips_dir), "type": "image/tiff", "roles": ["data"]},
            "labels": {"href": str(labels_dir), "type": "application/geo+json", "roles": ["labels"]},
        },
        "links": [],
    }


@pytest.fixture
def panel_chips(tmp_path: Path) -> Path:
    chips = tmp_path / "chips"
    chips.mkdir()
    write_chip(chips / "OAM-0-0.tif")
    return chips


@pytest.fixture
def toy_model(tmp_path: Path) -> Path:
    """(N, 3, 256, 256) to (N, 1, 256, 256) logits where the panel logit is 4 * blue - 2."""
    import onnx
    from onnx import TensorProto, helper

    graph = helper.make_graph(
        [
            helper.make_node("Slice", ["image", "start", "end", "axis"], ["blue"]),
            helper.make_node("Mul", ["blue", "scale"], ["scaled"]),
            helper.make_node("Sub", ["scaled", "offset"], ["logits"]),
        ],
        "toy_solar_unet",
        [helper.make_tensor_value_info("image", TensorProto.FLOAT, ["batch", 3, CHIP_PIXELS, CHIP_PIXELS])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", 1, CHIP_PIXELS, CHIP_PIXELS])],
        [
            helper.make_tensor("start", TensorProto.INT64, [1], [2]),
            helper.make_tensor("end", TensorProto.INT64, [1], [3]),
            helper.make_tensor("axis", TensorProto.INT64, [1], [1]),
            helper.make_tensor("scale", TensorProto.FLOAT, [], [4.0]),
            helper.make_tensor("offset", TensorProto.FLOAT, [], [2.0]),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=10)
    onnx.checker.check_model(model)
    path = tmp_path / "toy.onnx"
    onnx.save(model, path)
    return path
