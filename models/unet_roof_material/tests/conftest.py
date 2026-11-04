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
WEST, SOUTH = 40.685, -14.525
EAST, NORTH = WEST + STEP_DEG, SOUTH + STEP_DEG
CHIPS_PER_SIDE = 2
# Web map tile numbers of the first toy chip, so chip names follow fAIr's OAM-<x>-<y>-<z>.tif pattern.
TILE_X, TILE_Y, ZOOM = 642784, 567059, 20
# Each chip is four vertical stripes: metal sheet (blue), thatch (green), other roof (red), then dark ground.
STRIPES = [(32, 32, 224), (32, 224, 32), (224, 32, 32), (32, 32, 32)]
STRIPE = CHIP_PIXELS // len(STRIPES)


def write_chip(
    path: Path, pixels: int = CHIP_PIXELS, bounds: tuple[float, float, float, float] = (WEST, SOUTH, EAST, NORTH)
) -> None:
    """An RGB chip of four equal vertical stripes in the STRIPES colours."""
    west, south, east, north = bounds
    rgb = np.zeros((3, pixels, pixels), dtype=np.uint8)
    width = pixels // len(STRIPES)
    for i, colour in enumerate(STRIPES):
        rgb[:, :, i * width : (i + 1) * width] = np.array(colour, np.uint8)[:, None, None]
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


def _polygon(west: float, south: float, east: float, north: float, properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "Feature",
        "properties": properties,
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[west, south], [east, south], [east, north], [west, north], [west, south]]],
        },
    }


def create_toy_data(root: Path) -> dict[str, Path]:
    """Four chips in a 2 x 2 grid with a roof label over each roof stripe.

    The labels cover the three ways a roof can carry its class: an integer `label`, a roof:material
    property and a roof:material tag under `tags` as fAIr downloads it from OpenStreetMap. The top half
    of each dark stripe is a roof of a material the model does not know (roof_tiles).
    """
    chips_dir = root / "chips"
    chips_dir.mkdir(parents=True)
    roofs = []
    stripe_deg = STEP_DEG / len(STRIPES)
    for row in range(CHIPS_PER_SIDE):
        for col in range(CHIPS_PER_SIDE):
            west, north = WEST + col * STEP_DEG, NORTH - row * STEP_DEG
            south = north - STEP_DEG
            write_chip(
                chips_dir / f"OAM-{TILE_X + col}-{TILE_Y + row}-{ZOOM}.tif",
                bounds=(west, south, west + STEP_DEG, north),
            )
            edges = [west + i * stripe_deg for i in range(len(STRIPES) + 1)]
            roofs += [
                _polygon(edges[0], south, edges[1], north, {"label": 1}),
                _polygon(edges[1], south, edges[2], north, {"roof:material": "thatch"}),
                _polygon(edges[2], south, edges[3], north, {"label": 3, "tags": {"roof:material": "asbestos"}}),
                _polygon(edges[3], (south + north) / 2, edges[4], north, {"tags": {"roof:material": "roof_tiles"}}),
            ]

    labels_dir = root / "labels"
    labels_dir.mkdir()
    (labels_dir / "labels.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": roofs}))

    stac_path = root / "dataset-stac-item.json"
    stac_path.write_text(json.dumps(_build_dataset_stac_item(chips_dir, labels_dir), indent=2))
    return {"chips": chips_dir, "labels": labels_dir, "dataset_stac_item": stac_path}


@pytest.fixture(scope="session")
def generate_toy_dataset(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    return create_toy_data(tmp_path_factory.mktemp("toy_unet_roof_material"))


@pytest.fixture(scope="session")
def base_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A randomly initialised state_dict in the base checkpoint's format, so the tests need no download."""
    import torch

    from models.unet_roof_material.pipeline import _build_model

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
        "id": "toy-unet-roof-material",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[WEST, south], [east, south], [east, NORTH], [WEST, NORTH], [WEST, south]]],
        },
        "bbox": [WEST, south, east, NORTH],
        "properties": {
            "datetime": "2026-11-04T00:00:00Z",
            "description": "Toy roof material segmentation dataset",
            "label:type": "vector",
            "label:tasks": ["segmentation"],
            "label:classes": [
                {"name": "roof:material", "classes": ["metal_sheet"]},
                {"name": "roof:material", "classes": ["thatch"]},
                {"name": "roof:material", "classes": ["asbestos", "concrete"]},
            ],
            "label:description": "Roof outlines with their material",
            "keywords": ["building"],
            "fair:user_id": "test",
            "version": "1",
            "deprecated": False,
            "license": "ODbL-1.0",
            "providers": [{"name": "HOTOSM", "roles": ["producer"], "url": "https://www.hotosm.org"}],
        },
        "assets": {
            "chips-image": {"href": str(chips_dir), "type": "image/tiff", "roles": ["data"]},
            "labels": {"href": str(labels_dir), "type": "application/geo+json", "roles": ["labels"]},
        },
        "links": [],
    }


@pytest.fixture
def roof_chips(tmp_path: Path) -> Path:
    chips = tmp_path / "chips"
    chips.mkdir()
    write_chip(chips / "OAM-0-0.tif")
    return chips


@pytest.fixture
def toy_model(tmp_path: Path) -> Path:
    """(N, 3, 256, 256) to (N, 4, 256, 256) logits: background 0, metal sheet 4 * blue - 2, thatch
    4 * green - 2 and other roof 4 * red - 2."""
    import onnx
    from onnx import TensorProto, helper

    graph = helper.make_graph(
        [
            helper.make_node("Mul", ["image", "scale"], ["scaled"]),
            helper.make_node("Sub", ["scaled", "offset"], ["rgb_logits"]),
            # Channels red, green, blue become other roof, thatch, metal sheet: reverse them.
            helper.make_node("Gather", ["rgb_logits", "blue_green_red"], ["roof_logits"], axis=1),
            helper.make_node("Slice", ["image", "start", "end", "axis"], ["red"]),
            helper.make_node("Mul", ["red", "zero"], ["background_logit"]),
            helper.make_node("Concat", ["background_logit", "roof_logits"], ["logits"], axis=1),
        ],
        "toy_roof_unet",
        [helper.make_tensor_value_info("image", TensorProto.FLOAT, ["batch", 3, CHIP_PIXELS, CHIP_PIXELS])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", 4, CHIP_PIXELS, CHIP_PIXELS])],
        [
            helper.make_tensor("scale", TensorProto.FLOAT, [], [4.0]),
            helper.make_tensor("offset", TensorProto.FLOAT, [], [2.0]),
            helper.make_tensor("blue_green_red", TensorProto.INT64, [3], [2, 1, 0]),
            helper.make_tensor("start", TensorProto.INT64, [1], [0]),
            helper.make_tensor("end", TensorProto.INT64, [1], [1]),
            helper.make_tensor("axis", TensorProto.INT64, [1], [1]),
            helper.make_tensor("zero", TensorProto.FLOAT, [], [0.0]),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=10)
    onnx.checker.check_model(model)
    path = tmp_path / "toy.onnx"
    onnx.save(model, path)
    return path
