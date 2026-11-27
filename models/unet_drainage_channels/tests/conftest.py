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
WEST, SOUTH = 39.27, -6.801
EAST, NORTH = WEST + STEP_DEG, SOUTH + STEP_DEG
CHIPS_PER_SIDE = 2
# Web map tile numbers of the first toy chip, so chip names follow fAIr's OAM-<x>-<y>-<z>.tif pattern.
TILE_X, TILE_Y, ZOOM = 638670, 544219, 20
# Each chip is dark ground with a bright drain running north to south, and no data (black) on its east
# side, as at the edge of a drone image.
GROUND, DRAIN = (60, 50, 32), (40, 160, 224)
DRAIN_COLUMNS = (96, 112)
NO_DATA_FROM = 192
# A toy pixel is about 0.43 m on the ground, so the drain is about 7 m wide.
DRAIN_WIDTH_M = 7.0


def write_chip(
    path: Path, pixels: int = CHIP_PIXELS, bounds: tuple[float, float, float, float] = (WEST, SOUTH, EAST, NORTH)
) -> None:
    """An RGB chip of ground with a drain down columns DRAIN_COLUMNS and no data from column NO_DATA_FROM."""
    west, south, east, north = bounds
    rgb = np.zeros((3, pixels, pixels), dtype=np.uint8)
    rgb[:, :, :NO_DATA_FROM] = np.array(GROUND, np.uint8)[:, None, None]
    rgb[:, :, DRAIN_COLUMNS[0] : DRAIN_COLUMNS[1]] = np.array(DRAIN, np.uint8)[:, None, None]
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


def drain_longitude(west: float = WEST) -> float:
    """Longitude of the middle of the drain in a chip whose west edge is at `west`."""
    return west + sum(DRAIN_COLUMNS) / 2 / CHIP_PIXELS * STEP_DEG


def _line(coordinates: list[list[float]], properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "Feature",
        "properties": properties,
        "geometry": {"type": "LineString", "coordinates": coordinates},
    }


def create_toy_data(root: Path) -> dict[str, Path]:
    """Four chips in a 2 x 2 grid, each with a drain line down its drain.

    The labels carry their tags the two ways fAIr can hand them over: as plain properties and under
    `tags`, as downloaded from OpenStreetMap. Each chip also has a culvert crossing its west side, which
    the model must ignore, and the whole area is crossed by a stream, which is not a drain.
    """
    chips_dir = root / "chips"
    chips_dir.mkdir(parents=True)
    drains = []
    for row in range(CHIPS_PER_SIDE):
        for col in range(CHIPS_PER_SIDE):
            west, north = WEST + col * STEP_DEG, NORTH - row * STEP_DEG
            south = north - STEP_DEG
            write_chip(
                chips_dir / f"OAM-{TILE_X + col}-{TILE_Y + row}-{ZOOM}.tif",
                bounds=(west, south, west + STEP_DEG, north),
            )
            middle = drain_longitude(west)
            tags = {"waterway": "drain", "width": str(DRAIN_WIDTH_M)}
            drains += [
                _line([[middle, south], [middle, north]], tags if col else {"label": 1, "tags": tags}),
                _line(
                    [[west, (south + north) / 2], [west + 0.2 * STEP_DEG, (south + north) / 2]],
                    {"tags": {"waterway": "drain", "tunnel": "culvert", "width": "2"}},
                ),
            ]
    drains.append(
        _line(
            [[WEST, NORTH - 0.1 * STEP_DEG], [WEST + CHIPS_PER_SIDE * STEP_DEG, NORTH - 0.1 * STEP_DEG]],
            {"tags": {"waterway": "stream"}},
        )
    )

    labels_dir = root / "labels"
    labels_dir.mkdir()
    (labels_dir / "labels.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": drains}))

    stac_path = root / "dataset-stac-item.json"
    stac_path.write_text(json.dumps(_build_dataset_stac_item(chips_dir, labels_dir), indent=2))
    return {"chips": chips_dir, "labels": labels_dir, "dataset_stac_item": stac_path}


@pytest.fixture(scope="session")
def generate_toy_dataset(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    return create_toy_data(tmp_path_factory.mktemp("toy_unet_drainage_channels"))


@pytest.fixture(scope="session")
def base_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A randomly initialised state_dict in the base checkpoint's format, so the tests need no download."""
    import torch

    from models.unet_drainage_channels.pipeline import _build_model

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
        "id": "toy-unet-drainage-channels",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[WEST, south], [east, south], [east, NORTH], [WEST, NORTH], [WEST, south]]],
        },
        "bbox": [WEST, south, east, NORTH],
        "properties": {
            "datetime": "2026-11-11T00:00:00Z",
            "description": "Toy drainage channel segmentation dataset",
            "label:type": "vector",
            "label:tasks": ["segmentation"],
            "label:classes": [{"name": "waterway", "classes": ["drain", "ditch"]}],
            "label:description": "Centrelines of open drains and ditches",
            "keywords": ["waterway", "line"],
            "fair:geometry_type": "line",
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
def drain_chips(tmp_path: Path) -> Path:
    chips = tmp_path / "chips"
    chips.mkdir()
    write_chip(chips / "OAM-0-0.tif")
    return chips


@pytest.fixture
def toy_model(tmp_path: Path) -> Path:
    """(N, 3, 256, 256) to (N, 1, 256, 256) logits: 8 * blue - 4, about 3 on the toy drain and below -2.9 off it."""
    import onnx
    from onnx import TensorProto, helper

    graph = helper.make_graph(
        [
            helper.make_node("Slice", ["image", "start", "end", "axis"], ["blue"]),
            helper.make_node("Mul", ["blue", "scale"], ["scaled"]),
            helper.make_node("Sub", ["scaled", "offset"], ["logits"]),
        ],
        "toy_drain_unet",
        [helper.make_tensor_value_info("image", TensorProto.FLOAT, ["batch", 3, CHIP_PIXELS, CHIP_PIXELS])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", 1, CHIP_PIXELS, CHIP_PIXELS])],
        [
            helper.make_tensor("scale", TensorProto.FLOAT, [], [8.0]),
            helper.make_tensor("offset", TensorProto.FLOAT, [], [4.0]),
            helper.make_tensor("start", TensorProto.INT64, [1], [2]),
            helper.make_tensor("end", TensorProto.INT64, [1], [3]),
            helper.make_tensor("axis", TensorProto.INT64, [1], [1]),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=10)
    onnx.checker.check_model(model)
    path = tmp_path / "toy.onnx"
    onnx.save(model, path)
    return path
