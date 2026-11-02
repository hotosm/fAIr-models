"""Toy chips and a toy ONNX model with the U-Net's input and output contract."""

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds

CHIP_PIXELS = 256
STEP_DEG = 0.001
WEST, SOUTH = 7.432, 10.469
EAST, NORTH = WEST + STEP_DEG, SOUTH + STEP_DEG


def write_chip(path: Path, pixels: int = CHIP_PIXELS) -> None:
    """An RGB chip whose left half is blue (solar panel) and right half dark (background)."""
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
        transform=from_bounds(WEST, SOUTH, EAST, NORTH, pixels, pixels),
    ) as dst:
        dst.write(rgb)


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
