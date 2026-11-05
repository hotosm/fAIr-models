"""Serve-path tests: label burning, mask post-processing and end-to-end polygon predictions."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import rasterio

from .conftest import CHIP_PIXELS, ROAD_HALF_WIDTH_PX
from .test_steps import _split, _train


def _session(dataset: dict[str, Any]) -> Any:
    from onnxruntime import InferenceSession

    from models.unet_roads.pipeline import export_onnx

    onnx_bytes = export_onnx.entrypoint(_train(dataset, _split(dataset)))
    return InferenceSession(onnx_bytes, providers=["CPUExecutionProvider"])


def test_postprocess_returns_binary_mask() -> None:
    from models.unet_roads.pipeline import postprocess

    logits = np.array([[[-4.0, 4.0], [0.5, -0.5]]])
    assert postprocess(logits, 0.5).tolist() == [[0, 1], [1, 0]]


def test_road_mask_burns_polygons_as_drawn(generate_toy_dataset: dict[str, Any]) -> None:
    from models.unet_roads.pipeline import MODEL_INPUT_SIZE, _label_geometries, _road_mask

    roads = _label_geometries(str(generate_toy_dataset["labels"]))
    chip = sorted(Path(generate_toy_dataset["chips"]).glob("*.tif"))[0]
    with rasterio.open(chip) as src:
        # A large road width must not widen polygon labels.
        mask = _road_mask(src, roads, road_width_m=30.0)

    road_rows = np.flatnonzero(mask.any(axis=1))
    expected_rows = 2 * ROAD_HALF_WIDTH_PX * MODEL_INPUT_SIZE // CHIP_PIXELS
    assert abs(len(road_rows) - expected_rows) <= 2
    assert mask[road_rows].all()


def test_road_mask_buffers_line_labels(generate_toy_dataset: dict[str, Any], tmp_path: Path) -> None:
    from models.unet_roads.pipeline import _label_geometries, _road_mask

    chip = sorted(Path(generate_toy_dataset["chips"]).glob("*.tif"))[0]
    with rasterio.open(chip) as src:
        west, south, east, north = src.bounds
        centre = (south + north) / 2
        line = {
            "type": "Feature",
            "properties": {},
            "geometry": {"type": "LineString", "coordinates": [[west, centre], [east, centre]]},
        }
        (tmp_path / "labels.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": [line]}))
        roads = _label_geometries(str(tmp_path))
        narrow = _road_mask(src, roads, road_width_m=2.0)
        wide = _road_mask(src, roads, road_width_m=8.0)

    assert 0 < narrow.sum() < wide.sum()


def test_predict_returns_road_polygons(generate_toy_dataset: dict[str, Any]) -> None:
    from models.unet_roads.pipeline import predict

    params = {"confidence_threshold": 0.0, "min_area_m2": 0.0}
    result = predict(_session(generate_toy_dataset), {"image": str(generate_toy_dataset["chips"])}, params)

    assert result["type"] == "FeatureCollection"
    assert result["features"]
    assert {f["geometry"]["type"] for f in result["features"]} <= {"Polygon", "MultiPolygon"}
    assert all(f["properties"] == {"class": 1, "label": "road"} for f in result["features"])


def test_predict_merges_chips_into_one_polygon(generate_toy_dataset: dict[str, Any]) -> None:
    from models.unet_roads.pipeline import predict

    # At threshold zero every pixel is road: the four adjacent chips must come out as a single polygon.
    params = {"confidence_threshold": 0.0, "min_area_m2": 0.0}
    result = predict(_session(generate_toy_dataset), {"image": str(generate_toy_dataset["chips"])}, params)

    assert len(result["features"]) == 1


def test_predict_drops_polygons_below_min_area(generate_toy_dataset: dict[str, Any]) -> None:
    from models.unet_roads.pipeline import predict

    params = {"confidence_threshold": 0.0, "min_area_m2": 1e9}
    result = predict(_session(generate_toy_dataset), {"image": str(generate_toy_dataset["chips"])}, params)

    assert result["features"] == []
