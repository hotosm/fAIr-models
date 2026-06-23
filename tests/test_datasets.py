import json
from types import SimpleNamespace

import pytest

from fair.datasets import (
    _aoi_geometry_and_bbox,
    _osm_filters,
    _require_geomltoolkits,
    _stamp_class_label,
    materialize_dataset,
)

_SQUARE = [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]]


def test_aoi_geometry_and_bbox_from_feature_collection() -> None:
    aoi = {
        "type": "FeatureCollection",
        "features": [{"type": "Feature", "geometry": {"type": "Polygon", "coordinates": _SQUARE}}],
    }
    geometry, bbox = _aoi_geometry_and_bbox(aoi)
    assert geometry == {"type": "MultiPolygon", "coordinates": [_SQUARE]}
    assert bbox == [0.0, 0.0, 1.0, 1.0]


def test_aoi_geometry_and_bbox_accepts_bare_geometry() -> None:
    geometry, bbox = _aoi_geometry_and_bbox({"type": "Polygon", "coordinates": _SQUARE})
    assert geometry["type"] == "MultiPolygon"
    assert bbox == [0.0, 0.0, 1.0, 1.0]


def test_aoi_geometry_rejects_non_polygon() -> None:
    with pytest.raises(ValueError, match="Polygon or MultiPolygon"):
        _aoi_geometry_and_bbox({"type": "Point", "coordinates": [0.0, 0.0]})


def test_aoi_geometry_rejects_empty() -> None:
    with pytest.raises(ValueError, match="no geometry"):
        _aoi_geometry_and_bbox({"type": "FeatureCollection", "features": []})


def test_osm_filters_maps_wildcard_and_values() -> None:
    classes = [{"name": "building", "classes": ["*"]}, {"name": "highway", "classes": ["residential", "primary"]}]
    assert _osm_filters(classes, "polygon") == {
        "tags": {"polygon": {"join_or": {"building": [], "highway": ["residential", "primary"]}}}
    }


def test_stamp_class_label_assigns_first_match() -> None:
    classes = [{"name": "building", "classes": ["*"]}, {"name": "highway", "classes": ["residential"]}]
    feat = {"properties": {"tags": {"highway": "residential"}}}
    assert _stamp_class_label(feat, classes) == 2
    assert feat["properties"]["label"] == 2


def test_stamp_class_label_returns_none_when_unmatched() -> None:
    feat = {"properties": {"tags": {"natural": "water"}}}
    assert _stamp_class_label(feat, [{"name": "building", "classes": ["*"]}]) is None


def test_require_geomltoolkits_fails_loud_without_extra() -> None:
    with pytest.raises(RuntimeError, match="serve"):
        _require_geomltoolkits()


def test_materialize_dataset_downloads_and_stamps(tmp_path, monkeypatch) -> None:
    chips_dir = tmp_path / "out" / "chips"

    async def fake_download_tiles(**kwargs):
        return str(chips_dir)

    async def fake_download_osm(**kwargs):
        return {
            "type": "FeatureCollection",
            "features": [
                {"properties": {"tags": {"building": "yes"}}},
                {"properties": {"tags": {"highway": "residential"}}},
            ],
        }

    fake_tms = SimpleNamespace(download_tiles=fake_download_tiles)
    fake_osm = SimpleNamespace(download_osm_data=fake_download_osm)
    monkeypatch.setattr("fair.datasets._require_geomltoolkits", lambda: (fake_tms, fake_osm))

    aoi = {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": _SQUARE}}
    chips_href, labels_href, geometry, bbox = materialize_dataset(
        tms_url="https://tiles/{z}/{x}/{y}",
        aoi=aoi,
        zoom=19,
        out_dir=str(tmp_path / "out"),
        label_classes=[{"name": "building", "classes": ["*"]}],
        geometry_type="polygon",
        osm_api_url="https://osm",
    )

    assert chips_href == str(chips_dir)
    labels = json.loads((tmp_path / "out" / "labels" / "labels.geojson").read_text())
    assert labels_href.endswith("labels.geojson")
    assert len(labels["features"]) == 1
    assert labels["features"][0]["properties"]["label"] == 1
    assert geometry["type"] == "MultiPolygon"
    assert bbox == [0.0, 0.0, 1.0, 1.0]
