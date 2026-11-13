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
_ROOF_CLASSES = [
    {"name": "roof:material", "classes": ["metal_sheet"]},
    {"name": "roof:material", "classes": ["thatch"]},
    {"name": "roof:material", "classes": ["asbestos", "concrete"]},
]


def _matches_join_or(tags: dict[str, str], join_or: dict[str, list[str]]) -> bool:
    """Raw-data API `join_or`: any listed key carrying one of its values, an empty list meaning any value."""
    return any(key in tags and (not values or tags[key] in values) for key, values in join_or.items())


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


def test_osm_filters_merges_classes_that_share_a_key() -> None:
    assert _osm_filters(_ROOF_CLASSES, "polygon") == {
        "tags": {"polygon": {"join_or": {"roof:material": ["metal_sheet", "thatch", "asbestos", "concrete"]}}}
    }
    repeated = [{"name": "building", "classes": ["house"]}, {"name": "building", "classes": ["house", "hut"]}]
    assert _osm_filters(repeated, "polygon")["tags"]["polygon"]["join_or"] == {"building": ["house", "hut"]}


@pytest.mark.parametrize(("first", "second"), [(["*"], ["house"]), (["house"], ["*"])])
def test_osm_filters_wildcard_on_a_shared_key_asks_for_any_value(first, second) -> None:
    classes = [{"name": "building", "classes": first}, {"name": "building", "classes": second}]
    assert _osm_filters(classes, "polygon") == {"tags": {"polygon": {"join_or": {"building": []}}}}


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


def test_materialize_dataset_keeps_every_class_that_shares_a_key(tmp_path, monkeypatch) -> None:
    materials = ["metal_sheet", "thatch", "asbestos", "roof_tiles"]
    roofs = [{"properties": {"tags": {"roof:material": material}}} for material in materials]

    async def fake_download_tiles(**kwargs):
        return str(tmp_path / "out" / "chips")

    async def fake_download_osm(*, filters, **kwargs):
        join_or = filters["tags"]["polygon"]["join_or"]
        return {"features": [feat for feat in roofs if _matches_join_or(feat["properties"]["tags"], join_or)]}

    fake_tms = SimpleNamespace(download_tiles=fake_download_tiles)
    fake_osm = SimpleNamespace(download_osm_data=fake_download_osm)
    monkeypatch.setattr("fair.datasets._require_geomltoolkits", lambda: (fake_tms, fake_osm))

    materialize_dataset(
        tms_url="https://tiles/{z}/{x}/{y}",
        aoi={"type": "Polygon", "coordinates": _SQUARE},
        zoom=19,
        out_dir=str(tmp_path / "out"),
        label_classes=_ROOF_CLASSES,
        geometry_type="polygon",
        osm_api_url="https://osm",
    )

    labels = json.loads((tmp_path / "out" / "labels" / "labels.geojson").read_text())
    assert [feat["properties"]["label"] for feat in labels["features"]] == [1, 2, 3]
