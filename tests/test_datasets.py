import json
import sys
from pathlib import Path
from types import SimpleNamespace

import mercantile
import pytest

from fair.datasets import (
    _aoi_geometry_and_bbox,
    _osm_filters,
    _require_geomltoolkits,
    _stamp_class_label,
    materialize_dataset,
    preview_sample,
    select_labels,
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


def test_require_geomltoolkits_fails_loud_without_extra(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "geomltoolkits", None)
    with pytest.raises(RuntimeError, match="serve"):
        _require_geomltoolkits()


def test_materialize_dataset_downloads_and_stamps(tmp_path, monkeypatch) -> None:
    async def fake_download_tiles(*, out, **kwargs):
        nested = Path(out) / "chips"
        nested.mkdir(parents=True)
        (nested / "OAM-1-2-19.tif").write_bytes(b"tif")
        return str(nested)

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
    chips_dirs, labels_href, geometry, bbox = materialize_dataset(
        tms_urls={"image": "https://tiles/{z}/{x}/{y}"},
        aoi=aoi,
        zoom=19,
        out_dir=str(tmp_path / "out"),
        label_classes=[{"name": "building", "classes": ["*"]}],
        geometry_type="polygon",
        osm_api_url="https://osm",
    )

    assert chips_dirs == {"image": str(tmp_path / "out" / "chips-image")}
    assert [chip.name for chip in (tmp_path / "out" / "chips-image").iterdir()] == ["OAM-1-2-19.tif"]
    labels = json.loads((tmp_path / "out" / "labels" / "labels.geojson").read_text())
    assert labels_href.endswith("labels.geojson")
    assert len(labels["features"]) == 1
    assert labels["features"][0]["properties"]["label"] == 1
    assert geometry["type"] == "MultiPolygon"
    assert bbox == [0.0, 0.0, 1.0, 1.0]


def test_materialize_dataset_keeps_every_class_that_shares_a_key(tmp_path, monkeypatch) -> None:
    materials = ["metal_sheet", "thatch", "asbestos", "roof_tiles"]
    roofs = [{"properties": {"tags": {"roof:material": material}}} for material in materials]

    async def fake_download_tiles(*, out, **kwargs):
        (Path(out) / "chips").mkdir(parents=True)
        return str(Path(out) / "chips")

    async def fake_download_osm(*, filters, **kwargs):
        join_or = filters["tags"]["polygon"]["join_or"]
        return {"features": [feat for feat in roofs if _matches_join_or(feat["properties"]["tags"], join_or)]}

    fake_tms = SimpleNamespace(download_tiles=fake_download_tiles)
    fake_osm = SimpleNamespace(download_osm_data=fake_download_osm)
    monkeypatch.setattr("fair.datasets._require_geomltoolkits", lambda: (fake_tms, fake_osm))

    materialize_dataset(
        tms_urls={"image": "https://tiles/{z}/{x}/{y}"},
        aoi={"type": "Polygon", "coordinates": _SQUARE},
        zoom=19,
        out_dir=str(tmp_path / "out"),
        label_classes=_ROOF_CLASSES,
        geometry_type="polygon",
        osm_api_url="https://osm",
    )

    labels = json.loads((tmp_path / "out" / "labels" / "labels.geojson").read_text())
    assert [feat["properties"]["label"] for feat in labels["features"]] == [1, 2, 3]


def test_materialize_dataset_uses_own_labels_without_calling_osm(tmp_path, monkeypatch) -> None:
    async def fake_download_tiles(*, out, **kwargs):
        (Path(out) / "chips").mkdir(parents=True)
        return str(Path(out) / "chips")

    async def fail_osm(**kwargs):
        raise AssertionError("own labels must not query OSM")

    monkeypatch.setattr(
        "fair.datasets._require_geomltoolkits",
        lambda: (SimpleNamespace(download_tiles=fake_download_tiles), SimpleNamespace(download_osm_data=fail_osm)),
    )
    own = {"features": [_feature("Polygon", [[[0.2, 0.2], [0.4, 0.2], [0.4, 0.4], [0.2, 0.2]]], 1)]}

    chips_dirs, labels_href, _, _ = materialize_dataset(
        tms_urls={"post": "https://post/{z}/{x}/{y}", "pre": "https://pre/{z}/{x}/{y}"},
        aoi={"type": "Polygon", "coordinates": _SQUARE},
        zoom=18,
        out_dir=str(tmp_path / "out"),
        label_classes=[{"name": "waste", "classes": ["*"]}],
        geometry_type="polygon",
        osm_api_url=None,
        labels=own,
    )

    assert sorted(chips_dirs) == ["post", "pre"]
    assert json.loads(Path(labels_href).read_text())["features"] == own["features"]


def test_materialize_dataset_slim_labels_keep_label_and_tags_one_feature_per_line(tmp_path, monkeypatch) -> None:
    async def fake_download_tiles(*, out, **kwargs):
        (Path(out) / "chips").mkdir(parents=True)
        return str(Path(out) / "chips")

    async def fake_download_osm(**kwargs):
        return {
            "features": [{"properties": {"osm_id": 7, "version": 3, "tags": {"building": "yes"}}, "geometry": None}] * 2
        }

    monkeypatch.setattr(
        "fair.datasets._require_geomltoolkits",
        lambda: (
            SimpleNamespace(download_tiles=fake_download_tiles),
            SimpleNamespace(download_osm_data=fake_download_osm),
        ),
    )

    _, labels_href, _, _ = materialize_dataset(
        tms_urls={"image": "https://tiles/{z}/{x}/{y}"},
        aoi={"type": "Polygon", "coordinates": _SQUARE},
        zoom=18,
        out_dir=str(tmp_path / "out"),
        label_classes=[{"name": "building", "classes": ["*"]}],
        geometry_type="polygon",
        osm_api_url="https://osm",
        slim_labels=True,
    )

    text = Path(labels_href).read_text()
    expected = {"label": 1, "tags": {"building": "yes"}}
    assert [feature["properties"] for feature in json.loads(text)["features"]] == [expected, expected]
    assert len(text.splitlines()) == 4


def _model_item(inputs: list[str], keywords: list[str], preview: dict | None) -> dict:
    return {
        "id": "demo-model",
        "properties": {
            "mlm:input": [{"name": name} for name in inputs],
            "mlm:tasks": ["semantic-segmentation"],
            "keywords": keywords,
            "fair:preview": preview,
        },
    }


_PREVIEW = {"center": [85.52, 27.63], "zoom": {"recommended": 18}, "imagery": {"url": "https://post/{z}/{x}/{y}"}}


def test_preview_sample_covers_exactly_the_requested_tile_square() -> None:
    sample = preview_sample(_model_item(["image"], ["building", "polygon"], _PREVIEW), tiles=4)

    west, south = sample.aoi["coordinates"][0][0]
    east, north = sample.aoi["coordinates"][0][2]
    covered = list(mercantile.tiles(west, south, east, north, 18))
    assert len(covered) == 16
    assert mercantile.tile(85.52, 27.63, 18) in covered
    assert sample.tms_urls == {"image": "https://post/{z}/{x}/{y}"}
    assert sample.zoom == 18
    assert sample.geometry_types == ["polygon"]


def test_preview_sample_maps_pre_imagery_to_the_pre_input() -> None:
    preview = {**_PREVIEW, "pre_imagery": {"url": "https://pre/{z}/{x}/{y}"}}
    sample = preview_sample(_model_item(["post", "pre"], ["building", "polygon"], preview), tiles=2)

    assert sample.tms_urls == {"pre": "https://pre/{z}/{x}/{y}", "post": "https://post/{z}/{x}/{y}"}


@pytest.mark.parametrize(
    ("inputs", "preview", "message"),
    [
        (["image"], None, "no fair:preview"),
        (["post", "pre"], _PREVIEW, "pre_imagery"),
        (["a", "b"], _PREVIEW, "maps to one input"),
    ],
)
def test_preview_sample_rejects_previews_it_cannot_map(inputs, preview, message) -> None:
    with pytest.raises(ValueError, match=message):
        preview_sample(_model_item(inputs, ["polygon"], preview), tiles=2)


def _feature(geometry_type: str, coordinates, label) -> dict:
    return {
        "type": "Feature",
        "properties": {"label": label},
        "geometry": {"type": geometry_type, "coordinates": coordinates},
    }


_AREA = {"type": "Polygon", "coordinates": _SQUARE}
_CLASSES = [{"name": "minor", "classes": ["*"]}, {"name": "major", "classes": ["*"]}]


def test_select_labels_keeps_only_features_inside_the_area() -> None:
    inside = _feature("Point", [0.5, 0.5], 2)
    background = _feature("Point", [0.6, 0.6], 0)
    outside = _feature("Point", [5.0, 5.0], 1)

    assert select_labels({"features": [inside, background, outside]}, _AREA, "point", _CLASSES) == [inside, background]


@pytest.mark.parametrize(
    ("feature", "message"),
    [
        (_feature("LineString", [[0.1, 0.1], [0.9, 0.9]], 1), "is not a point"),
        (_feature("Point", [0.5, 0.5], 3), r"label=3 not in \[0, 1, 2\]"),
        (_feature("Point", [0.5, 0.5], None), "label=None not in"),
    ],
)
def test_select_labels_rejects_wrong_geometry_or_class(feature, message) -> None:
    with pytest.raises(ValueError, match=message):
        select_labels({"features": [feature]}, _AREA, "point", _CLASSES)


def test_select_labels_fails_when_nothing_is_inside_the_area() -> None:
    with pytest.raises(ValueError, match="no label inside"):
        select_labels({"features": [_feature("Point", [5.0, 5.0], 1)]}, _AREA, "point", _CLASSES)
