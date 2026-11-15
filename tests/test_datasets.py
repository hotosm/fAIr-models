import json
import sys
from pathlib import Path
from types import SimpleNamespace

import mercantile
import pytest

from fair.datasets import (
    _aoi_geometry_and_bbox,
    _require_geomltoolkits,
    label_classes_of,
    materialize_dataset,
    model_osm_tags,
    osm_filters,
    preview_sample,
    select_labels,
)

_SQUARE = [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]]


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
    osm_tags = {"building": ["*"], "highway": ["residential", "primary"]}
    assert osm_filters(osm_tags, "polygon") == {
        "tags": {"polygon": {"join_or": {"building": [], "highway": ["residential", "primary"]}}}
    }


def test_model_osm_tags_merge_classes_by_key() -> None:
    properties = {
        "mlm:output": [
            {
                "classification:classes": [
                    {"name": "background", "value": 0},
                    {"name": "metal", "value": 1, "fair:osm_tags": {"roof:material": ["metal_sheet"]}},
                    {"name": "thatch", "value": 2, "fair:osm_tags": {"roof:material": ["thatch", "grass"]}},
                    {"name": "any", "value": 3, "fair:osm_tags": {"building": ["*"]}},
                ]
            }
        ]
    }
    assert model_osm_tags(properties) == {"roof:material": ["grass", "metal_sheet", "thatch"], "building": ["*"]}


def test_label_classes_of_lists_values_per_property() -> None:
    features = [
        {"properties": {"building": "yes"}},
        {"properties": {"building": "house", "roof:material": "thatch"}},
        {"properties": {"class": "waste", "tags": {"natural": "tree"}}},
    ]
    assert label_classes_of(features, ["building", "class", "natural"]) == [
        {"name": "building", "classes": ["house", "yes"]},
        {"name": "class", "classes": ["waste"]},
    ]


def test_require_geomltoolkits_fails_loud_without_extra(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "geomltoolkits", None)
    with pytest.raises(RuntimeError, match="serve"):
        _require_geomltoolkits()


def _fake_toolkits(monkeypatch, osm_features: list[dict] | None = None) -> None:
    async def fake_download_tiles(*, out, **kwargs):
        nested = Path(out) / "chips"
        nested.mkdir(parents=True)
        (nested / "OAM-1-2-19.tif").write_bytes(b"tif")
        return str(nested)

    async def fake_download_osm(*, filters, **kwargs):
        if osm_features is None:
            raise AssertionError("own labels must not query OSM")
        join_or = filters["tags"]["polygon"]["join_or"]
        return {"features": [feat for feat in osm_features if _matches_join_or(feat["properties"]["tags"], join_or)]}

    monkeypatch.setattr(
        "fair.datasets._require_geomltoolkits",
        lambda: (
            SimpleNamespace(download_tiles=fake_download_tiles),
            SimpleNamespace(download_osm_data=fake_download_osm),
        ),
    )


_POLYGON = {"type": "Polygon", "coordinates": [[[0.2, 0.2], [0.4, 0.2], [0.4, 0.4], [0.2, 0.2]]]}


def test_materialize_dataset_keeps_osm_tags_and_reports_label_classes(tmp_path, monkeypatch) -> None:
    osm = [
        {"properties": {"osm_id": 7, "tags": {"building": "yes"}}, "geometry": _POLYGON},
        {"properties": {"osm_id": 8, "tags": {"building": "house"}}, "geometry": _POLYGON},
        {"properties": {"osm_id": 9, "tags": {"highway": "residential"}}, "geometry": _POLYGON},
    ]
    _fake_toolkits(monkeypatch, osm)

    built = materialize_dataset(
        tms_urls={"image": "https://tiles/{z}/{x}/{y}"},
        aoi={"type": "Feature", "geometry": {"type": "Polygon", "coordinates": _SQUARE}},
        zoom=19,
        out_dir=str(tmp_path / "out"),
        geometry_type="polygon",
        osm_tags={"building": ["*"]},
        osm_api_url="https://osm",
    )

    assert built.chips_dirs == {"image": str(tmp_path / "out" / "chips-image")}
    assert [chip.name for chip in (tmp_path / "out" / "chips-image").iterdir()] == ["OAM-1-2-19.tif"]
    text = Path(built.labels_file).read_text()
    assert [feat["properties"] for feat in json.loads(text)["features"]] == [
        {"building": "yes", "tags": {"building": "yes"}},
        {"building": "house", "tags": {"building": "house"}},
    ]
    assert len(text.splitlines()) == 4
    assert built.label_properties == ["building"]
    assert built.label_classes == [{"name": "building", "classes": ["house", "yes"]}]
    assert built.geometry["type"] == "MultiPolygon"
    assert built.bbox == [0.0, 0.0, 1.0, 1.0]


def test_materialize_dataset_drops_osm_values_outside_the_requested_tags(tmp_path, monkeypatch) -> None:
    roofs = [
        {"properties": {"tags": {"roof:material": material}}, "geometry": _POLYGON}
        for material in ["metal_sheet", "thatch", "roof_tiles"]
    ]
    _fake_toolkits(monkeypatch, roofs)

    built = materialize_dataset(
        tms_urls={"image": "https://tiles/{z}/{x}/{y}"},
        aoi={"type": "Polygon", "coordinates": _SQUARE},
        zoom=19,
        out_dir=str(tmp_path / "out"),
        geometry_type="polygon",
        osm_tags={"roof:material": ["metal_sheet", "thatch"]},
        osm_api_url="https://osm",
    )

    assert built.label_classes == [{"name": "roof:material", "classes": ["metal_sheet", "thatch"]}]


def test_materialize_dataset_uses_own_labels_without_calling_osm(tmp_path, monkeypatch) -> None:
    _fake_toolkits(monkeypatch)
    own = {"features": [{"type": "Feature", "properties": {"class": "waste", "note": "x"}, "geometry": _POLYGON}]}

    built = materialize_dataset(
        tms_urls={"post": "https://post/{z}/{x}/{y}", "pre": "https://pre/{z}/{x}/{y}"},
        aoi={"type": "Polygon", "coordinates": _SQUARE},
        zoom=18,
        out_dir=str(tmp_path / "out"),
        geometry_type="polygon",
        labels=own,
    )

    assert sorted(built.chips_dirs) == ["post", "pre"]
    assert [feat["properties"] for feat in json.loads(Path(built.labels_file).read_text())["features"]] == [
        {"class": "waste"}
    ]
    assert built.label_classes == [{"name": "class", "classes": ["waste"]}]


def test_materialize_dataset_needs_osm_tags_without_own_labels(tmp_path, monkeypatch) -> None:
    _fake_toolkits(monkeypatch, [])
    with pytest.raises(ValueError, match="need osm_tags"):
        materialize_dataset(
            tms_urls={"image": "https://tiles/{z}/{x}/{y}"},
            aoi={"type": "Polygon", "coordinates": _SQUARE},
            zoom=18,
            out_dir=str(tmp_path / "out"),
            geometry_type="polygon",
            osm_api_url="https://osm",
        )


def _model_item(inputs: list[str], keywords: list[str], preview: dict | None) -> dict:
    return {
        "id": "demo-model",
        "properties": {
            "mlm:input": [{"name": name} for name in inputs],
            "mlm:output": [
                {
                    "classification:classes": [
                        {"name": "background", "value": 0},
                        {"name": "building", "value": 1, "fair:osm_tags": {"building": ["*"]}},
                    ]
                }
            ],
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
    assert sample.osm_tags == {"building": ["*"]}
    assert sample.keywords == ["building"]


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


def _feature(geometry_type: str, coordinates, properties: dict) -> dict:
    return {
        "type": "Feature",
        "properties": properties,
        "geometry": {"type": geometry_type, "coordinates": coordinates},
    }


_AREA = {"type": "Polygon", "coordinates": _SQUARE}


def test_select_labels_keeps_features_inside_the_area_with_only_the_class() -> None:
    inside = _feature("Point", [0.5, 0.5], {"class": "major", "id": 3})
    outside = _feature("Point", [5.0, 5.0], {"class": "minor"})

    assert select_labels({"features": [inside, outside]}, _AREA, "point", "class") == [
        _feature("Point", [0.5, 0.5], {"class": "major"})
    ]


@pytest.mark.parametrize(
    ("feature", "message"),
    [
        (_feature("LineString", [[0.1, 0.1], [0.9, 0.9]], {"class": "major"}), "is not a point"),
        (_feature("Point", [0.5, 0.5], {"kind": "major"}), "no 'class' property"),
    ],
)
def test_select_labels_rejects_wrong_geometry_or_missing_class(feature, message) -> None:
    with pytest.raises(ValueError, match=message):
        select_labels({"features": [feature]}, _AREA, "point", "class")


def test_select_labels_fails_when_nothing_is_inside_the_area() -> None:
    with pytest.raises(ValueError, match="no label inside"):
        select_labels({"features": [_feature("Point", [5.0, 5.0], {"class": "a"})]}, _AREA, "point", "class")


def test_materialize_dataset_declares_only_the_requested_values_of_each_key(tmp_path, monkeypatch) -> None:
    osm = [
        {"properties": {"tags": {"natural": "tree"}}, "geometry": _POLYGON},
        {"properties": {"tags": {"landuse": "forest", "natural": "wood"}}, "geometry": _POLYGON},
    ]
    _fake_toolkits(monkeypatch, osm)

    built = materialize_dataset(
        tms_urls={"image": "https://tiles/{z}/{x}/{y}"},
        aoi={"type": "Polygon", "coordinates": _SQUARE},
        zoom=19,
        out_dir=str(tmp_path / "out"),
        geometry_type="polygon",
        osm_tags={"natural": ["tree"], "landuse": ["forest"]},
        osm_api_url="https://osm",
    )

    assert built.label_classes == [
        {"name": "natural", "classes": ["tree"]},
        {"name": "landuse", "classes": ["forest"]},
    ]
    second = json.loads(Path(built.labels_file).read_text())["features"][1]["properties"]
    assert second == {"landuse": "forest", "tags": {"landuse": "forest", "natural": "wood"}}
