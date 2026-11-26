import io
import json
import os
import shutil
from pathlib import Path
from typing import Literal

import geopandas as gpd
import mercantile
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import rasterio
from PIL import Image
from pmtiles.reader import Reader
from pmtiles.tile import zxy_to_tileid
from rasterio.transform import from_bounds, from_origin
from shapely.geometry import Point, box, mapping
from upath import UPath

from fair.bundle import (
    chip_bounds,
    collect_chips,
    grid_tile,
    grid_tiles,
    index_chips,
    read_labels,
    read_tile_ids,
    write_bundle,
    write_imagery_pmtiles,
    write_labels_parquet,
    write_labels_pmtiles,
    write_tiles_parquet,
)
from fair.stac.builders import add_bundle_links, build_dataset_item
from fair.stac.constants import WEB_MAP_LINKS_SCHEMA
from fair.stac.validators import validate_item

Z = 19
X, Y = 233651, 245320


def _geotiff(path: Path, crs: str, transform: rasterio.Affine, size: int = 64) -> np.ndarray:
    rng = np.random.default_rng(len(path.name))
    pixels = rng.integers(0, 255, size=(3, size, size), dtype=np.uint8)
    with rasterio.open(
        path, "w", driver="GTiff", width=size, height=size, count=3, dtype="uint8", crs=crs, transform=transform
    ) as chip:
        chip.write(pixels)
    return np.moveaxis(pixels, 0, -1)


def _transform(bounds: tuple[float, float, float, float], size: int) -> rasterio.Affine:
    west, south, east, north = bounds
    return from_bounds(west, south, east, north, size, size)


def _chip(folder: Path, x: int, y: int, z: int = Z, name: str | None = None) -> np.ndarray:
    path = folder / (name or f"OAM-{x}-{y}-{z}.tif")
    return _geotiff(path, "EPSG:3857", _transform(mercantile.xy_bounds(x, y, z), 64))


def _scene(folder: Path, name: str = "drone.tif") -> Path:
    west, _, _, north = mercantile.xy_bounds(X, Y, Z)
    _geotiff(folder / name, "EPSG:3857", from_origin(west + 7, north - 7, 0.5, 0.5), size=200)
    return folder / name


def _inside(x: int, y: int, value: str) -> dict:
    west, south, east, north = mercantile.bounds(x, y, Z)
    centre = Point((west + east) / 2, (south + north) / 2)
    return {"type": "Feature", "properties": {"class": value}, "geometry": mapping(centre.buffer(1e-6))}


AREA = box(*mercantile.bounds(X, Y, Z))


def test_labels_parquet_is_geoparquet_1_1_with_a_bbox_column_and_only_the_class(tmp_path: Path) -> None:
    target = tmp_path / "labels.parquet"
    feature = _inside(X, Y, "building")
    feature["properties"]["osm_id"] = 7

    write_labels_parquet([feature], ["class"], target)

    geo = json.loads(pq.read_schema(target).metadata[b"geo"])
    assert geo["version"] == "1.1.0"
    assert geo["columns"]["geometry"]["covering"]["bbox"]["xmin"] == ["bbox", "xmin"]
    assert sorted(pq.read_schema(target).names) == ["bbox", "class", "geometry"]
    assert pq.read_metadata(target).row_group(0).column(0).compression == "ZSTD"


@pytest.mark.parametrize("version", ["1.0.0", "1.1.0", "2.0.0"])
def test_reading_labels_keeps_only_features_in_the_area(
    tmp_path: Path, version: Literal["1.0.0", "1.1.0", "2.0.0"]
) -> None:
    target = tmp_path / "labels.parquet"
    frame = gpd.GeoDataFrame.from_features([_inside(X, Y, "near"), _inside(X + 50, Y, "far")], crs="EPSG:4326")
    frame.to_parquet(target, index=False, schema_version=version)

    found = read_labels(target.read_bytes(), AREA, "class")

    assert found["class"].tolist() == ["near"]
    assert list(found.columns) == ["class", "geometry"]


def test_labels_in_another_crs_are_reprojected(tmp_path: Path) -> None:
    target = tmp_path / "labels.parquet"
    frame = gpd.GeoDataFrame.from_features([_inside(X, Y, "a")], crs="EPSG:4326").to_crs(3857)
    frame.rename_geometry("geom", inplace=True)
    frame.to_parquet(target)

    found = read_labels(target.read_bytes(), AREA, "class")

    assert found.crs == "EPSG:4326"
    assert found.geometry.iloc[0].within(AREA)


def test_a_features_list_without_a_type_is_read_too() -> None:
    body = json.dumps({"features": [_inside(X, Y, "a")]}).encode()

    assert read_labels(body, AREA, "class")["class"].tolist() == ["a"]


def test_geojson_labels_are_read_too(tmp_path: Path) -> None:
    target = tmp_path / "labels.geojson"
    target.write_text(json.dumps({"type": "FeatureCollection", "features": [_inside(X, Y, "a")]}))

    assert read_labels(target.read_bytes(), AREA, "class")["class"].tolist() == ["a"]


@pytest.mark.parametrize("suffix", ["parquet", "geojson"])
def test_reading_labels_without_the_class_fails(tmp_path: Path, suffix: str) -> None:
    target = tmp_path / f"labels.{suffix}"
    frame = gpd.GeoDataFrame.from_features([_inside(X, Y, "a")], crs="EPSG:4326")
    if suffix == "parquet":
        frame.to_parquet(target)
    else:
        frame.to_file(target, driver="GeoJSON")

    with pytest.raises(ValueError, match="'kind'"):
        read_labels(target.read_bytes(), AREA, "kind")


def test_reading_a_plain_parquet_file_fails_clearly(tmp_path: Path) -> None:
    target = tmp_path / "plain.parquet"
    pq.write_table(pa.table({"class": ["a"]}), target)

    with pytest.raises(ValueError, match="geo metadata"):
        read_labels(target.read_bytes(), AREA, "class")


def test_tiles_parquet_counts_the_features_on_each_tile(tmp_path: Path) -> None:
    chips_dir = tmp_path / "chips"
    chips_dir.mkdir()
    _chip(chips_dir, X, Y)
    _chip(chips_dir, X + 1, Y)
    labels = tmp_path / "labels.parquet"
    write_labels_parquet([_inside(X, Y, "tree"), _inside(X, Y, "tree"), _inside(X, Y, "tree_group")], ["class"], labels)
    chips = collect_chips([UPath(chips_dir)])

    write_tiles_parquet(chips, chip_bounds(chips), labels, "class", tmp_path / "tiles.parquet")

    rows = {row["chip_id"]: row for row in pq.read_table(tmp_path / "tiles.parquet").to_pylist()}
    assert rows[f"OAM-{X}-{Y}-{Z}"]["num_features"] == 3
    assert json.loads(rows[f"OAM-{X}-{Y}-{Z}"]["class_counts"]) == {"tree": 2, "tree_group": 1}
    assert (rows[f"OAM-{X + 1}-{Y}-{Z}"]["num_features"], rows[f"OAM-{X + 1}-{Y}-{Z}"]["class_counts"]) == (0, "{}")


@pytest.mark.parametrize(("quality", "exact"), [("lossless", True), ("high", False), ("medium", False)])
def test_imagery_pmtiles_keeps_every_pixel_only_when_lossless(tmp_path: Path, quality: str, exact: bool) -> None:
    chips_dir = tmp_path / "chips"
    chips_dir.mkdir()
    pixels = _chip(chips_dir, X, Y)
    chips = collect_chips([UPath(chips_dir)])
    tiles = grid_tiles(chip_bounds(chips))
    target = tmp_path / "imagery.pmtiles"

    assert tiles is not None
    write_imagery_pmtiles({tile: chips[chip_id] for tile, chip_id in tiles.items()}, target, quality)

    data = target.read_bytes()
    tile = Reader(lambda offset, length: data[offset : offset + length]).get(Z, X, Y)
    decoded = np.asarray(Image.open(io.BytesIO(tile)).convert("RGB"))
    assert np.array_equal(decoded, pixels) is exact


def test_chips_from_several_folders_keep_the_first_file_of_each_name(tmp_path: Path) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    _chip(first, X, Y)
    _chip(second, X, Y)
    _chip(second, X + 1, Y)

    chips = collect_chips([UPath(first), UPath(second)])

    assert sorted(chips) == [f"OAM-{x}-{Y}-{Z}" for x in (X, X + 1)]
    assert chips[f"OAM-{X}-{Y}-{Z}"].parent == UPath(first)


def test_chips_are_tiles_by_their_georeferencing_whatever_their_name(tmp_path: Path) -> None:
    _chip(tmp_path, X, Y, name="esri_scene_a.tif")
    _chip(tmp_path, X + 1, Y, name="OAM-1-1-1.tif")
    chips = index_chips(UPath(tmp_path).iterdir())

    tiles = grid_tiles(chip_bounds(chips))

    assert tiles == {
        mercantile.Tile(X, Y, Z): "esri_scene_a",
        mercantile.Tile(X + 1, Y, Z): "OAM-1-1-1",
    }


def test_a_chip_georeferenced_in_epsg_4326_is_a_tile_too(tmp_path: Path) -> None:
    _geotiff(tmp_path / "chip.tif", "EPSG:4326", _transform(mercantile.bounds(X, Y, Z), 64))
    chips = index_chips(UPath(tmp_path).iterdir())

    assert grid_tile(chip_bounds(chips)["chip"]) == mercantile.Tile(X, Y, Z)


@pytest.mark.parametrize("other", ["scene", "zoom", "same-tile"])
def test_chips_that_are_not_tiles_at_one_zoom_are_no_grid(tmp_path: Path, other: str) -> None:
    _chip(tmp_path, X, Y)
    if other == "scene":
        _scene(tmp_path)
    elif other == "same-tile":
        _chip(tmp_path, X, Y, name="copy.tif")
    else:
        _chip(tmp_path, X // 2, Y // 2, Z - 1)
    chips = index_chips(UPath(tmp_path).iterdir())

    assert grid_tiles(chip_bounds(chips)) is None


def test_an_ungeoreferenced_chip_is_refused(tmp_path: Path) -> None:
    with rasterio.open(tmp_path / "plain.tif", "w", driver="GTiff", width=4, height=4, count=1, dtype="uint8") as tif:
        tif.write(np.zeros((1, 4, 4), dtype=np.uint8))

    with pytest.raises(ValueError, match="not georeferenced"):
        chip_bounds(index_chips(UPath(tmp_path).iterdir()))


@pytest.mark.skipif(shutil.which("tippecanoe") is None and "CI" not in os.environ, reason="tippecanoe is not installed")
def test_labels_pmtiles_merges_parquet_and_geojson_into_one_layer(tmp_path: Path) -> None:
    points, groups = tmp_path / "points.parquet", tmp_path / "groups.geojson"
    write_labels_parquet([_inside(X, Y, "tree")], ["class"], points)
    groups.write_text(json.dumps({"type": "FeatureCollection", "features": [_inside(X, Y, "tree_group")]}))
    target = tmp_path / "labels.pmtiles"

    write_labels_pmtiles([points, groups], Z, target)

    data = target.read_bytes()
    stats = Reader(lambda offset, length: data[offset : offset + length]).metadata()["tilestats"]
    assert [layer["layer"] for layer in stats["layers"]] == ["labels"]
    assert sorted(stats["layers"][0]["attributes"][0]["values"]) == ["tree", "tree_group"]


@pytest.mark.skipif(shutil.which("tippecanoe") is None and "CI" not in os.environ, reason="tippecanoe is not installed")
def test_write_bundle_writes_one_imagery_file_per_input(tmp_path: Path) -> None:
    chips_dir = tmp_path / "chips"
    chips_dir.mkdir()
    _chip(chips_dir, X, Y)
    out = tmp_path / "bundle"
    out.mkdir()

    chips = collect_chips([UPath(chips_dir)])

    single = write_bundle({"image": chips}, [_inside(X, Y, "building")], ["class"], out)
    double = write_bundle({"pre": chips, "post": chips}, [_inside(X, Y, "building")], ["class"], out)

    assert sorted(single) == ["imagery_pmtiles", "labels_parquet", "labels_pmtiles", "tiles_parquet"]
    assert all(path.stat().st_size > 0 for path in single.values())
    assert double["imagery_pmtiles_pre"].name == "imagery-pre.pmtiles"
    assert "imagery_pmtiles" not in double


@pytest.mark.skipif(shutil.which("tippecanoe") is None and "CI" not in os.environ, reason="tippecanoe is not installed")
def test_a_bundle_of_images_that_are_not_tiles_has_no_imagery_pmtiles(tmp_path: Path) -> None:
    chips_dir = tmp_path / "chips"
    chips_dir.mkdir()
    _scene(chips_dir, "drone_a.tif")
    out = tmp_path / "bundle"
    out.mkdir()

    paths = write_bundle({"image": collect_chips([UPath(chips_dir)])}, [_inside(X, Y, "building")], ["class"], out)

    assert sorted(paths) == ["labels_parquet", "labels_pmtiles", "tiles_parquet"]
    row = pq.read_table(paths["tiles_parquet"]).to_pylist()[0]
    assert (row["chip_id"], row["num_features"]) == ("drone_a", 1)
    with pytest.raises(ValueError, match="not distinct web-mercator tiles"):
        read_tile_ids(UPath(paths["tiles_parquet"]))


def _dataset_item(tmp_path: Path, bundle_hrefs: dict[str, str] | None):
    labels = tmp_path / "labels.geojson"
    labels.write_text(json.dumps({"type": "FeatureCollection", "features": [_inside(X, Y, "building")]}))
    return build_dataset_item(
        label_type="vector",
        label_tasks=["segmentation"],
        label_classes=[{"name": "class", "classes": ["building"]}],
        keywords=["building"],
        chips_hrefs={"image": "chips/"},
        labels_href=str(labels),
        title="Bundle Dataset",
        description="Chips and labels.",
        user_id="osm-user-42",
        providers=[{"name": "HOT", "roles": ["producer"]}],
        bundle_hrefs=bundle_hrefs,
    )


def test_bundle_files_become_assets_and_pmtiles_become_web_map_links(tmp_path: Path) -> None:
    hrefs = {
        "labels_parquet": "https://data.example/labels.parquet",
        "tiles_parquet": "https://data.example/tiles.parquet",
        "imagery_pmtiles": "https://data.example/imagery.pmtiles",
        "labels_pmtiles": "https://data.example/labels.pmtiles",
    }
    item = _dataset_item(tmp_path, hrefs)

    add_bundle_links(item)

    assert item.assets["labels_parquet"].media_type == "application/vnd.apache.parquet"
    assert item.assets["imagery_pmtiles"].media_type == "application/vnd.pmtiles"
    links = {link.target: link for link in item.links if link.rel == "pmtiles"}
    assert set(links) == {hrefs["imagery_pmtiles"], hrefs["labels_pmtiles"]}
    assert links[hrefs["labels_pmtiles"]].extra_fields["pmtiles:layers"] == ["labels"]
    assert all(link.media_type is None for link in links.values())
    assert WEB_MAP_LINKS_SCHEMA in item.stac_extensions
    assert validate_item(item) == []


def test_an_item_without_a_bundle_gets_no_map_links(tmp_path: Path) -> None:
    item = _dataset_item(tmp_path, None)

    add_bundle_links(item)

    assert not [link for link in item.links if link.rel == "pmtiles"]
    assert WEB_MAP_LINKS_SCHEMA not in item.stac_extensions


def test_tiles_parquet_leaves_missing_classes_out_of_the_counts(tmp_path: Path) -> None:
    chips_dir = tmp_path / "chips"
    chips_dir.mkdir()
    _chip(chips_dir, X, Y)
    labels = tmp_path / "labels.parquet"
    untagged = _inside(X, Y, "x")
    untagged["properties"]["class"] = None
    write_labels_parquet([untagged, _inside(X, Y, "yes")], ["class"], labels)
    chips = collect_chips([UPath(chips_dir)])

    write_tiles_parquet(chips, chip_bounds(chips), labels, "class", tmp_path / "tiles.parquet")

    row = pq.read_table(tmp_path / "tiles.parquet").to_pylist()[0]
    assert (row["num_features"], json.loads(row["class_counts"])) == (2, {"yes": 1})


def test_numeric_classes_are_written_as_their_str(tmp_path: Path) -> None:
    target = tmp_path / "labels.parquet"
    features = [_inside(X, Y, "x") for _ in range(4)]
    for feature, value in zip(features, [1, 2.5, None, "x"], strict=True):
        feature["properties"]["class"] = value

    write_labels_parquet(features, ["class"], target)

    assert pq.read_table(target).column("class").to_pylist() == ["1", "2.5", None, "x"]


def test_only_geotiffs_are_indexed_as_chips(tmp_path: Path) -> None:
    _chip(tmp_path, X, Y)
    (tmp_path / "notes.txt").write_text("not a chip")
    (tmp_path / "OAM-1-2-19.tif.aux.xml").write_text("sidecar")
    (tmp_path / "OAM-1-2-19").mkdir()

    chips = index_chips(UPath(tmp_path).iterdir())

    assert chips == {f"OAM-{X}-{Y}-{Z}": UPath(tmp_path) / f"OAM-{X}-{Y}-{Z}.tif"}


def test_tile_ids_come_from_tiles_parquet(tmp_path: Path) -> None:
    chips_dir = tmp_path / "chips"
    chips_dir.mkdir()
    _chip(chips_dir, X + 1, Y)
    _chip(chips_dir, X, Y)
    labels = tmp_path / "labels.parquet"
    write_labels_parquet([_inside(X, Y, "a")], ["class"], labels)
    chips = collect_chips([UPath(chips_dir)])
    write_tiles_parquet(chips, chip_bounds(chips), labels, "class", tmp_path / "tiles.parquet")

    assert read_tile_ids(UPath(tmp_path / "tiles.parquet")) == sorted(zxy_to_tileid(Z, x, Y) for x in (X, X + 1))


@pytest.mark.parametrize("quality", ["low", "90", ""])
def test_unknown_imagery_quality_is_refused(tmp_path: Path, quality: str) -> None:
    with pytest.raises(ValueError, match="must be one of"):
        write_imagery_pmtiles({}, tmp_path / "imagery.pmtiles", quality)


def test_a_gdal_vrt_is_not_opened_as_labels(tmp_path: Path) -> None:
    secret = tmp_path / "secret.csv"
    secret.write_text("class,WKT\nsecret,POINT (0 0)\n")
    vrt = tmp_path / "labels.vrt"
    vrt.write_text(
        f'<OGRVRTDataSource><OGRVRTLayer name="secret"><SrcDataSource>{secret}</SrcDataSource>'
        '<GeometryField encoding="WKT" field="WKT"/></OGRVRTLayer></OGRVRTDataSource>'
    )

    with pytest.raises(ValueError, match="neither GeoParquet nor GeoJSON"):
        read_labels(vrt.read_bytes(), AREA, "class")


def test_geojson_that_is_not_a_feature_collection_is_refused(tmp_path: Path) -> None:
    target = tmp_path / "labels.geojson"
    target.write_text(json.dumps(_inside(X, Y, "a")))

    with pytest.raises(ValueError, match="FeatureCollection"):
        read_labels(target.read_bytes(), AREA, "class")


def test_geoparquet_with_a_null_crs_is_refused(tmp_path: Path) -> None:
    target = tmp_path / "labels.parquet"
    gpd.GeoDataFrame.from_features([_inside(X, Y, "a")]).to_parquet(target)

    with pytest.raises(ValueError, match="naive geometries"):
        read_labels(target.read_bytes(), AREA, "class")


@pytest.mark.parametrize("body", [b"[" * 50000, b"\x89PNG\r\n"], ids=["nested-too-deep", "not-utf-8"])
def test_labels_that_are_not_json_are_refused_clearly(tmp_path: Path, body: bytes) -> None:
    target = tmp_path / "labels"
    target.write_bytes(body)

    with pytest.raises(ValueError, match="neither GeoParquet nor GeoJSON"):
        read_labels(target.read_bytes(), AREA, "class")


@pytest.mark.parametrize("class_property", ["tile_id", "chip_id", "index_right", "num_features"])
def test_tiles_parquet_takes_any_class_property_name(tmp_path: Path, class_property: str) -> None:
    _chip(tmp_path, X, Y)
    labels = tmp_path / "labels.parquet"
    feature = _inside(X, Y, "x")
    feature["properties"] = {class_property: "a"}
    write_labels_parquet([feature], [class_property], labels)
    chips = index_chips(UPath(tmp_path).iterdir())

    write_tiles_parquet(chips, chip_bounds(chips), labels, class_property, tmp_path / "tiles.parquet")

    row = pq.read_table(tmp_path / "tiles.parquet").to_pylist()[0]
    assert (row["chip_id"], row["num_features"], row["class_counts"]) == (f"OAM-{X}-{Y}-{Z}", 1, '{"a": 1}')


@pytest.mark.skipif(shutil.which("tippecanoe") is None and "CI" not in os.environ, reason="tippecanoe is not installed")
def test_labels_pmtiles_skips_null_geometries(tmp_path: Path) -> None:
    with_null = tmp_path / "with_null.geojson"
    null = {"type": "Feature", "properties": {"class": "none"}, "geometry": None}
    with_null.write_text(json.dumps({"type": "FeatureCollection", "features": [null, _inside(X, Y, "tree")]}))
    target = tmp_path / "labels.pmtiles"

    write_labels_pmtiles([with_null], Z, target)

    data = target.read_bytes()
    stats = Reader(lambda offset, length: data[offset : offset + length]).metadata()["tilestats"]
    assert stats["layers"][0]["attributes"][0]["values"] == ["tree"]


def _jpeg_chip_with_sidecar(folder: Path) -> None:
    path = folder / f"OAM-{X}-{Y}-{Z}.tif"
    with rasterio.open(path, "w", driver="JPEG", width=64, height=64, count=3, dtype="uint8") as chip:
        chip.write(np.zeros((3, 64, 64), dtype=np.uint8))
    with rasterio.open(path, "r+") as chip:
        chip.crs = rasterio.crs.CRS.from_epsg(4326)
        chip.transform = _transform(mercantile.bounds(X, Y, Z), 64)


@pytest.mark.parametrize("storage", ["local", "memory"])
def test_a_jpeg_chip_is_georeferenced_by_its_aux_xml_sidecar(tmp_path: Path, storage: str) -> None:
    _jpeg_chip_with_sidecar(tmp_path)
    assert (tmp_path / f"OAM-{X}-{Y}-{Z}.tif.aux.xml").exists()
    folder = UPath(tmp_path)
    if storage == "memory":
        folder = UPath("memory:///chips")
        for path in tmp_path.iterdir():
            (folder / path.name).write_bytes(path.read_bytes())
    chips = collect_chips([folder])

    assert grid_tiles(chip_bounds(chips)) == {mercantile.Tile(X, Y, Z): f"OAM-{X}-{Y}-{Z}"}


@pytest.mark.skipif(shutil.which("tippecanoe") is None and "CI" not in os.environ, reason="tippecanoe is not installed")
@pytest.mark.parametrize(("count", "dtype"), [(1, "uint8"), (3, "uint16")])
def test_tile_chips_that_are_not_8_bit_rgb_get_no_imagery_pmtiles(tmp_path: Path, count: int, dtype: str) -> None:
    transform = _transform(mercantile.xy_bounds(X, Y, Z), 4)
    profile = {"driver": "GTiff", "width": 4, "height": 4, "count": count, "dtype": dtype, "crs": "EPSG:3857"}
    with rasterio.open(tmp_path / "chip.tif", "w", transform=transform, **profile) as chip:
        chip.write(np.zeros((count, 4, 4), dtype=dtype))
    out = tmp_path / "bundle"
    out.mkdir()

    paths = write_bundle({"image": index_chips(UPath(tmp_path).iterdir())}, [_inside(X, Y, "a")], ["class"], out)

    assert sorted(paths) == ["labels_parquet", "labels_pmtiles", "tiles_parquet"]
