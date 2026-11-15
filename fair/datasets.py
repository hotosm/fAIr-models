"""Local dataset materialization for the `fair dataset build` CLI command.

Mirrors the backend's server-side build (download imagery chips + OSM labels for
an AOI) so an operator can produce a draft dataset STAC item from a TMS URL and a
GeoJSON AOI, edit it, and register it. The download itself uses geomltoolkits,
which lives in the `serve` optional extra.

Labels keep their source form (OSM `tags`, or the contributor's class property);
a model's class values are applied at training time by `fair.stac.label_classes`.
"""

import asyncio
import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fair.stac.builders import _bbox_from_coords, _flatten_coords
from fair.stac.label_classes import model_classes
from fair.stac.validators import GEOMETRY_GEOJSON_TYPES, accepted_label_geometries

logger = logging.getLogger(__name__)

PRE_INPUT = "pre"
DEFAULT_CLASS_PROPERTY = "class"


def _require_geomltoolkits() -> tuple[Any, Any]:
    try:
        from geomltoolkits.downloader import osm, tms
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "`fair dataset build` needs geomltoolkits (the 'serve' extra). Install it with: uv sync --extra serve"
        ) from exc
    return tms, osm


def _aoi_geometry_and_bbox(aoi: dict[str, Any]) -> tuple[dict[str, Any], list[float]]:
    """Union the AOI's polygon geometries into a MultiPolygon and compute its bbox."""
    kind = aoi.get("type")
    if kind == "FeatureCollection":
        geoms = [feat["geometry"] for feat in aoi.get("features", [])]
    elif kind == "Feature":
        geoms = [aoi["geometry"]]
    else:
        geoms = [aoi]
    if not geoms:
        raise ValueError("AOI contains no geometry")

    polygons: list[Any] = []
    coords: list[list[float]] = []
    for geom in geoms:
        geom_type = geom["type"]
        coords.extend(_flatten_coords(geom["coordinates"]))
        if geom_type == "Polygon":
            polygons.append(geom["coordinates"])
        elif geom_type == "MultiPolygon":
            polygons.extend(geom["coordinates"])
        else:
            raise ValueError(f"AOI geometry must be Polygon or MultiPolygon, got {geom_type}")

    return {"type": "MultiPolygon", "coordinates": polygons}, _bbox_from_coords(coords)


def osm_filters(osm_tags: Mapping[str, list[str]], geometry_type: str) -> dict[str, Any]:
    """raw-data API `filters` for `osm_tags`; `["*"]` asks for every value, which the API spells as an empty list."""
    join_or = {key: [] if "*" in values else list(values) for key, values in osm_tags.items()}
    return {"tags": {geometry_type: {"join_or": join_or}}}


def model_osm_tags(model_properties: Mapping[str, Any]) -> dict[str, list[str]]:
    """OSM tags of every model class that declares `fair:osm_tags`, merged by key."""
    merged: dict[str, list[str]] = {}
    for cls in model_classes(model_properties):
        for key, values in cls.osm_tags.items():
            if "*" in values or merged.get(key) == ["*"]:
                merged[key] = ["*"]
            else:
                merged[key] = sorted({*merged.get(key, []), *values})
    return merged


def osm_label_features(osm_features: list[dict[str, Any]], osm_tags: Mapping[str, list[str]]) -> list[dict[str, Any]]:
    """Label features from raw-data API features: the tag keys that match `osm_tags`, beside the full `tags`."""
    features = []
    for feature in osm_features:
        tags = (feature.get("properties") or {}).get("tags") or {}
        matched = {
            key: tags[key] for key, values in osm_tags.items() if key in tags and ("*" in values or tags[key] in values)
        }
        if matched:
            features.append(
                {"type": "Feature", "properties": {**matched, "tags": tags}, "geometry": feature["geometry"]}
            )
    return features


@dataclass(frozen=True)
class PreviewSample:
    """Build parameters for a model's sample dataset, read from its `fair:preview` and classes."""

    tms_urls: dict[str, str]
    zoom: int
    aoi: dict[str, Any]
    tasks: list[str]
    geometry_types: list[str]
    osm_tags: dict[str, list[str]]
    keywords: list[str]


def preview_sample(model_item: dict[str, Any], tiles: int) -> PreviewSample:
    """Square of `tiles` x `tiles` tiles at the preview centre and zoom, imagery mapped to the model's inputs."""
    import mercantile

    properties = model_item["properties"]
    preview = properties.get("fair:preview")
    if not preview:
        raise ValueError(f"model item {model_item['id']!r} has no fair:preview")
    inputs = [entry["name"] for entry in properties["mlm:input"]]
    tms_urls: dict[str, str] = {}
    if PRE_INPUT in inputs:
        if "pre_imagery" not in preview:
            raise ValueError(f"model input {PRE_INPUT!r} needs fair:preview.pre_imagery")
        tms_urls[PRE_INPUT] = preview["pre_imagery"]["url"]
    others = [name for name in inputs if name != PRE_INPUT]
    if len(others) != 1:
        raise ValueError(f"fair:preview.imagery maps to one input, the model declares {others}")
    tms_urls[others[0]] = preview["imagery"]["url"]

    zoom = int(preview["zoom"]["recommended"])
    lon, lat = preview["center"]
    centre = mercantile.tile(lon, lat, zoom)
    first_x, first_y = centre.x - tiles // 2, centre.y - tiles // 2
    north_west = mercantile.bounds(first_x, first_y, zoom)
    south_east = mercantile.bounds(first_x + tiles - 1, first_y + tiles - 1, zoom)
    inset = (north_west.east - north_west.west) / 100
    west, east = north_west.west + inset, south_east.east - inset
    south, north = south_east.south + inset, north_west.north - inset
    ring = [[west, south], [east, south], [east, north], [west, north], [west, south]]
    aoi = {"type": "Polygon", "coordinates": [ring]}

    geometry_types = accepted_label_geometries(properties)
    if not geometry_types:
        raise ValueError(f"model declares none of the label geometry types {sorted(GEOMETRY_GEOJSON_TYPES)}")
    keywords = [keyword for keyword in properties.get("keywords", []) if keyword not in GEOMETRY_GEOJSON_TYPES]
    return PreviewSample(
        tms_urls=tms_urls,
        zoom=zoom,
        aoi=aoi,
        tasks=list(properties["mlm:tasks"]),
        geometry_types=geometry_types,
        osm_tags=model_osm_tags(properties),
        keywords=keywords,
    )


def select_labels(
    labels: dict[str, Any], aoi_geometry: dict[str, Any], geometry_type: str, class_property: str
) -> list[dict[str, Any]]:
    """Features of `labels` inside the AOI, keeping only `class_property`; raise on a wrong geometry or no class."""
    from shapely.geometry import shape

    area = shape(aoi_geometry)
    selected: list[dict[str, Any]] = []
    problems: list[str] = []
    for position, feature in enumerate(labels.get("features", [])):
        geometry = feature.get("geometry")
        if geometry is None or not shape(geometry).intersects(area):
            continue
        if geometry["type"] not in GEOMETRY_GEOJSON_TYPES[geometry_type]:
            problems.append(f"feature {position}: {geometry['type']} is not a {geometry_type}")
        value = (feature.get("properties") or {}).get(class_property)
        if value is None:
            problems.append(f"feature {position}: no {class_property!r} property")
        selected.append({"type": "Feature", "properties": {class_property: value}, "geometry": geometry})
    if problems:
        raise ValueError("labels do not match the dataset:\n" + "\n".join(problems[:20]))
    if not selected:
        raise ValueError("no label inside the dataset area")
    return selected


def label_classes_of(features: list[dict[str, Any]], names: list[str]) -> list[dict[str, Any]]:
    """STAC Label extension `label:classes`: the values each property in `names` takes across the features."""
    values: dict[str, set[str]] = {name: set() for name in names}
    for feature in features:
        properties = feature.get("properties") or {}
        for name in names:
            if name in properties:
                values[name].add(str(properties[name]))
    return [{"name": name, "classes": sorted(found)} for name, found in values.items() if found]


def feature_collection_lines(features: list[dict[str, Any]]) -> str:
    lines = ",\n".join(json.dumps(feature, separators=(",", ":")) for feature in features)
    return f'{{"type":"FeatureCollection","features":[\n{lines}\n]}}\n'


def _download_chips(tms_mod: Any, tms_url: str, zoom: int, aoi: str, target: Path) -> None:
    downloaded = Path(
        asyncio.run(
            tms_mod.download_tiles(
                tms=tms_url, zoom=zoom, out=str(target), geojson=aoi, georeference=True, max_failures=0
            )
        )
    )
    for chip in downloaded.iterdir():
        chip.rename(target / chip.name)
    downloaded.rmdir()


@dataclass(frozen=True)
class MaterializedDataset:
    chips_dirs: dict[str, str]
    labels_file: str
    geometry: dict[str, Any]
    bbox: list[float]
    label_properties: list[str]
    label_classes: list[dict[str, Any]]


def materialize_dataset(
    *,
    tms_urls: dict[str, str],
    aoi: dict[str, Any],
    zoom: int,
    out_dir: str,
    geometry_type: str,
    osm_tags: Mapping[str, list[str]] | None = None,
    osm_api_url: str | None = None,
    labels: dict[str, Any] | None = None,
    class_property: str = DEFAULT_CLASS_PROPERTY,
) -> MaterializedDataset:
    """Download chips per input and labels (own `labels`, else OSM features matching `osm_tags`) for `aoi`."""
    tms_mod, osm_mod = _require_geomltoolkits()

    out = Path(out_dir)
    labels_dir = out / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    geometry, bbox = _aoi_geometry_and_bbox(aoi)
    aoi_str = json.dumps(aoi)

    chips_dirs: dict[str, str] = {}
    for input_name, tms_url in tms_urls.items():
        target = out / f"chips-{input_name}"
        target.mkdir(parents=True, exist_ok=True)
        _download_chips(tms_mod, tms_url, zoom, aoi_str, target)
        chips_dirs[input_name] = str(target)

    if labels is not None:
        features = select_labels(labels, geometry, geometry_type, class_property)
        names = [class_property]
    else:
        if not osm_tags or osm_api_url is None:
            raise ValueError("OSM labels need osm_tags and the raw-data API URL")
        osm = asyncio.run(
            osm_mod.download_osm_data(
                geojson=aoi_str,
                api_url=osm_api_url,
                filters=osm_filters(osm_tags, geometry_type),
                geometry_types=[geometry_type],
            )
        )
        features = osm_label_features(osm.get("features", []), osm_tags)
        names = list(osm_tags)
    if not features:
        raise ValueError("no label inside the dataset area")
    label_classes = label_classes_of(features, names)
    labels_file = labels_dir / "labels.geojson"
    labels_file.write_text(feature_collection_lines(features))

    logger.info("Materialized %d labels -> %s, chips -> %s", len(features), labels_file, chips_dirs)
    return MaterializedDataset(
        chips_dirs=chips_dirs,
        labels_file=str(labels_file),
        geometry=geometry,
        bbox=bbox,
        label_properties=[cls["name"] for cls in label_classes],
        label_classes=label_classes,
    )
