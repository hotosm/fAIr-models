"""Local dataset materialization for the `fair dataset build` CLI command.

Mirrors the backend's server-side build (download imagery chips + OSM labels for
an AOI) so an operator can produce a draft dataset STAC item from a TMS URL and a
GeoJSON AOI, edit it, and register it. The download itself uses geomltoolkits,
which lives in the `serve` optional extra.
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fair.stac.builders import _bbox_from_coords, _flatten_coords
from fair.stac.constants import LABEL_CLASS_PROPERTY
from fair.stac.validators import GEOMETRY_GEOJSON_TYPES, accepted_label_geometries

logger = logging.getLogger(__name__)

PRE_INPUT = "pre"
BACKGROUND_LABEL = 0
SLIM_PROPERTIES = {LABEL_CLASS_PROPERTY, "tags"}


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


def _osm_filters(label_classes: list[dict[str, Any]], geometry_type: str) -> dict[str, Any]:
    """Translate STAC `label:classes` into a raw-data API `filters` block.

    Classes that share a tag key (one class per `roof:material` value, say) are merged under that key, and `["*"]`
    on any of them asks for every value of the key, which the API spells as an empty list.
    """
    join_or: dict[str, list[str]] = {}
    any_value: set[str] = set()
    for cls in label_classes:
        values = join_or.setdefault(cls["name"], [])
        if cls["classes"] == ["*"]:
            any_value.add(cls["name"])
        else:
            values.extend(value for value in cls["classes"] if value not in values)
    for key in any_value:
        join_or[key] = []
    return {"tags": {geometry_type: {"join_or": join_or}}}


def _stamp_class_label(feature: dict[str, Any], label_classes: list[dict[str, Any]]) -> int | None:
    tags = (feature.get("properties") or {}).get("tags") or {}
    for index, cls in enumerate(label_classes, start=1):
        key = cls["name"]
        values = cls["classes"]
        if key in tags and (values == ["*"] or tags[key] in values):
            value = cls.get("value", index)
            feature.setdefault("properties", {})[LABEL_CLASS_PROPERTY] = value
            return value
    return None


@dataclass(frozen=True)
class PreviewSample:
    """Build parameters for a model's sample dataset, read from its `fair:preview`."""

    tms_urls: dict[str, str]
    zoom: int
    aoi: dict[str, Any]
    tasks: list[str]
    geometry_types: list[str]


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
    return PreviewSample(tms_urls, zoom, aoi, list(properties["mlm:tasks"]), geometry_types)


def select_labels(
    labels: dict[str, Any], aoi_geometry: dict[str, Any], geometry_type: str, label_classes: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Features of `labels` inside the AOI; raise when a geometry type or class value does not match."""
    from shapely.geometry import shape

    area = shape(aoi_geometry)
    allowed_values = {BACKGROUND_LABEL} | {cls.get("value", index) for index, cls in enumerate(label_classes, start=1)}
    selected: list[dict[str, Any]] = []
    problems: list[str] = []
    for position, feature in enumerate(labels.get("features", [])):
        geometry = feature.get("geometry")
        if geometry is None or not shape(geometry).intersects(area):
            continue
        if geometry["type"] not in GEOMETRY_GEOJSON_TYPES[geometry_type]:
            problems.append(f"feature {position}: {geometry['type']} is not a {geometry_type}")
        value = (feature.get("properties") or {}).get(LABEL_CLASS_PROPERTY)
        if value not in allowed_values:
            problems.append(f"feature {position}: {LABEL_CLASS_PROPERTY}={value!r} not in {sorted(allowed_values)}")
        selected.append(feature)
    if problems:
        raise ValueError("labels do not match the dataset:\n" + "\n".join(problems[:20]))
    if not selected:
        raise ValueError("no label inside the dataset area")
    return selected


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


def materialize_dataset(
    *,
    tms_urls: dict[str, str],
    aoi: dict[str, Any],
    zoom: int,
    out_dir: str,
    label_classes: list[dict[str, Any]],
    geometry_type: str,
    osm_api_url: str | None,
    labels: dict[str, Any] | None = None,
    slim_labels: bool = False,
) -> tuple[dict[str, str], str, dict[str, Any], list[float]]:
    """Download chips per input and labels (own `labels`, else OSM) for `aoi`; return chips, labels, geometry, bbox."""
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
        features = select_labels(labels, geometry, geometry_type, label_classes)
    else:
        if osm_api_url is None:
            raise ValueError("OSM labels need the raw-data API URL")
        osm = asyncio.run(
            osm_mod.download_osm_data(
                geojson=aoi_str,
                api_url=osm_api_url,
                filters=_osm_filters(label_classes, geometry_type),
                geometry_types=[geometry_type],
            )
        )
        features = [feat for feat in osm.get("features", []) if _stamp_class_label(feat, label_classes) is not None]
    labels_file = labels_dir / "labels.geojson"
    if slim_labels:
        features = [
            {
                "type": "Feature",
                "properties": {key: value for key, value in feature["properties"].items() if key in SLIM_PROPERTIES},
                "geometry": feature["geometry"],
            }
            for feature in features
        ]
    labels_file.write_text(feature_collection_lines(features))

    logger.info("Materialized %d labels -> %s, chips -> %s", len(features), labels_file, chips_dirs)
    return chips_dirs, str(labels_file), geometry, bbox
