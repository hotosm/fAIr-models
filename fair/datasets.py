"""Local dataset materialization for the `fair dataset build` CLI command.

Mirrors the backend's server-side build (download imagery chips + OSM labels for
an AOI) so an operator can produce a draft dataset STAC item from a TMS URL and a
GeoJSON AOI, edit it, and register it. The download itself uses geomltoolkits,
which lives in the `serve` optional extra.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from fair.stac.builders import _bbox_from_coords, _flatten_coords

logger = logging.getLogger(__name__)


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
    """Translate STAC `label:classes` into a raw-data API `filters` block."""
    join_or = {cls["name"]: ([] if cls["classes"] == ["*"] else list(cls["classes"])) for cls in label_classes}
    return {"tags": {geometry_type: {"join_or": join_or}}}


def _stamp_class_label(feature: dict[str, Any], label_classes: list[dict[str, Any]]) -> int | None:
    """Stamp `properties.label = i` for the first matching class (1-based; 0 = background)."""
    tags = (feature.get("properties") or {}).get("tags") or {}
    for index, cls in enumerate(label_classes, start=1):
        key = cls["name"]
        values = cls["classes"]
        if key in tags and (values == ["*"] or tags[key] in values):
            feature.setdefault("properties", {})["label"] = index
            return index
    return None


def materialize_dataset(
    *,
    tms_url: str,
    aoi: dict[str, Any],
    zoom: int,
    out_dir: str,
    label_classes: list[dict[str, Any]],
    geometry_type: str,
    osm_api_url: str,
) -> tuple[str, str, dict[str, Any], list[float]]:
    """Download chips + OSM labels for `aoi`; return (chips_dir, labels_file, geometry, bbox)."""
    tms_mod, osm_mod = _require_geomltoolkits()

    out = Path(out_dir)
    chips_root = out / "chips"
    labels_dir = out / "labels"
    chips_root.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    geometry, bbox = _aoi_geometry_and_bbox(aoi)
    aoi_str = json.dumps(aoi)

    chips_path = asyncio.run(
        tms_mod.download_tiles(tms=tms_url, zoom=zoom, out=str(chips_root), geojson=aoi_str, georeference=True)
    )
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
    labels_file.write_text(json.dumps({"type": "FeatureCollection", "features": features}))

    logger.info("Materialized %d labels -> %s, chips -> %s", len(features), labels_file, chips_path)
    return str(chips_path), str(labels_file), geometry, bbox
