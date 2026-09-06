"""Derive display-location props from a model item's preview point and footprint."""

from collections.abc import Mapping, Sequence
from typing import Any

import reverse_geocode

_GLOBAL_MIN_SPAN_LON = 359.0
_GLOBAL_MIN_SPAN_LAT = 179.0


def _is_global(bbox: Sequence[float]) -> bool:
    return (bbox[2] - bbox[0]) >= _GLOBAL_MIN_SPAN_LON and (bbox[3] - bbox[1]) >= _GLOBAL_MIN_SPAN_LAT


def coverage_from_bbox(bbox: Sequence[float]) -> str:
    if _is_global(bbox):
        return "global"
    lon = (bbox[0] + bbox[2]) / 2
    lat = (bbox[1] + bbox[3]) / 2
    return reverse_geocode.get((lat, lon))["country"]


def place_from_center(lon: float, lat: float) -> dict[str, str]:
    place = reverse_geocode.get((lat, lon))
    return {"name": place["city"], "country": place["country"], "country_code": place["country_code"]}


def derive_location_props(properties: Mapping[str, Any], bbox: Sequence[float]) -> dict[str, Any]:
    props: dict[str, Any] = {"fair:coverage": coverage_from_bbox(bbox)}
    center = _center_from_properties(properties)
    if center is not None:
        place = place_from_center(*center)
        props["fair:preview_place"] = place["name"]
        props["fair:preview_country"] = place["country"]
        props["fair:preview_country_code"] = place["country_code"]
    return props


def _center_from_properties(properties: Mapping[str, Any]) -> tuple[float, float] | None:
    preview = properties.get("fair:preview")
    if isinstance(preview, dict) and isinstance(preview.get("center"), (list, tuple)):
        lon, lat = preview["center"][:2]
        return lon, lat
    # Fall back to the legacy point for items that predate fair:preview.
    legacy = properties.get("fair:preview_location")
    if isinstance(legacy, dict) and isinstance(legacy.get("coordinates"), (list, tuple)):
        lon, lat = legacy["coordinates"][:2]
        return lon, lat
    return None
