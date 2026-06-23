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


def derive_location_props(properties: Mapping[str, Any], bbox: Sequence[float]) -> dict[str, Any]:
    props: dict[str, Any] = {"fair:coverage": coverage_from_bbox(bbox)}
    preview = properties.get("fair:preview_location")
    if preview is not None:
        lon, lat = preview["coordinates"][:2]
        place = reverse_geocode.get((lat, lon))
        props["fair:preview_place"] = place["city"]
        props["fair:preview_country"] = place["country"]
        props["fair:preview_country_code"] = place["country_code"]
    return props
