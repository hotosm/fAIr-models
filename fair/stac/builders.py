from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Literal

import pystac

from fair.stac.constants import (
    CONTAINER_REGISTRIES,
    DATASET_EXTENSIONS,
    MODEL_EXTENSIONS,
    OCI_IMAGE_INDEX_TYPE,
)

_SOURCE_CODE_EXTENSIONS = {
    ".py": "text/x-python",
    ".js": "text/javascript",
}


def _infer_source_code_media_type(href: str) -> str:
    lower = href.lower()
    for ext, mime in _SOURCE_CODE_EXTENSIONS.items():
        if lower.endswith(ext):
            return mime
    if any(host in lower for host in ("github.com", "gitlab.com", "bitbucket.org")):
        return "text/html"
    return "text/plain"


def _infer_runtime_media_type(href: str) -> str:
    lower = href.lower()
    if any(r in lower for r in CONTAINER_REGISTRIES):
        return OCI_IMAGE_INDEX_TYPE
    if "dockerfile" in lower:
        return "text/x-dockerfile"
    return "text/plain"


def _bbox_from_coords(coords: list[list[float]]) -> list[float]:
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    return [min(lons), min(lats), max(lons), max(lats)]


def _bbox_from_geometry(geometry: dict[str, Any]) -> list[float]:
    return _bbox_from_coords(_flatten_coords(geometry["coordinates"]))


def _flatten_coords(coords: Any) -> list[list[float]]:
    if isinstance(coords[0], (int, float)):
        return [coords]
    result: list[list[float]] = []
    for c in coords:
        result.extend(_flatten_coords(c))
    return result


def _geometry_and_bbox_from_geojson(labels_href: str) -> tuple[dict[str, Any], list[float]]:
    with open(labels_href, encoding="utf-8") as f:
        geojson: Any = json.load(f)

    all_coords: list[list[float]] = []
    features: Any = geojson.get("features", [])
    if features:
        for feat in features:
            all_coords.extend(_flatten_coords(feat["geometry"]["coordinates"]))
    elif "coordinates" in geojson:
        all_coords.extend(_flatten_coords(geojson["coordinates"]))

    if not all_coords:
        msg = f"No coordinates found in {labels_href}"
        raise ValueError(msg)

    west, south, east, north = _bbox_from_coords(all_coords)
    geometry = {
        "type": "Polygon",
        "coordinates": [[[west, south], [east, south], [east, north], [west, north], [west, south]]],
    }
    return geometry, [west, south, east, north]


def build_dataset_item(
    item_id: str,
    dt: datetime,
    label_type: Literal["vector", "raster"],
    label_tasks: list[str],
    label_classes: list[dict[str, Any]],
    keywords: list[str],
    chips_href: str,
    labels_href: str,
    download_href: str | None = None,
) -> pystac.Item:
    geometry, bbox = _geometry_and_bbox_from_geojson(labels_href)

    item = pystac.Item(
        id=item_id,
        geometry=geometry,
        bbox=bbox,
        datetime=dt,
        properties={
            "label:type": label_type,
            "label:tasks": label_tasks,
            "label:classes": label_classes,
            "keywords": keywords,
        },
        stac_extensions=DATASET_EXTENSIONS,
    )

    item.add_asset(
        "chips",
        pystac.Asset(
            href=chips_href,
            media_type="image/png",
            roles=["data"],
        ),
    )
    item.add_asset(
        "labels",
        pystac.Asset(
            href=labels_href,
            media_type="application/geo+json",
            roles=["labels"],
        ),
    )

    if download_href:
        item.add_asset(
            "download",
            pystac.Asset(
                href=download_href,
                media_type="application/zip",
                roles=["data", "archive"],
            ),
        )

    return item


def build_base_model_item(
    item_id: str,
    geometry: dict[str, Any],
    dt: datetime,
    mlm_name: str,
    mlm_architecture: str,
    mlm_tasks: list[str],
    mlm_framework: str,
    mlm_framework_version: str,
    mlm_input: list[dict[str, Any]],
    mlm_output: list[dict[str, Any]],
    mlm_hyperparameters: dict[str, Any],
    keywords: list[str],
    model_href: str,
    model_artifact_type: Literal[
        "torch.save",
        "torch.jit.save",
        "torch.export.save",
        "onnx",
        "pickle",
        "tf.keras.Model.save",
        "tf.keras.Model.save_weights",
        "tf.keras.Model.export",
    ],
    mlm_pretrained: bool,
    mlm_pretrained_source: str | None,
    source_code_href: str,
    source_code_entrypoint: str,
    training_runtime_href: str,
    inference_runtime_href: str,
) -> pystac.Item:
    bbox = _bbox_from_geometry(geometry)

    item = pystac.Item(
        id=item_id,
        geometry=geometry,
        bbox=bbox,
        datetime=dt,
        properties={
            "mlm:name": mlm_name,
            "mlm:architecture": mlm_architecture,
            "mlm:tasks": mlm_tasks,
            "mlm:framework": mlm_framework,
            "mlm:framework_version": mlm_framework_version,
            "mlm:pretrained": mlm_pretrained,
            "mlm:pretrained_source": mlm_pretrained_source,
            "mlm:input": mlm_input,
            "mlm:output": mlm_output,
            "mlm:hyperparameters": mlm_hyperparameters,
            "keywords": keywords,
            "version": "1",
        },
        stac_extensions=MODEL_EXTENSIONS,
    )

    item.add_asset(
        "model",
        pystac.Asset(
            href=model_href,
            media_type=f"application/octet-stream; framework={mlm_framework}",
            roles=["mlm:model"],
            extra_fields={"mlm:artifact_type": model_artifact_type},
        ),
    )
    item.add_asset(
        "source-code",
        pystac.Asset(
            href=source_code_href,
            media_type=_infer_source_code_media_type(source_code_href),
            roles=["code"],
            extra_fields={"mlm:entrypoint": source_code_entrypoint},
        ),
    )
    item.add_asset(
        "mlm:training",
        pystac.Asset(
            href=training_runtime_href,
            media_type=_infer_runtime_media_type(training_runtime_href),
            roles=["mlm:training-runtime"],
        ),
    )
    item.add_asset(
        "mlm:inference",
        pystac.Asset(
            href=inference_runtime_href,
            media_type=_infer_runtime_media_type(inference_runtime_href),
            roles=["mlm:inference-runtime"],
        ),
    )

    return item


def build_local_model_item(
    base_model_item: pystac.Item,
    item_id: str,
    dt: datetime,
    model_href: str,
    mlm_hyperparameters: dict[str, Any],
    keywords: list[str],
    base_model_item_id: str,
    dataset_item_id: str,
    version: str,
    geometry: dict[str, Any] | None = None,
    predecessor_version_item_id: str | None = None,
    zenml_artifact_version_id: str | None = None,
) -> pystac.Item:
    geom = geometry if geometry is not None else base_model_item.geometry
    if geom is None:
        msg = "geometry must be provided when base_model_item.geometry is None"
        raise ValueError(msg)
    bbox = _bbox_from_geometry(geom)

    base_props = base_model_item.properties
    item = pystac.Item(
        id=item_id,
        geometry=geom,
        bbox=bbox,
        datetime=dt,
        properties={
            "mlm:name": base_props.get("mlm:name"),
            "mlm:architecture": base_props.get("mlm:architecture"),
            "mlm:tasks": base_props.get("mlm:tasks"),
            "mlm:framework": base_props.get("mlm:framework"),
            "mlm:framework_version": base_props.get("mlm:framework_version"),
            "mlm:pretrained": True,
            "mlm:pretrained_source": base_model_item_id,
            "mlm:input": base_props.get("mlm:input"),
            "mlm:output": base_props.get("mlm:output"),
            "mlm:hyperparameters": mlm_hyperparameters,
            "keywords": keywords,
            "version": version,
            "deprecated": False,
        },
        stac_extensions=MODEL_EXTENSIONS,
    )

    item.add_link(pystac.Link(rel="derived_from", target=base_model_item_id))
    item.add_link(pystac.Link(rel="derived_from", target=dataset_item_id))
    item.add_link(pystac.Link(rel="latest-version", target=item_id))
    if predecessor_version_item_id:
        item.add_link(pystac.Link(rel="predecessor-version", target=predecessor_version_item_id))

    for key, asset in base_model_item.assets.items():
        if key == "model":
            model_extra = {k: v for k, v in asset.extra_fields.items() if k != "href"}
            if zenml_artifact_version_id:
                model_extra["zenml:artifact_version_id"] = zenml_artifact_version_id
            item.add_asset(
                "model",
                pystac.Asset(
                    href=model_href,
                    media_type=asset.media_type,
                    roles=asset.roles,
                    extra_fields=model_extra,
                ),
            )
        else:
            item.add_asset(key, asset.clone())

    return item
