from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import pystac

from fair.stac.constants import (
    BASE_MODEL_EXTENSIONS,
    CONTAINER_REGISTRIES,
    DATASET_EXTENSIONS,
    LOCAL_MODEL_EXTENSIONS,
    OCI_IMAGE_INDEX_TYPE,
)
from fair.stac.versioning import add_version_links


@dataclass
class DatasetItemParams:
    label_type: Literal["vector", "raster"]
    label_tasks: list[str]
    label_classes: list[dict[str, Any]]
    keywords: list[str]
    chips_href: str
    labels_href: str
    title: str
    description: str
    user_id: str
    providers: list[dict[str, Any]]
    item_id: str | None = None
    download_href: str | None = None
    thumbnail_href: str | None = None
    source_imagery: str | None = None
    chip_count: int | None = None
    geometry: dict[str, Any] | None = None
    bbox: list[float] | None = None
    version: str = "1"
    deprecated: bool = False
    license_id: str | None = None
    label_properties: list[str] | None = None
    label_description: str | None = None
    label_methods: list[str] | None = None
    source_imagery_href: str | None = None
    self_href: str | None = None
    predecessor_version_href: str | None = None


CheckpointArtifactType = Literal[
    "torch.save",
    "torch.jit.save",
    "torch.export.save",
    "pickle",
    "tf.keras.Model.save",
    "tf.keras.Model.save_weights",
    "tf.keras.Model.export",
]


@dataclass
class BaseModelItemParams:
    item_id: str
    geometry: dict[str, Any]
    mlm_name: str
    mlm_architecture: str
    mlm_tasks: list[str]
    mlm_framework: str
    mlm_framework_version: str
    mlm_input: list[dict[str, Any]]
    mlm_output: list[dict[str, Any]]
    mlm_hyperparameters: dict[str, Any]
    keywords: list[str]
    checkpoint_href: str
    checkpoint_artifact_type: CheckpointArtifactType
    mlm_pretrained: bool
    mlm_pretrained_source: str | None
    source_code_href: str
    source_code_entrypoint: str
    training_runtime_href: str
    inference_runtime_href: str
    title: str
    description: str
    fair_metrics_spec: list[dict[str, Any]]
    providers: list[dict[str, Any]]
    onnx_href: str
    readme_href: str = ""


@dataclass
class LocalModelItemParams:
    base_model_item: pystac.Item
    checkpoint_href: str
    onnx_href: str
    mlm_hyperparameters: dict[str, Any]
    keywords: list[str]
    base_model_href: str
    dataset_href: str
    version: str
    title: str
    description: str
    user_id: str
    providers: list[dict[str, Any]]
    item_id: str | None = None
    mlm_name: str | None = None
    geometry: dict[str, Any] | None = None
    metrics: dict[str, Any] | None = None
    labeled_chip_count: int | None = None
    thumbnail_href: str | None = None
    predecessor_version_href: str | None = None
    self_href: str | None = None
    zenml_artifact_version_id: str | None = None
    training_started_at: str | None = None
    training_ended_at: str | None = None
    training_duration_seconds: float | None = None
    base_model_id: str | None = None
    dataset_id: str | None = None
    dataset_title: str | None = None
    split_info: dict[str, Any] | None = None


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


def geometry_and_bbox_from_geojson(labels_href: str) -> tuple[dict[str, Any], list[float]]:
    labels_path = Path(labels_href)
    if labels_path.is_dir():
        geojson_files = sorted(labels_path.rglob("*.geojson"))
        if not geojson_files:
            msg = f"No .geojson files found in {labels_href}"
            raise ValueError(msg)
        all_coords: list[list[float]] = []
        for gf in geojson_files:
            all_coords.extend(_extract_coords_from_file(gf))
    else:
        all_coords = _extract_coords_from_file(labels_path)

    if not all_coords:
        msg = f"No coordinates found in {labels_href}"
        raise ValueError(msg)

    west, south, east, north = _bbox_from_coords(all_coords)
    geometry = {
        "type": "Polygon",
        "coordinates": [[[west, south], [east, south], [east, north], [west, north], [west, south]]],
    }
    return geometry, [west, south, east, north]


def _extract_coords_from_file(path: Path) -> list[list[float]]:
    with open(path, encoding="utf-8") as f:
        geojson: Any = json.load(f)

    coords: list[list[float]] = []
    features: Any = geojson.get("features", [])
    if features:
        for feat in features:
            coords.extend(_flatten_coords(feat["geometry"]["coordinates"]))
    elif "coordinates" in geojson:
        coords.extend(_flatten_coords(geojson["coordinates"]))
    return coords


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "-", text.lower().strip()).strip("-")


def _validate_providers(providers: list[dict[str, Any]]) -> None:
    if not providers:
        msg = "providers is required and must be a non-empty list of STAC Provider objects"
        raise ValueError(msg)
    for entry in providers:
        if not isinstance(entry, dict) or not entry.get("name") or not entry.get("roles"):
            msg = "each provider must be a dict with non-empty 'name' and 'roles'"
            raise ValueError(msg)


def _raster_bands_from_model_input(mlm_input: list[dict[str, Any]]) -> list[dict[str, str]] | None:
    seen: set[str] = set()
    bands: list[dict[str, str]] = []
    for model_input in mlm_input:
        for band in model_input.get("bands", []):
            name = band if isinstance(band, str) else band.get("name")
            if not name or name in seen:
                continue
            seen.add(name)
            bands.append({"name": name})
    return bands or None


def build_dataset_item(
    label_type: Literal["vector", "raster"],
    label_tasks: list[str],
    label_classes: list[dict[str, Any]],
    keywords: list[str],
    chips_href: str,
    labels_href: str,
    title: str,
    description: str,
    user_id: str,
    providers: list[dict[str, Any]],
    item_id: str | None = None,
    download_href: str | None = None,
    thumbnail_href: str | None = None,
    source_imagery: str | None = None,
    chip_count: int | None = None,
    geometry: dict[str, Any] | None = None,
    bbox: list[float] | None = None,
    version: str = "1",
    deprecated: bool = False,
    license_id: str | None = None,
    label_properties: list[str] | None = None,
    label_description: str | None = None,
    label_methods: list[str] | None = None,
    source_imagery_href: str | None = None,
    self_href: str | None = None,
    predecessor_version_href: str | None = None,
) -> pystac.Item:
    _validate_providers(providers)

    if geometry is None or bbox is None:
        labels_path = Path(labels_href)
        if labels_path.is_dir() or labels_path.suffix.lower() == ".geojson":
            geometry, bbox = geometry_and_bbox_from_geojson(labels_href)
        else:
            msg = "geometry and bbox are required when labels are not GeoJSON"
            raise ValueError(msg)

    resolved_id = item_id if item_id is not None else _slugify(title)

    resolved_label_properties = (
        label_properties if label_properties is not None else (None if label_type == "raster" else ["class"])
    )

    properties: dict[str, Any] = {
        "title": title,
        "description": description,
        "label:type": label_type,
        "label:tasks": label_tasks,
        "label:classes": label_classes,
        "label:properties": resolved_label_properties,
        "keywords": keywords,
        "fair:user_id": user_id,
        "version": version,
        "deprecated": deprecated,
        "providers": providers,
    }
    if chip_count is not None:
        properties["fair:chip_count"] = chip_count
    if source_imagery is not None:
        properties["fair:source_imagery"] = source_imagery
    if license_id is not None:
        properties["license"] = license_id
    properties["label:description"] = label_description if label_description is not None else description
    if label_methods is not None:
        properties["label:methods"] = label_methods

    now = datetime.now(UTC)
    now_str = now.isoformat()
    properties["created"] = now_str
    properties["updated"] = now_str

    item = pystac.Item(
        id=resolved_id,
        geometry=geometry,
        bbox=bbox,
        datetime=now,
        properties=properties,
        stac_extensions=DATASET_EXTENSIONS,
    )

    if source_imagery_href:
        item.add_link(
            pystac.Link(
                rel="source",
                target=source_imagery_href,
                media_type="image/tiff; application=geotiff",
                title="Source imagery",
            )
        )

    item.add_asset(
        "chips",
        pystac.Asset(
            href=chips_href,
            media_type="image/tiff; application=geotiff",
            roles=["data"],
            title="Training chips",
        ),
    )
    item.add_asset(
        "labels",
        pystac.Asset(
            href=labels_href,
            media_type="application/geo+json",
            roles=["labels"],
            title="Training labels",
        ),
    )

    if download_href:
        item.add_asset(
            "download",
            pystac.Asset(
                href=download_href,
                media_type="application/zip",
                roles=["data", "archive"],
                title="Full dataset archive",
            ),
        )

    if thumbnail_href:
        item.add_asset(
            "thumbnail",
            pystac.Asset(
                href=thumbnail_href,
                media_type="image/png",
                roles=["thumbnail"],
                title="Dataset thumbnail",
            ),
        )

    add_version_links(item, self_href, predecessor_version_href)

    return item


def build_base_model_item(
    item_id: str,
    geometry: dict[str, Any],
    mlm_name: str,
    mlm_architecture: str,
    mlm_tasks: list[str],
    mlm_framework: str,
    mlm_framework_version: str,
    mlm_input: list[dict[str, Any]],
    mlm_output: list[dict[str, Any]],
    mlm_hyperparameters: dict[str, Any],
    keywords: list[str],
    checkpoint_href: str,
    checkpoint_artifact_type: CheckpointArtifactType,
    mlm_pretrained: bool,
    mlm_pretrained_source: str | None,
    source_code_href: str,
    source_code_entrypoint: str,
    training_runtime_href: str,
    inference_runtime_href: str,
    title: str,
    description: str,
    fair_metrics_spec: list[dict[str, Any]],
    providers: list[dict[str, Any]],
    onnx_href: str,
    readme_href: str = "",
) -> pystac.Item:
    _validate_providers(providers)
    bbox = _bbox_from_geometry(geometry)

    now = datetime.now(UTC)
    now_str = now.isoformat()

    item = pystac.Item(
        id=item_id,
        geometry=geometry,
        bbox=bbox,
        datetime=now,
        properties={
            "title": title,
            "description": description,
            "created": now_str,
            "updated": now_str,
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
            "deprecated": False,
            "fair:metrics_spec": fair_metrics_spec,
            "providers": providers,
        },
        stac_extensions=BASE_MODEL_EXTENSIONS,
    )

    asset_bands = _raster_bands_from_model_input(mlm_input)
    checkpoint_extra = {"mlm:artifact_type": checkpoint_artifact_type}
    if asset_bands:
        checkpoint_extra["raster:bands"] = asset_bands
    item.add_asset(
        "checkpoint",
        pystac.Asset(
            href=checkpoint_href,
            media_type=f"application/octet-stream; framework={mlm_framework}",
            roles=["mlm:model", "mlm:weights"],
            extra_fields=checkpoint_extra,
        ),
    )
    model_extra = {"mlm:artifact_type": "onnx"}
    if asset_bands:
        model_extra["raster:bands"] = asset_bands
    item.add_asset(
        "model",
        pystac.Asset(
            href=onnx_href,
            media_type="application/octet-stream; framework=onnx",
            roles=["mlm:model", "mlm:compiled"],
            extra_fields=model_extra,
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
    if readme_href:
        item.add_asset(
            "readme",
            pystac.Asset(
                href=readme_href,
                media_type="text/markdown",
                roles=["metadata"],
                title="Model README",
            ),
        )

    return item


def build_local_model_item(
    base_model_item: pystac.Item,
    checkpoint_href: str,
    onnx_href: str,
    mlm_hyperparameters: dict[str, Any],
    keywords: list[str],
    base_model_href: str,
    dataset_href: str,
    version: str,
    title: str,
    description: str,
    user_id: str,
    providers: list[dict[str, Any]],
    item_id: str | None = None,
    mlm_name: str | None = None,
    geometry: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    labeled_chip_count: int | None = None,
    thumbnail_href: str | None = None,
    predecessor_version_href: str | None = None,
    self_href: str | None = None,
    zenml_artifact_version_id: str | None = None,
    training_started_at: str | None = None,
    training_ended_at: str | None = None,
    training_duration_seconds: float | None = None,
    base_model_id: str | None = None,
    dataset_id: str | None = None,
    dataset_title: str | None = None,
    split_info: dict[str, Any] | None = None,
    training_metrics_href: str | None = None,
) -> pystac.Item:
    _validate_providers(providers)
    geom = geometry if geometry is not None else base_model_item.geometry
    if geom is None:
        msg = "geometry must be provided when base_model_item.geometry is None"
        raise ValueError(msg)
    bbox = _bbox_from_geometry(geom)

    resolved_id = item_id if item_id is not None else _slugify(title)
    base_props = base_model_item.properties

    properties: dict[str, Any] = {
        "title": title,
        "description": description,
        "mlm:name": mlm_name if mlm_name is not None else base_props.get("mlm:name"),
        "mlm:architecture": base_props.get("mlm:architecture"),
        "mlm:tasks": base_props.get("mlm:tasks"),
        "mlm:framework": base_props.get("mlm:framework"),
        "mlm:framework_version": base_props.get("mlm:framework_version"),
        "mlm:pretrained": True,
        "mlm:pretrained_source": base_model_href,
        "mlm:input": base_props.get("mlm:input"),
        "mlm:output": base_props.get("mlm:output"),
        "mlm:hyperparameters": mlm_hyperparameters,
        "keywords": keywords,
        "version": version,
        "deprecated": False,
        "fair:user_id": user_id,
        "providers": providers,
    }

    if base_model_id is not None:
        properties["fair:base_model_id"] = base_model_id
    if dataset_id is not None:
        properties["fair:dataset_id"] = dataset_id

    # Copy fields from base model that apply to the finetuned variant
    for field in ("license", "mlm:accelerator", "mlm:accelerator_count", "fair:metrics_spec"):
        if field in base_props:
            properties[field] = base_props[field]

    if metrics:
        properties.update(metrics)

    if training_started_at:
        properties["fair:training_started_at"] = training_started_at
    if training_ended_at:
        properties["fair:training_ended_at"] = training_ended_at
    if training_duration_seconds is not None:
        properties["fair:training_duration_seconds"] = training_duration_seconds

    if labeled_chip_count is not None:
        properties["fair:labeled_chip_count"] = labeled_chip_count
    if split_info:
        properties["fair:split"] = split_info

    now = datetime.now(UTC)
    now_str = now.isoformat()
    properties["created"] = now_str
    properties["updated"] = now_str

    item = pystac.Item(
        id=resolved_id,
        geometry=geom,
        bbox=bbox,
        datetime=now,
        properties=properties,
        stac_extensions=LOCAL_MODEL_EXTENSIONS,
    )

    base_mlm_name = base_props.get("mlm:name", "")
    item.add_link(
        pystac.Link(
            rel="derived_from",
            target=base_model_href,
            media_type="application/geo+json",
            extra_fields={"mlm:name": base_mlm_name},
        )
    )
    dataset_link_extra: dict[str, Any] = {}
    if dataset_title:
        dataset_link_extra["title"] = dataset_title
    item.add_link(
        pystac.Link(
            rel="derived_from",
            target=dataset_href,
            media_type="application/geo+json",
            extra_fields=dataset_link_extra,
        )
    )
    add_version_links(item, self_href, predecessor_version_href)

    base_framework = base_model_item.properties.get("mlm:framework", "PyTorch")
    base_checkpoint = base_model_item.assets.get("checkpoint")
    checkpoint_artifact_type = (
        base_checkpoint.extra_fields.get("mlm:artifact_type", "torch.save") if base_checkpoint else "torch.save"
    )

    asset_bands = _raster_bands_from_model_input(base_props.get("mlm:input", []))

    checkpoint_extra: dict[str, Any] = {"mlm:artifact_type": checkpoint_artifact_type}
    if zenml_artifact_version_id:
        checkpoint_extra["zenml:artifact_version_id"] = zenml_artifact_version_id
    if asset_bands:
        checkpoint_extra["raster:bands"] = asset_bands
    item.add_asset(
        "checkpoint",
        pystac.Asset(
            href=checkpoint_href,
            media_type=f"application/octet-stream; framework={base_framework}",
            roles=["mlm:model", "mlm:weights"],
            extra_fields=checkpoint_extra,
        ),
    )

    onnx_extra: dict[str, Any] = {"mlm:artifact_type": "onnx"}
    if zenml_artifact_version_id:
        onnx_extra["zenml:artifact_version_id"] = zenml_artifact_version_id
    if asset_bands:
        onnx_extra["raster:bands"] = asset_bands
    item.add_asset(
        "model",
        pystac.Asset(
            href=onnx_href,
            media_type="application/octet-stream; framework=onnx",
            roles=["mlm:model", "mlm:compiled"],
            extra_fields=onnx_extra,
        ),
    )

    for key in ("source-code", "mlm:training", "mlm:inference"):
        if key in base_model_item.assets:
            item.add_asset(key, base_model_item.assets[key].clone())

    if thumbnail_href:
        item.add_asset(
            "thumbnail",
            pystac.Asset(
                href=thumbnail_href,
                media_type="image/png",
                roles=["thumbnail"],
                title="Model thumbnail",
            ),
        )

    if training_metrics_href:
        item.add_asset(
            "training-metrics",
            pystac.Asset(
                href=training_metrics_href,
                media_type="application/json",
                roles=["metadata"],
                title="Per-epoch training metrics",
            ),
        )

    return item
