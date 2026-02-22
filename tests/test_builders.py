from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pystac
import pytest

from fair_models.stac.builders import (
    _infer_runtime_media_type,
    build_base_model_item,
    build_dataset_item,
    build_local_model_item,
)

_DT = datetime(2024, 1, 1, tzinfo=UTC)
_GEOM = {"type": "Polygon", "coordinates": [[[0, -90], [180, -90], [180, 90], [0, 90], [0, -90]]]}
_MLM_INPUT = [
    {
        "name": "RGB chips",
        "bands": [{"name": "red"}, {"name": "green"}, {"name": "blue"}],
        "input": {
            "shape": [-1, 3, 512, 512],
            "dim_order": ["batch", "bands", "height", "width"],
            "data_type": "float32",
        },
        "pre_processing_function": {"format": "python", "expression": "mod:preprocess"},
    }
]
_MLM_OUTPUT = [
    {
        "name": "mask",
        "tasks": ["semantic-segmentation"],
        "result": {
            "shape": [-1, 2, 512, 512],
            "dim_order": ["batch", "channel", "height", "width"],
            "data_type": "float32",
        },
        "classification:classes": [{"name": "background", "value": 0}, {"name": "building", "value": 1}],
        "post_processing_function": {"format": "python", "expression": "mod:postprocess"},
    }
]


@pytest.fixture()
def geojson_path(tmp_path):
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[85.51, 27.63], [85.53, 27.63], [85.53, 27.64], [85.51, 27.64], [85.51, 27.63]]],
                },
                "properties": {},
            }
        ],
    }
    path = tmp_path / "labels.geojson"
    path.write_text(json.dumps(geojson))
    return str(path)


def _base_model(**kw: Any) -> pystac.Item:
    return build_base_model_item(
        item_id=kw.get("item_id", "example-unet"),
        geometry=kw.get("geometry", _GEOM),
        dt=kw.get("dt", _DT),
        mlm_name=kw.get("mlm_name", "example-unet"),
        mlm_architecture=kw.get("mlm_architecture", "UNet"),
        mlm_tasks=kw.get("mlm_tasks", ["semantic-segmentation"]),
        mlm_framework=kw.get("mlm_framework", "pytorch"),
        mlm_framework_version=kw.get("mlm_framework_version", "2.1.0"),
        mlm_input=kw.get("mlm_input", _MLM_INPUT),
        mlm_output=kw.get("mlm_output", _MLM_OUTPUT),
        mlm_hyperparameters=kw.get("mlm_hyperparameters", {"epochs": 15, "batch_size": 4, "learning_rate": 0.0001}),
        keywords=kw.get("keywords", ["building", "semantic-segmentation", "polygon"]),
        model_href=kw.get("model_href", "weights.pt"),
        model_artifact_type=kw.get("model_artifact_type", "pt"),
        mlm_pretrained=kw.get("mlm_pretrained", True),
        mlm_pretrained_source=kw.get("mlm_pretrained_source", "OAM-TCD"),
        source_code_href=kw.get("source_code_href", "https://github.com/example"),
        source_code_entrypoint=kw.get("source_code_entrypoint", "mod:train"),
        training_runtime_href=kw.get("training_runtime_href", "local"),
        inference_runtime_href=kw.get("inference_runtime_href", "local"),
    )


class TestBuildDatasetItem:
    def test_properties_assets_bbox(self, geojson_path):
        item = build_dataset_item(
            item_id="ds",
            dt=_DT,
            label_type="vector",
            label_tasks=["segmentation"],
            label_classes=[{"name": "building", "classes": ["building"]}],
            keywords=["building"],
            chips_href="chips/",
            labels_href=geojson_path,
            download_href="s3://bucket/data.zip",
        )
        assert item.properties["label:type"] == "vector"
        assert item.assets["chips"].href == "chips/"
        assert item.assets["download"].media_type == "application/zip"
        bbox = item.bbox
        assert bbox is not None
        assert pytest.approx(bbox[0], abs=0.01) == 85.51

    def test_empty_geojson_raises(self, tmp_path):
        path = tmp_path / "empty.geojson"
        path.write_text('{"type": "FeatureCollection", "features": []}')
        with pytest.raises(ValueError, match="No coordinates"):
            build_dataset_item(
                item_id="e",
                dt=_DT,
                label_type="vector",
                label_tasks=["segmentation"],
                label_classes=[],
                keywords=["building"],
                chips_href="c/",
                labels_href=str(path),
            )


class TestBuildBaseModelItem:
    def test_mlm_fields_and_assets(self):
        item = _base_model()
        assert item.properties["mlm:name"] == "example-unet"
        assert item.properties["mlm:hyperparameters"]["epochs"] == 15
        assert item.properties["version"] == "1"
        assert item.assets["model"].extra_fields["mlm:artifact_type"] == "pt"
        bbox = item.bbox
        assert bbox is not None
        assert bbox[0] == 0 and bbox[2] == 180

    def test_framework_in_media_type(self):
        item = _base_model(mlm_framework="tensorflow")
        media_type = item.assets["model"].media_type
        assert media_type is not None
        assert "framework=tensorflow" in media_type


class TestBuildLocalModelItem:
    def test_inherits_base_and_overrides(self):
        base = _base_model()
        local = build_local_model_item(
            base_model_item=base,
            item_id="local-v1",
            dt=_DT,
            model_href="finetuned.pth",
            mlm_hyperparameters={"epochs": 1},
            keywords=["building"],
            base_model_item_id="example-unet",
            dataset_item_id="ds-1",
            version="2",
            predecessor_version_item_id="local-v0",
        )
        assert local.properties["mlm:architecture"] == "UNet"
        assert local.assets["model"].href == "finetuned.pth"
        assert local.assets["source-code"].href == base.assets["source-code"].href
        derived = {lnk.target for lnk in local.links if lnk.rel == "derived_from"}
        assert "example-unet" in derived and "ds-1" in derived
        assert any(lnk.rel == "predecessor-version" for lnk in local.links)

    def test_none_geometry_raises(self):
        base = _base_model()
        base.geometry = None
        with pytest.raises(ValueError, match="geometry must be provided"):
            build_local_model_item(
                base_model_item=base,
                item_id="bad",
                dt=_DT,
                model_href="w.pth",
                mlm_hyperparameters={},
                keywords=["building"],
                base_model_item_id="b",
                dataset_item_id="d",
                version="1",
            )


@pytest.mark.parametrize(
    ("href", "expected"),
    [
        ("ghcr.io/hotosm/fair:v1", "application/vnd.oci.image.index.v1+json"),
        ("docker.io/library/python:3.12", "application/vnd.oci.image.index.v1+json"),
        ("models/ramp/Dockerfile", "text/x-dockerfile"),
        ("local", "text/plain"),
    ],
)
def test_infer_runtime_media_type(href, expected):
    assert _infer_runtime_media_type(href) == expected
