from __future__ import annotations

import json
from typing import Any

import pystac
import pytest

from fair.stac.builders import (
    _infer_runtime_media_type,
    _slugify,
    build_base_model_item,
    build_dataset_item,
    build_local_model_item,
)

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


_METRICS_SPEC = [
    {"name": "accuracy", "description": "Pixel accuracy", "higher_is_better": True},
]

_BASE_DEFAULTS: dict[str, Any] = {
    "item_id": "example-unet",
    "geometry": _GEOM,
    "mlm_name": "example-unet",
    "mlm_architecture": "UNet",
    "mlm_tasks": ["semantic-segmentation"],
    "mlm_framework": "pytorch",
    "mlm_framework_version": "2.1.0",
    "mlm_input": _MLM_INPUT,
    "mlm_output": _MLM_OUTPUT,
    "mlm_hyperparameters": {"epochs": 15, "batch_size": 4, "learning_rate": 0.0001},
    "keywords": ["building", "semantic-segmentation", "polygon"],
    "checkpoint_href": "https://example.com/weights.pt",
    "checkpoint_artifact_type": "torch.save",
    "mlm_pretrained": True,
    "mlm_pretrained_source": "OAM-TCD",
    "source_code_href": "https://github.com/example",
    "source_code_entrypoint": "mod:train",
    "training_runtime_href": "local",
    "inference_runtime_href": "local",
    "title": "Example UNet",
    "description": "A UNet model for building segmentation.",
    "fair_metrics_spec": _METRICS_SPEC,
}


def _base_model(**kw: Any) -> pystac.Item:
    return build_base_model_item(**{**_BASE_DEFAULTS, **kw})


class TestBuildDatasetItem:
    def test_properties_assets_bbox(self, geojson_path):
        item = build_dataset_item(
            label_type="vector",
            label_tasks=["segmentation"],
            label_classes=[{"name": "building", "classes": ["building"]}],
            keywords=["building"],
            chips_href="chips/",
            labels_href=geojson_path,
            title="Test Dataset",
            description="Chips for testing.",
            user_id="osm-user-42",
            download_href="s3://bucket/data.zip",
        )
        assert item.properties["label:type"] == "vector"
        assert item.properties["title"] == "Test Dataset"
        assert item.properties["fair:user_id"] == "osm-user-42"
        assert item.assets["chips"].href == "chips/"
        assert item.assets["download"].media_type == "application/zip"
        bbox = item.bbox
        assert bbox is not None
        assert pytest.approx(bbox[0], abs=0.01) == 85.51

    def test_slug_generated_when_item_id_none(self, geojson_path):
        item = build_dataset_item(
            label_type="vector",
            label_tasks=["segmentation"],
            label_classes=[],
            keywords=["building"],
            chips_href="c/",
            labels_href=geojson_path,
            title="Buildings Banepa",
            description="D",
            user_id="u",
        )
        assert item.id == "buildings-banepa"

    def test_empty_geojson_raises(self, tmp_path):
        path = tmp_path / "empty.geojson"
        path.write_text('{"type": "FeatureCollection", "features": []}')
        with pytest.raises(ValueError, match="No coordinates"):
            build_dataset_item(
                label_type="vector",
                label_tasks=["segmentation"],
                label_classes=[],
                keywords=["building"],
                chips_href="c/",
                labels_href=str(path),
                title="T",
                description="D",
                user_id="u",
            )

    def test_versioning_and_enriched_metadata(self, geojson_path):
        providers = [{"name": "HOTOSM", "roles": ["producer"], "url": "https://www.hotosm.org"}]
        item = build_dataset_item(
            label_type="vector",
            label_tasks=["segmentation"],
            label_classes=[],
            keywords=["building"],
            chips_href="c/",
            labels_href=geojson_path,
            title="T",
            description="D",
            user_id="u",
            version="2",
            deprecated=False,
            license_id="CC-BY-4.0",
            providers=providers,
            label_description="Manually labeled buildings",
            label_methods=["manual"],
            source_imagery_href="https://tiles.oam.org/{z}/{x}/{y}",
            self_href="https://api/collections/datasets/items/t",
            predecessor_version_href="https://api/collections/datasets/items/t-v1",
        )
        assert item.properties["version"] == "2"
        assert item.properties["deprecated"] is False
        assert item.properties["license"] == "CC-BY-4.0"
        assert item.properties["providers"] == providers
        assert item.properties["label:description"] == "Manually labeled buildings"
        assert item.properties["label:methods"] == ["manual"]
        source_links = [lnk for lnk in item.links if lnk.rel == "source"]
        assert len(source_links) == 1
        assert source_links[0].get_href() == "https://tiles.oam.org/{z}/{x}/{y}"
        pred = [lnk for lnk in item.links if lnk.rel == "predecessor-version"]
        assert len(pred) == 1
        self_links = [lnk for lnk in item.links if lnk.rel == "self"]
        assert len(self_links) == 1


class TestBuildBaseModelItem:
    def test_mlm_fields_and_assets(self):
        item = _base_model()
        assert item.properties["mlm:name"] == "example-unet"
        assert item.properties["mlm:hyperparameters"]["epochs"] == 15
        assert item.properties["version"] == "1"
        assert item.properties["title"] == "Example UNet"
        assert item.properties["fair:metrics_spec"] == _METRICS_SPEC
        assert item.assets["checkpoint"].extra_fields["mlm:artifact_type"] == "torch.save"
        assert "readme" not in item.assets  # readme_href defaults to empty string
        bbox = item.bbox
        assert bbox is not None
        assert bbox[0] == 0 and bbox[2] == 180

    def test_readme_asset_present_when_href_given(self):
        item = _base_model(readme_href="https://example.com/README.md")
        assert "readme" in item.assets
        assert item.assets["readme"].media_type == "text/markdown"

    def test_framework_in_media_type(self):
        item = _base_model(mlm_framework="tensorflow")
        media_type = item.assets["checkpoint"].media_type
        assert media_type is not None
        assert "framework=tensorflow" in media_type


class TestBuildLocalModelItem:
    def test_inherits_base_and_overrides(self):
        base = _base_model()
        local = build_local_model_item(
            base_model_item=base,
            item_id="local-v1",
            checkpoint_href="https://example.com/finetuned.pt",
            onnx_href="https://example.com/finetuned.onnx",
            mlm_hyperparameters={"epochs": 1},
            keywords=["building"],
            base_model_href="../base-models/example-unet/example-unet.json",
            dataset_href="../datasets/ds-1/ds-1.json",
            version="2",
            title="Local UNet v2",
            description="Finetuned model.",
            user_id="osm-42",
            predecessor_version_href="../local-models/local-v0/local-v0.json",
            base_model_id="example-unet",
            dataset_id="ds-1",
            dataset_title="Buildings Banepa",
        )
        assert local.properties["mlm:architecture"] == "UNet"
        assert local.properties["title"] == "Local UNet v2"
        assert local.properties["fair:user_id"] == "osm-42"
        assert local.properties["fair:base_model_id"] == "example-unet"
        assert local.properties["fair:dataset_id"] == "ds-1"
        assert local.assets["checkpoint"].href == "https://example.com/finetuned.pt"
        assert local.assets["model"].href == "https://example.com/finetuned.onnx"
        assert local.assets["source-code"].href == base.assets["source-code"].href
        derived_hrefs = {lnk.get_href() for lnk in local.links if lnk.rel == "derived_from"}
        assert "../base-models/example-unet/example-unet.json" in derived_hrefs
        assert "../datasets/ds-1/ds-1.json" in derived_hrefs
        assert any(lnk.rel == "predecessor-version" for lnk in local.links)
        dataset_links = [
            lnk for lnk in local.links if lnk.rel == "derived_from" and "datasets" in (lnk.get_href() or "")
        ]
        assert dataset_links[0].extra_fields.get("title") == "Buildings Banepa"

    def test_zenml_artifact_version_id_stored_on_asset(self):
        base = _base_model()
        local = build_local_model_item(
            base_model_item=base,
            item_id="local-v1",
            checkpoint_href="https://example.com/model.pt",
            onnx_href="https://example.com/model.onnx",
            mlm_hyperparameters={},
            keywords=["building"],
            base_model_href="../base-models/example-unet/example-unet.json",
            dataset_href="../datasets/ds-1/ds-1.json",
            version="1",
            title="T",
            description="D",
            user_id="u",
            zenml_artifact_version_id="aaaa-bbbb-cccc",
        )
        assert local.assets["checkpoint"].extra_fields["zenml:artifact_version_id"] == "aaaa-bbbb-cccc"

    def test_zenml_artifact_version_id_absent_when_none(self):
        base = _base_model()
        local = build_local_model_item(
            base_model_item=base,
            item_id="local-v1",
            checkpoint_href="https://example.com/weights.pt",
            onnx_href="https://example.com/weights.onnx",
            mlm_hyperparameters={},
            keywords=["building"],
            base_model_href="../base-models/example-unet/example-unet.json",
            dataset_href="../datasets/ds-1/ds-1.json",
            version="1",
            title="T",
            description="D",
            user_id="u",
        )
        assert "zenml:artifact_version_id" not in local.assets["checkpoint"].extra_fields

    def test_none_geometry_raises(self):
        base = _base_model()
        base.geometry = None
        with pytest.raises(ValueError, match="geometry must be provided"):
            build_local_model_item(
                base_model_item=base,
                item_id="bad",
                checkpoint_href="w.pth",
                onnx_href="w.onnx",
                mlm_hyperparameters={},
                keywords=["building"],
                base_model_href="../base-models/b/b.json",
                dataset_href="../datasets/d/d.json",
                version="1",
                title="T",
                description="D",
                user_id="u",
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


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Buildings Banepa", "buildings-banepa"),
        ("My Dataset (v2)", "my-dataset-v2"),
        ("  simple  ", "simple"),
        ("UPPER-case_Mix", "upper-case_mix"),
    ],
)
def test_slugify(text, expected):
    assert _slugify(text) == expected


class TestLocalModelMetricsAndTiming:
    def test_fair_metrics_stored_directly(self):
        base = _base_model()
        metrics = {"fair:accuracy": 0.95, "fair:mean_iou": 0.72, "fair:per_class_iou": {"building": 0.81}}
        local = build_local_model_item(
            base_model_item=base,
            item_id="local-m",
            checkpoint_href="https://example.com/w.pt",
            onnx_href="https://example.com/w.onnx",
            mlm_hyperparameters={},
            keywords=["building"],
            base_model_href="../base-models/b/b.json",
            dataset_href="../datasets/d/d.json",
            version="1",
            title="T",
            description="D",
            user_id="u",
            metrics=metrics,
        )
        assert local.properties["fair:accuracy"] == 0.95
        assert local.properties["fair:per_class_iou"] == {"building": 0.81}

    def test_training_timing_stored(self):
        base = _base_model()
        local = build_local_model_item(
            base_model_item=base,
            item_id="local-t",
            checkpoint_href="https://example.com/w.pt",
            onnx_href="https://example.com/w.onnx",
            mlm_hyperparameters={},
            keywords=["building"],
            base_model_href="../base-models/b/b.json",
            dataset_href="../datasets/d/d.json",
            version="1",
            title="T",
            description="D",
            user_id="u",
            training_started_at="2024-01-01T00:00:00+00:00",
            training_ended_at="2024-01-01T01:30:00+00:00",
            training_duration_seconds=5400.0,
        )
        assert local.properties["fair:training_started_at"] == "2024-01-01T00:00:00+00:00"
        assert local.properties["fair:training_ended_at"] == "2024-01-01T01:30:00+00:00"
        assert local.properties["fair:training_duration_seconds"] == 5400.0

    def test_self_link_present(self):
        base = _base_model()
        local = build_local_model_item(
            base_model_item=base,
            item_id="local-s",
            checkpoint_href="https://example.com/w.pt",
            onnx_href="https://example.com/w.onnx",
            mlm_hyperparameters={},
            keywords=["building"],
            base_model_href="../base-models/b/b.json",
            dataset_href="../datasets/d/d.json",
            version="1",
            title="T",
            description="D",
            user_id="u",
            self_href="https://api.example.com/collections/local-models/items/local-s",
        )
        self_links = [lnk for lnk in local.links if lnk.rel == "self"]
        assert len(self_links) == 1
        assert self_links[0].get_href() == "https://api.example.com/collections/local-models/items/local-s"


class TestSlugifyUnderscores:
    def test_preserves_underscores(self) -> None:
        assert _slugify("my_model") == "my_model"

    def test_spaces_become_hyphens(self) -> None:
        assert _slugify("my model") == "my-model"

    def test_underscores_and_spaces_produce_different_ids(self) -> None:
        assert _slugify("my_model") != _slugify("my model")
