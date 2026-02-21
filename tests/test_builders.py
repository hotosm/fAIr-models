from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from fair_models.stac.builders import (
    _infer_runtime_media_type,
    build_base_model_item,
    build_dataset_item,
    build_local_model_item,
)


@pytest.fixture()
def sample_geojson(tmp_path):
    """Create a minimal GeoJSON file for dataset builder tests."""
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[85.51, 27.63], [85.53, 27.63], [85.53, 27.64], [85.51, 27.64], [85.51, 27.63]]],
                },
                "properties": {"class": "building"},
            }
        ],
    }
    path = tmp_path / "labels.geojson"
    path.write_text(json.dumps(geojson))
    return str(path)


@pytest.fixture()
def base_model_item():
    """A fully populated base model item for local model builder tests."""
    return build_base_model_item(
        item_id="example-unet",
        geometry={"type": "Polygon", "coordinates": [[[0, -90], [180, -90], [180, 90], [0, 90], [0, -90]]]},
        dt=datetime(2024, 1, 1, tzinfo=UTC),
        mlm_name="example-unet",
        mlm_architecture="UNet",
        mlm_tasks=["semantic-segmentation"],
        mlm_framework="pytorch",
        mlm_framework_version="2.1.0",
        mlm_input=[
            {
                "name": "RGB chips",
                "bands": [{"name": "red"}, {"name": "green"}, {"name": "blue"}],
                "input": {
                    "shape": [-1, 3, 512, 512],
                    "dim_order": ["batch", "channel", "height", "width"],
                    "data_type": "float32",
                },
                "pre_processing_function": {
                    "format": "python",
                    "expression": "models.example_unet.pipeline:preprocess",
                },
            }
        ],
        mlm_output=[
            {
                "name": "segmentation mask",
                "tasks": ["semantic-segmentation"],
                "result": {
                    "shape": [-1, 2, 512, 512],
                    "dim_order": ["batch", "channel", "height", "width"],
                    "data_type": "float32",
                },
                "classification:classes": [
                    {"name": "background", "value": 0},
                    {"name": "building", "value": 1},
                ],
                "post_processing_function": {
                    "format": "python",
                    "expression": "models.example_unet.pipeline:postprocess",
                },
            }
        ],
        mlm_hyperparameters={"epochs": 15, "batch_size": 4, "learning_rate": 0.0001},
        keywords=["building", "semantic-segmentation", "polygon"],
        model_href="torchgeo.models.Unet_Weights.OAM_RGB_RESNET50_TCD",
        model_artifact_type="pt",
        mlm_pretrained=True,
        mlm_pretrained_source="OAM-TCD",
        source_code_href="https://github.com/hotosm/fAIr-models/tree/main/models/example_unet",
        source_code_entrypoint="models.example_unet.pipeline:train",
        training_runtime_href="local",
        inference_runtime_href="local",
    )


class TestBuildDatasetItem:
    def test_label_extension_fields(self, sample_geojson):
        item = build_dataset_item(
            item_id="buildings-banepa",
            dt=datetime(2024, 6, 1, tzinfo=UTC),
            label_type="vector",
            label_tasks=["segmentation"],
            label_classes=[{"name": "building", "classes": ["building"]}],
            keywords=["building", "semantic-segmentation", "polygon"],
            chips_href="data/banepa/oam/",
            labels_href=sample_geojson,
        )

        assert item.properties["label:type"] == "vector"
        assert item.properties["label:tasks"] == ["segmentation"]
        assert item.properties["label:classes"] == [{"name": "building", "classes": ["building"]}]
        assert item.properties["keywords"] == ["building", "semantic-segmentation", "polygon"]

    def test_asset_hrefs(self, sample_geojson):
        item = build_dataset_item(
            item_id="buildings-banepa",
            dt=datetime(2024, 6, 1, tzinfo=UTC),
            label_type="vector",
            label_tasks=["segmentation"],
            label_classes=[{"name": "building", "classes": ["building"]}],
            keywords=["building"],
            chips_href="data/banepa/oam/",
            labels_href=sample_geojson,
        )

        assert "chips" in item.assets
        assert item.assets["chips"].href == "data/banepa/oam/"
        assert "labels" in item.assets
        assert item.assets["labels"].href == sample_geojson

    def test_bbox_derived_from_geojson(self, sample_geojson):
        item = build_dataset_item(
            item_id="test",
            dt=datetime(2024, 6, 1, tzinfo=UTC),
            label_type="vector",
            label_tasks=["segmentation"],
            label_classes=[],
            keywords=["building"],
            chips_href="chips/",
            labels_href=sample_geojson,
        )

        assert item.bbox is not None
        assert pytest.approx(item.bbox[0], abs=0.01) == 85.51
        assert pytest.approx(item.bbox[1], abs=0.01) == 27.63

    def test_download_asset(self, sample_geojson):
        item = build_dataset_item(
            item_id="test-dl",
            dt=datetime(2024, 6, 1, tzinfo=UTC),
            label_type="vector",
            label_tasks=["segmentation"],
            label_classes=[],
            keywords=["building"],
            chips_href="chips/",
            labels_href=sample_geojson,
            download_href="s3://bucket/datasets/banepa.zip",
        )

        assert "download" in item.assets
        assert item.assets["download"].href == "s3://bucket/datasets/banepa.zip"
        assert item.assets["download"].media_type == "application/zip"
        assert item.assets["download"].roles is not None
        assert "archive" in item.assets["download"].roles

    def test_no_download_asset_when_omitted(self, sample_geojson):
        item = build_dataset_item(
            item_id="test-no-dl",
            dt=datetime(2024, 6, 1, tzinfo=UTC),
            label_type="vector",
            label_tasks=["segmentation"],
            label_classes=[],
            keywords=["building"],
            chips_href="chips/",
            labels_href=sample_geojson,
        )

        assert "download" not in item.assets


class TestBuildBaseModelItem:
    def test_mlm_fields(self, base_model_item):
        assert base_model_item.properties["mlm:name"] == "example-unet"
        assert base_model_item.properties["mlm:architecture"] == "UNet"
        assert base_model_item.properties["mlm:tasks"] == ["semantic-segmentation"]
        assert base_model_item.properties["mlm:framework"] == "pytorch"
        assert base_model_item.properties["mlm:pretrained"] is True
        assert base_model_item.properties["version"] == "1"

    def test_hyperparameters(self, base_model_item):
        hp = base_model_item.properties["mlm:hyperparameters"]
        assert hp["epochs"] == 15
        assert hp["batch_size"] == 4

    def test_assets(self, base_model_item):
        assert "model" in base_model_item.assets
        assert base_model_item.assets["model"].extra_fields["mlm:artifact_type"] == "pt"
        assert "source-code" in base_model_item.assets
        entrypoint = base_model_item.assets["source-code"].extra_fields["mlm:entrypoint"]
        assert entrypoint == "models.example_unet.pipeline:train"
        assert "training-runtime" in base_model_item.assets
        assert "inference-runtime" in base_model_item.assets

    def test_model_media_type_uses_framework(self):
        item = build_base_model_item(
            item_id="tf-model",
            geometry={"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]},
            dt=datetime(2024, 1, 1, tzinfo=UTC),
            mlm_name="tf-model",
            mlm_architecture="ResNet",
            mlm_tasks=["classification"],
            mlm_framework="tensorflow",
            mlm_framework_version="2.15.0",
            mlm_input=[],
            mlm_output=[],
            mlm_hyperparameters={},
            keywords=["building"],
            model_href="s3://weights/model.h5",
            model_artifact_type="h5",
            mlm_pretrained=False,
            mlm_pretrained_source=None,
            source_code_href="https://github.com/example",
            source_code_entrypoint="models.tf.pipeline:train",
            training_runtime_href="ghcr.io/hotosm/fair-tf:v1",
            inference_runtime_href="ghcr.io/hotosm/fair-tf:v1",
        )

        assert item.assets["model"].media_type == "application/octet-stream; framework=tensorflow"

    def test_runtime_media_type_oci_for_registry(self):
        item = build_base_model_item(
            item_id="docker-model",
            geometry={"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]},
            dt=datetime(2024, 1, 1, tzinfo=UTC),
            mlm_name="docker-model",
            mlm_architecture="UNet",
            mlm_tasks=["semantic-segmentation"],
            mlm_framework="pytorch",
            mlm_framework_version="2.1.0",
            mlm_input=[],
            mlm_output=[],
            mlm_hyperparameters={},
            keywords=["building"],
            model_href="s3://weights/model.pt",
            model_artifact_type="pt",
            mlm_pretrained=False,
            mlm_pretrained_source=None,
            source_code_href="https://github.com/example",
            source_code_entrypoint="models.unet.pipeline:train",
            training_runtime_href="ghcr.io/hotosm/fair-unet:v1",
            inference_runtime_href="ghcr.io/hotosm/fair-unet:v1",
        )

        assert item.assets["training-runtime"].media_type == "application/vnd.oci.image.index.v1+json"
        assert item.assets["inference-runtime"].media_type == "application/vnd.oci.image.index.v1+json"

    def test_runtime_media_type_plain_for_local(self, base_model_item):
        # base_model_item uses href="local"
        assert base_model_item.assets["training-runtime"].media_type == "text/plain"
        assert base_model_item.assets["inference-runtime"].media_type == "text/plain"

    def test_bbox_computed_from_geometry(self, base_model_item):
        assert base_model_item.bbox is not None
        assert base_model_item.bbox[0] == 0
        assert base_model_item.bbox[2] == 180

    def test_processing_expressions(self, base_model_item):
        inp = base_model_item.properties["mlm:input"][0]
        assert inp["pre_processing_function"]["expression"] == "models.example_unet.pipeline:preprocess"
        out = base_model_item.properties["mlm:output"][0]
        assert out["post_processing_function"]["expression"] == "models.example_unet.pipeline:postprocess"


class TestBuildLocalModelItem:
    def test_derived_from_links(self, base_model_item):
        local = build_local_model_item(
            base_model_item=base_model_item,
            item_id="example-unet-finetuned-banepa-v1",
            dt=datetime(2024, 7, 1, tzinfo=UTC),
            model_href="artifacts/finetuned.pth",
            mlm_hyperparameters={"epochs": 1, "batch_size": 4},
            keywords=["building", "semantic-segmentation"],
            base_model_item_id="example-unet",
            dataset_item_id="buildings-banepa",
            version="1",
        )

        derived_links = [lnk for lnk in local.links if lnk.rel == "derived_from"]
        targets = {lnk.target for lnk in derived_links}
        assert "example-unet" in targets
        assert "buildings-banepa" in targets

    def test_version_extension(self, base_model_item):
        local = build_local_model_item(
            base_model_item=base_model_item,
            item_id="example-unet-finetuned-banepa-v2",
            dt=datetime(2024, 7, 1, tzinfo=UTC),
            model_href="artifacts/finetuned_v2.pth",
            mlm_hyperparameters={"epochs": 5},
            keywords=["building"],
            base_model_item_id="example-unet",
            dataset_item_id="buildings-banepa",
            version="2",
            predecessor_version_item_id="example-unet-finetuned-banepa-v1",
        )

        assert local.properties["version"] == "2"
        assert local.properties["deprecated"] is False

        pred_links = [lnk for lnk in local.links if lnk.rel == "predecessor-version"]
        assert len(pred_links) == 1
        assert pred_links[0].target == "example-unet-finetuned-banepa-v1"

        latest_links = [lnk for lnk in local.links if lnk.rel == "latest-version"]
        assert len(latest_links) == 1

    def test_copies_mlm_fields_from_base(self, base_model_item):
        local = build_local_model_item(
            base_model_item=base_model_item,
            item_id="local-1",
            dt=datetime(2024, 7, 1, tzinfo=UTC),
            model_href="weights.pth",
            mlm_hyperparameters={"epochs": 1},
            keywords=["building"],
            base_model_item_id="example-unet",
            dataset_item_id="ds-1",
            version="1",
        )

        assert local.properties["mlm:architecture"] == "UNet"
        assert local.properties["mlm:framework"] == "pytorch"
        assert local.properties["mlm:input"] == base_model_item.properties["mlm:input"]
        assert local.properties["mlm:output"] == base_model_item.properties["mlm:output"]

    def test_model_href_overridden(self, base_model_item):
        local = build_local_model_item(
            base_model_item=base_model_item,
            item_id="local-1",
            dt=datetime(2024, 7, 1, tzinfo=UTC),
            model_href="my/custom/weights.pth",
            mlm_hyperparameters={"epochs": 1},
            keywords=["building"],
            base_model_item_id="example-unet",
            dataset_item_id="ds-1",
            version="1",
        )

        assert local.assets["model"].href == "my/custom/weights.pth"
        # Other assets copied from base
        assert local.assets["source-code"].href == base_model_item.assets["source-code"].href


class TestInferRuntimeMediaType:
    def test_ghcr(self):
        assert _infer_runtime_media_type("ghcr.io/hotosm/fair:v1") == "application/vnd.oci.image.index.v1+json"

    def test_docker_hub(self):
        assert _infer_runtime_media_type("docker.io/library/python:3.12") == "application/vnd.oci.image.index.v1+json"

    def test_quay(self):
        assert _infer_runtime_media_type("quay.io/org/image:latest") == "application/vnd.oci.image.index.v1+json"

    def test_dockerfile(self):
        assert _infer_runtime_media_type("models/ramp/Dockerfile") == "text/x-dockerfile"

    def test_local_fallback(self):
        assert _infer_runtime_media_type("local") == "text/plain"
