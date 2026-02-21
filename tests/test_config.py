"""Tests for fair_models.zenml.config -- YAML config generation from STAC items."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from fair_models.stac.builders import build_base_model_item, build_dataset_item
from fair_models.zenml.config import generate_inference_config, generate_training_config


def _make_base_model(**overrides: Any) -> Any:
    defaults: dict[str, Any] = {
        "item_id": "example-unet",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
        },
        "dt": datetime(2024, 1, 1, tzinfo=UTC),
        "mlm_name": "example-unet",
        "mlm_architecture": "UNet",
        "mlm_tasks": ["semantic-segmentation"],
        "mlm_framework": "pytorch",
        "mlm_framework_version": "2.1.0",
        "mlm_input": [
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
        "mlm_output": [
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
        "mlm_hyperparameters": {"epochs": 15, "batch_size": 4, "learning_rate": 0.0001},
        "keywords": ["building", "semantic-segmentation", "polygon"],
        "model_href": "torchgeo.models.Unet_Weights.OAM_RGB_RESNET50_TCD",
        "model_artifact_type": "pt",
        "mlm_pretrained": True,
        "mlm_pretrained_source": "OAM-TCD",
        "source_code_href": "https://github.com/hotosm/fAIr-models/tree/main/models/example_unet",
        "source_code_entrypoint": "models.example_unet.pipeline:train",
        "training_runtime_href": "ghcr.io/hotosm/fair-unet:v1",
        "inference_runtime_href": "ghcr.io/hotosm/fair-unet:v1",
    }
    defaults.update(overrides)
    return build_base_model_item(**defaults)


def _make_dataset(tmp_path: Any) -> Any:
    import json

    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[85.0, 27.0], [85.1, 27.0], [85.1, 27.1], [85.0, 27.1], [85.0, 27.0]]],
                },
                "properties": {},
            }
        ],
    }
    labels_path = tmp_path / "labels.geojson"
    labels_path.write_text(json.dumps(geojson))
    return build_dataset_item(
        item_id="buildings-banepa",
        dt=datetime(2024, 6, 1, tzinfo=UTC),
        label_type="vector",
        label_tasks=["segmentation"],
        label_classes=[{"name": "building", "classes": ["building"]}],
        keywords=["building", "semantic-segmentation"],
        chips_href="data/banepa/oam/",
        labels_href=str(labels_path),
    )


class TestGenerateTrainingConfig:
    def test_model_name_in_config(self, tmp_path):
        cfg = generate_training_config(
            _make_base_model(),
            _make_dataset(tmp_path),
            model_name="unet-finetuned-banepa",
        )
        assert cfg["model"]["name"] == "unet-finetuned-banepa"

    def test_default_hyperparameters(self, tmp_path):
        cfg = generate_training_config(
            _make_base_model(),
            _make_dataset(tmp_path),
            model_name="test",
        )
        params = cfg["parameters"]
        assert params["epochs"] == 15
        assert params["batch_size"] == 4
        assert params["learning_rate"] == 0.0001

    def test_overrides_take_precedence(self, tmp_path):
        cfg = generate_training_config(
            _make_base_model(),
            _make_dataset(tmp_path),
            model_name="test",
            overrides={"epochs": 1, "learning_rate": 0.001},
        )
        assert cfg["parameters"]["epochs"] == 1
        assert cfg["parameters"]["learning_rate"] == 0.001
        # Non-overridden defaults remain
        assert cfg["parameters"]["batch_size"] == 4

    def test_dataset_paths(self, tmp_path):
        ds = _make_dataset(tmp_path)
        cfg = generate_training_config(_make_base_model(), ds, model_name="test")
        assert cfg["parameters"]["dataset_chips"] == "data/banepa/oam/"
        assert "labels.geojson" in cfg["parameters"]["dataset_labels"]

    def test_input_spec_extracted(self, tmp_path):
        cfg = generate_training_config(_make_base_model(), _make_dataset(tmp_path), model_name="test")
        assert cfg["parameters"]["chip_size"] == 512
        assert cfg["parameters"]["bands"] == ["red", "green", "blue"]
        assert cfg["parameters"]["num_classes"] == 2

    def test_docker_settings(self, tmp_path):
        cfg = generate_training_config(_make_base_model(), _make_dataset(tmp_path), model_name="test")
        assert cfg["settings"]["docker"]["parent_image"] == "ghcr.io/hotosm/fair-unet:v1"

    def test_base_model_weights_href(self, tmp_path):
        cfg = generate_training_config(_make_base_model(), _make_dataset(tmp_path), model_name="test")
        assert cfg["parameters"]["base_model_weights"] == "torchgeo.models.Unet_Weights.OAM_RGB_RESNET50_TCD"


class TestGenerateInferenceConfig:
    def test_model_weights_and_input(self, tmp_path):
        base = _make_base_model()
        cfg = generate_inference_config(base, "/data/prediction/input/")
        assert cfg["parameters"]["model_weights"] == "torchgeo.models.Unet_Weights.OAM_RGB_RESNET50_TCD"
        assert cfg["parameters"]["input_images"] == "/data/prediction/input/"

    def test_post_processing_entrypoint(self, tmp_path):
        base = _make_base_model()
        cfg = generate_inference_config(base, "/data/input/")
        assert cfg["parameters"]["post_processing"] == "models.example_unet.pipeline:postprocess"

    def test_inference_docker_settings(self, tmp_path):
        base = _make_base_model()
        cfg = generate_inference_config(base, "/data/input/")
        assert cfg["settings"]["docker"]["parent_image"] == "ghcr.io/hotosm/fair-unet:v1"

    def test_input_spec(self, tmp_path):
        base = _make_base_model()
        cfg = generate_inference_config(base, "/data/input/")
        assert cfg["parameters"]["chip_size"] == 512
        assert cfg["parameters"]["num_classes"] == 2
