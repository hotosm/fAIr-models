from __future__ import annotations

import json
from datetime import UTC, datetime
from unittest.mock import patch

import pystac
import pytest

from fair.stac.builders import build_base_model_item, build_dataset_item
from fair.stac.validators import validate_base_model_item, validate_compatibility, validate_mlm_schema

_MLM_INPUT = [
    {
        "name": "RGB chips",
        "bands": [{"name": "red"}, {"name": "green"}, {"name": "blue"}],
        "input": {
            "shape": [-1, 3, 256, 256],
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
            "shape": [-1, 2, 256, 256],
            "dim_order": ["batch", "channel", "height", "width"],
            "data_type": "float32",
        },
        "classification:classes": [{"name": "background", "value": 0}, {"name": "building", "value": 1}],
        "post_processing_function": {"format": "python", "expression": "mod:postprocess"},
    }
]


def _model(keywords=None, tasks=None):
    return build_base_model_item(
        item_id="m",
        dt=datetime(2024, 1, 1, tzinfo=UTC),
        geometry={"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]},
        mlm_name="m",
        mlm_architecture="UNet",
        mlm_tasks=tasks or ["semantic-segmentation"],
        mlm_framework="PyTorch",
        mlm_framework_version="2.1.0",
        mlm_input=[],
        mlm_output=[],
        mlm_hyperparameters={},
        keywords=keywords or ["building", "semantic-segmentation"],
        model_href="w.pt",
        model_artifact_type="torch.save",
        mlm_pretrained=False,
        mlm_pretrained_source=None,
        source_code_href="https://example.com",
        source_code_entrypoint="mod:train",
        training_runtime_href="local",
        inference_runtime_href="local",
    )


def _dataset(tmp_path, keywords=None, label_tasks=None):
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]},
                "properties": {},
            }
        ],
    }
    path = tmp_path / "labels.geojson"
    path.write_text(json.dumps(geojson))
    return build_dataset_item(
        item_id="d",
        dt=datetime(2024, 6, 1, tzinfo=UTC),
        label_type="vector",
        label_tasks=label_tasks or ["segmentation"],
        label_classes=[{"name": "building", "classes": ["building"]}],
        keywords=keywords or ["building", "semantic-segmentation"],
        chips_href="chips/",
        labels_href=str(path),
    )


def test_compatible_pair(tmp_path):
    assert validate_compatibility(_model(), _dataset(tmp_path)) == []


def test_disjoint_keywords_and_tasks(tmp_path):
    """Disjoint keywords + disjoint tasks should produce multiple errors."""
    errors = validate_compatibility(
        _model(tasks=["object-detection"], keywords=["road"]),
        _dataset(tmp_path, label_tasks=["segmentation"], keywords=["tree"]),
    )
    assert any("No keywords in common" in e for e in errors)
    assert any("No task overlap" in e for e in errors)


@pytest.mark.parametrize(
    ("model_task", "label_task"),
    [
        ("semantic-segmentation", "segmentation"),
        ("instance-segmentation", "segmentation"),
        ("object-detection", "detection"),
    ],
)
def test_task_label_mapping(tmp_path, model_task, label_task):
    errors = validate_compatibility(
        _model(tasks=[model_task], keywords=["building"]),
        _dataset(tmp_path, label_tasks=[label_task], keywords=["building"]),
    )
    assert not any("No task overlap" in e for e in errors)


def test_unknown_keywords(tmp_path):
    errors = validate_compatibility(
        _model(keywords=["alien"]),
        _dataset(tmp_path, keywords=["alien"]),
    )
    assert any("Unknown" in e for e in errors)


def test_geometry_and_task_keywords_are_valid(tmp_path):
    """polygon, semantic-segmentation etc. should pass vocabulary check."""
    errors = validate_compatibility(
        _model(keywords=["building", "polygon", "semantic-segmentation"]),
        _dataset(tmp_path, keywords=["building", "polygon", "semantic-segmentation"]),
    )
    assert not any("Unknown" in e for e in errors)


def test_validate_mlm_schema_delegates_to_pystac():
    item = _model()
    with patch.object(item, "validate", return_value=None):
        assert validate_mlm_schema(item) == []
    with patch.object(item, "validate", side_effect=pystac.errors.STACValidationError(message="bad")):
        assert "bad" in validate_mlm_schema(item)[0]


def _valid_base_model():
    item = build_base_model_item(
        item_id="test-model",
        dt=datetime(2024, 1, 1, tzinfo=UTC),
        geometry={"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]},
        mlm_name="test-model",
        mlm_architecture="UNet",
        mlm_tasks=["semantic-segmentation"],
        mlm_framework="PyTorch",
        mlm_framework_version="2.1.0",
        mlm_input=json.loads(json.dumps(_MLM_INPUT)),
        mlm_output=json.loads(json.dumps(_MLM_OUTPUT)),
        mlm_hyperparameters={"epochs": 10, "batch_size": 4},
        keywords=["building", "semantic-segmentation", "polygon"],
        model_href="weights.pt",
        model_artifact_type="torch.save",
        mlm_pretrained=True,
        mlm_pretrained_source="imagenet",
        source_code_href="https://github.com/example",
        source_code_entrypoint="mod:train",
        training_runtime_href="ghcr.io/hotosm/test:v1",
        inference_runtime_href="ghcr.io/hotosm/test:v1",
    )
    item.properties["license"] = "GPL-3.0-only"
    return item


class TestValidateBaseModelItem:
    def test_valid_item_passes(self):
        assert validate_base_model_item(_valid_base_model()) == []

    def test_missing_property(self):
        item = _valid_base_model()
        del item.properties["mlm:name"]
        errors = validate_base_model_item(item)
        assert any("mlm:name" in e for e in errors)

    def test_missing_asset(self):
        item = _valid_base_model()
        del item.assets["source-code"]
        errors = validate_base_model_item(item)
        assert any("source-code" in e for e in errors)

    def test_missing_asset_field(self):
        item = _valid_base_model()
        del item.assets["model"].extra_fields["mlm:artifact_type"]
        errors = validate_base_model_item(item)
        assert any("mlm:artifact_type" in e for e in errors)

    def test_empty_tasks_list(self):
        item = _valid_base_model()
        item.properties["mlm:tasks"] = []
        errors = validate_base_model_item(item)
        assert any("non-empty" in e for e in errors)

    def test_unknown_keyword(self):
        item = _valid_base_model()
        item.properties["keywords"] = ["alien"]
        errors = validate_base_model_item(item)
        assert any("Unknown keywords" in e for e in errors)

    def test_unknown_task(self):
        item = _valid_base_model()
        item.properties["mlm:tasks"] = ["time-travel"]
        errors = validate_base_model_item(item)
        assert any("Invalid mlm:tasks" in e for e in errors)

    def test_missing_pre_processing_function(self):
        item = _valid_base_model()
        del item.properties["mlm:input"][0]["pre_processing_function"]
        errors = validate_base_model_item(item)
        assert any("pre_processing_function" in e for e in errors)

    def test_missing_post_processing_function(self):
        item = _valid_base_model()
        del item.properties["mlm:output"][0]["post_processing_function"]
        errors = validate_base_model_item(item)
        assert any("post_processing_function" in e for e in errors)

    def test_missing_classification_classes(self):
        item = _valid_base_model()
        del item.properties["mlm:output"][0]["classification:classes"]
        errors = validate_base_model_item(item)
        assert any("classification:classes" in e for e in errors)

    def test_processing_fn_missing_format(self):
        item = _valid_base_model()
        del item.properties["mlm:input"][0]["pre_processing_function"]["format"]
        errors = validate_base_model_item(item)
        assert any("format" in e for e in errors)

    def test_missing_extension(self):
        item = _valid_base_model()
        item.stac_extensions = []
        errors = validate_base_model_item(item)
        assert len([e for e in errors if "Missing extension" in e]) == 5

    def test_missing_license(self):
        item = _valid_base_model()
        del item.properties["license"]
        errors = validate_base_model_item(item)
        assert any("license" in e for e in errors)

    def test_invalid_license(self):
        item = _valid_base_model()
        item.properties["license"] = "WTFPL"
        errors = validate_base_model_item(item)
        assert any("Invalid license" in e for e in errors)

    def test_invalid_framework(self):
        item = _valid_base_model()
        item.properties["mlm:framework"] = "jax"
        errors = validate_base_model_item(item)
        assert any("Invalid mlm:framework" in e for e in errors)
