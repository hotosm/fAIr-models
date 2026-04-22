from __future__ import annotations

import copy
import json
from typing import ClassVar
from unittest.mock import patch

import pystac
import pytest

from fair.stac.builders import build_base_model_item, build_dataset_item
from fair.stac.validators import (
    validate_compatibility,
    validate_hyperparameters,
    validate_item,
    validate_metrics_against_spec,
    validate_mlm_schema,
    validate_pipeline_config,
    validate_predictions_geojson,
)

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
        "classification:classes": [
            {"name": "background", "value": 0, "description": "Background pixels"},
            {"name": "building", "value": 1, "description": "Building footprint pixels"},
        ],
        "post_processing_function": {"format": "python", "expression": "mod:postprocess"},
    }
]


def _model(keywords=None, tasks=None):
    return build_base_model_item(
        item_id="m",
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
        checkpoint_href="w.pt",
        onnx_href="https://example.com/model.onnx",
        checkpoint_artifact_type="torch.save",
        mlm_pretrained=False,
        mlm_pretrained_source=None,
        source_code_href="https://example.com",
        source_code_entrypoint="mod:train",
        training_runtime_href="local",
        inference_runtime_href="local",
        title="Validator test model",
        description="Model used in validator tests.",
        fair_metrics_spec=[{"name": "accuracy", "description": "Pixel accuracy", "higher_is_better": True}],
        providers=[{"name": "HOTOSM", "roles": ["producer"]}],
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
    item = build_dataset_item(
        item_id="d",
        label_type="vector",
        label_tasks=label_tasks or ["segmentation"],
        label_classes=[{"name": "building", "classes": ["building"]}],
        keywords=keywords or ["building", "semantic-segmentation"],
        chips_href="chips/",
        labels_href=str(path),
        title="Validator test dataset",
        description="Dataset used in validator tests.",
        user_id="osm-test",
        providers=[{"name": "osm-test", "roles": ["producer"]}],
    )
    item.properties["label:properties"] = ["class"]
    item.properties["label:description"] = "Test labels"
    return item


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
        geometry={"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]},
        mlm_name="test-model",
        mlm_architecture="UNet",
        mlm_tasks=["semantic-segmentation"],
        mlm_framework="PyTorch",
        mlm_framework_version="2.1.0",
        mlm_input=copy.deepcopy(_MLM_INPUT),
        mlm_output=copy.deepcopy(_MLM_OUTPUT),
        mlm_hyperparameters={
            "training.epochs": 10,
            "training.batch_size": 4,
            "training.learning_rate": 0.001,
            "inference.confidence_threshold": 0.5,
        },
        keywords=["building", "semantic-segmentation", "polygon"],
        checkpoint_href="https://example.com/weights.pt",
        onnx_href="https://example.com/model.onnx",
        checkpoint_artifact_type="torch.save",
        mlm_pretrained=True,
        mlm_pretrained_source="imagenet",
        source_code_href="https://github.com/example",
        source_code_entrypoint="mod:train",
        training_runtime_href="ghcr.io/hotosm/test:v1",
        inference_runtime_href="ghcr.io/hotosm/test:v1",
        title="Valid test model",
        description="A valid base model for validator tests.",
        fair_metrics_spec=[{"name": "accuracy", "description": "Pixel accuracy", "higher_is_better": True}],
        providers=[{"name": "HOTOSM", "roles": ["producer"], "url": "https://www.hotosm.org"}],
        readme_href="https://example.com/README.md",
    )
    item.properties["license"] = "AGPL-3.0-only"
    item.properties["fair:split_spec"] = {
        "strategy": "random",
        "default_ratio": 0.2,
        "seed": 42,
        "description": "Random split for testing",
    }
    item.assets["checkpoint"].extra_fields["raster:bands"] = [
        {"name": "red"},
        {"name": "green"},
        {"name": "blue"},
    ]
    return item


class TestValidateBaseModelItem:
    def test_valid_item_passes(self):
        assert validate_item(_valid_base_model()) == []

    def test_missing_property(self):
        item = _valid_base_model()
        del item.properties["mlm:name"]
        errors = validate_item(item)
        assert any("mlm:name" in e for e in errors)

    def test_missing_asset(self):
        item = _valid_base_model()
        del item.assets["source-code"]
        errors = validate_item(item)
        assert any("source-code" in e for e in errors)

    def test_readme_asset_is_optional(self):
        item = _valid_base_model()
        del item.assets["readme"]
        errors = validate_item(item)
        assert not any("readme" in e for e in errors)

    def test_missing_asset_field(self):
        item = _valid_base_model()
        del item.assets["checkpoint"].extra_fields["mlm:artifact_type"]
        errors = validate_item(item)
        assert any("mlm:artifact_type" in e for e in errors)

    def test_empty_tasks_list(self):
        item = _valid_base_model()
        item.properties["mlm:tasks"] = []
        errors = validate_item(item)
        assert any("non-empty" in e for e in errors)

    def test_unknown_keyword(self):
        item = _valid_base_model()
        item.properties["keywords"] = ["alien"]
        errors = validate_item(item)
        assert any("Unknown keywords" in e for e in errors)

    def test_unknown_task(self):
        item = _valid_base_model()
        item.properties["mlm:tasks"] = ["time-travel"]
        errors = validate_item(item)
        assert any("time-travel" in e for e in errors)

    def test_missing_pre_processing_function(self):
        item = _valid_base_model()
        del item.properties["mlm:input"][0]["pre_processing_function"]
        errors = validate_item(item)
        assert any("pre_processing_function" in e for e in errors)

    def test_missing_post_processing_function(self):
        item = _valid_base_model()
        del item.properties["mlm:output"][0]["post_processing_function"]
        errors = validate_item(item)
        assert any("post_processing_function" in e for e in errors)

    def test_missing_classification_classes(self):
        item = _valid_base_model()
        del item.properties["mlm:output"][0]["classification:classes"]
        errors = validate_item(item)
        assert any("classification:classes" in e for e in errors)

    def test_processing_fn_missing_format(self):
        item = _valid_base_model()
        del item.properties["mlm:input"][0]["pre_processing_function"]["format"]
        errors = validate_item(item)
        assert any("format" in e for e in errors)

    def test_missing_extensions_skips_schema_validation(self):
        item = _valid_base_model()
        item.stac_extensions = []
        errors = validate_item(item)
        # With no extensions declared, PySTAC skips extension schema validation.
        # Only keyword vocabulary check still runs.
        assert not any("required property" in e for e in errors)

    def test_missing_license(self):
        item = _valid_base_model()
        del item.properties["license"]
        errors = validate_item(item)
        assert any("license" in e for e in errors)

    def test_invalid_license(self):
        item = _valid_base_model()
        item.properties["license"] = "WTFPL"
        errors = validate_item(item)
        assert any("WTFPL" in e for e in errors)

    def test_invalid_framework(self):
        item = _valid_base_model()
        item.properties["mlm:framework"] = "jax"
        errors = validate_item(item)
        assert any("jax" in e for e in errors)

    def test_valid_keywords(self):
        item = _valid_base_model()
        item.properties["keywords"] = ["building", "semantic-segmentation", "polygon"]
        assert validate_item(item) == []

    def test_line_geometry_keyword(self):
        item = _valid_base_model()
        item.properties["keywords"] = ["road", "semantic-segmentation", "line"]
        assert validate_item(item) == []

    def test_point_geometry_keyword(self):
        item = _valid_base_model()
        item.properties["keywords"] = ["tree", "object-detection", "point"]
        item.properties["mlm:tasks"] = ["object-detection"]
        assert validate_item(item) == []


class TestMandatoryHyperparameters:
    def test_missing_training_epochs_flagged(self):
        item = _valid_base_model()
        hp = dict(item.properties["mlm:hyperparameters"])
        del hp["training.epochs"]
        item.properties["mlm:hyperparameters"] = hp
        errors = validate_item(item)
        assert any("training.epochs" in e for e in errors)

    def test_missing_inference_confidence_threshold_flagged(self):
        item = _valid_base_model()
        hp = dict(item.properties["mlm:hyperparameters"])
        del hp["inference.confidence_threshold"]
        item.properties["mlm:hyperparameters"] = hp
        errors = validate_item(item)
        assert any("inference.confidence_threshold" in e for e in errors)

    def test_confidence_threshold_out_of_range_flagged(self):
        item = _valid_base_model()
        item.properties["mlm:hyperparameters"]["inference.confidence_threshold"] = 1.5
        errors = validate_item(item)
        assert any("1.5" in e or "maximum" in e for e in errors)


class TestSplitSpecValidation:
    def test_missing_split_spec_flagged(self):
        item = _valid_base_model()
        del item.properties["fair:split_spec"]
        errors = validate_item(item)
        assert any("fair:split_spec" in e for e in errors)

    def test_split_spec_missing_keys(self):
        item = _valid_base_model()
        item.properties["fair:split_spec"] = {"strategy": "random"}
        errors = validate_item(item)
        assert any("default_ratio" in e or "seed" in e or "description" in e for e in errors)

    def test_split_spec_invalid_ratio(self):
        item = _valid_base_model()
        item.properties["fair:split_spec"]["default_ratio"] = 1.5
        errors = validate_item(item)
        assert any("1.5" in e or "exclusiveMaximum" in e for e in errors)

    def test_valid_split_spec_passes(self):
        item = _valid_base_model()
        assert not any("split_spec" in e for e in validate_item(item))


class TestGeometryTypeCompatibility:
    def test_matching_geometry_types(self, tmp_path):
        errors = validate_compatibility(
            _model(keywords=["building", "polygon", "semantic-segmentation"]),
            _dataset(tmp_path, keywords=["building", "polygon", "semantic-segmentation"]),
        )
        assert not any("Geometry type mismatch" in e for e in errors)

    def test_mismatched_geometry_types(self, tmp_path):
        errors = validate_compatibility(
            _model(keywords=["building", "polygon", "semantic-segmentation"]),
            _dataset(tmp_path, keywords=["building", "point", "semantic-segmentation"]),
        )
        assert any("Geometry type mismatch" in e for e in errors)

    def test_no_geometry_in_either_skips_check(self, tmp_path):
        errors = validate_compatibility(
            _model(keywords=["building", "semantic-segmentation"]),
            _dataset(tmp_path, keywords=["building", "semantic-segmentation"]),
        )
        assert not any("Geometry type mismatch" in e for e in errors)


class TestValidatePipelineConfig:
    def test_valid_training_config(self):
        config = {
            "parameters": {
                "base_model_weights": "w.pt",
                "dataset_chips": "chips/",
                "dataset_labels": "labels.geojson",
                "hyperparameters": {"epochs": 10, "batch_size": 4},
            }
        }
        assert validate_pipeline_config(config, is_training=True) == []

    def test_missing_training_parameter(self):
        config = {"parameters": {"base_model_weights": "w.pt"}}
        errors = validate_pipeline_config(config, is_training=True)
        assert any("dataset_chips" in e for e in errors)
        assert any("hyperparameters" in e for e in errors)

    def test_missing_epochs_in_hyperparameters(self):
        config = {
            "parameters": {
                "base_model_weights": "w.pt",
                "dataset_chips": "chips/",
                "dataset_labels": "labels.geojson",
                "hyperparameters": {"batch_size": 4},
            }
        }
        errors = validate_pipeline_config(config, is_training=True)
        assert any("epochs" in e for e in errors)

    def test_valid_inference_config(self):
        config = {"parameters": {"model_uri": "s3://model.pt", "input_images": "/data/"}}
        assert validate_pipeline_config(config, is_training=False) == []

    def test_missing_inference_parameter(self):
        config = {"parameters": {"model_uri": "s3://model.pt"}}
        errors = validate_pipeline_config(config, is_training=False)
        assert any("input_images" in e for e in errors)


class TestValidatePredictionsGeojson:
    def test_valid_feature_collection(self):
        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [0, 0]},
                    "properties": {"class": 1},
                }
            ],
        }
        assert validate_predictions_geojson(geojson) == []

    def test_wrong_type(self):
        errors = validate_predictions_geojson({"type": "Feature"})
        assert any("FeatureCollection" in e for e in errors)

    def test_features_not_list(self):
        errors = validate_predictions_geojson({"type": "FeatureCollection", "features": "bad"})
        assert any("list" in e for e in errors)

    def test_feature_missing_geometry(self):
        geojson = {
            "type": "FeatureCollection",
            "features": [{"type": "Feature", "properties": {}}],
        }
        errors = validate_predictions_geojson(geojson)
        assert any("geometry" in e for e in errors)


class TestValidateMetricsAgainstSpec:
    def test_all_metrics_present(self):
        model = _model()
        errors = validate_metrics_against_spec({"accuracy": 0.95}, model)
        assert errors == []

    def test_missing_declared_metric(self):
        model = _model()
        errors = validate_metrics_against_spec({}, model)
        assert any("accuracy" in e for e in errors)

    def test_no_spec_allows_any_metrics(self):
        model = _model()
        model.properties["fair:metrics_spec"] = []
        assert validate_metrics_against_spec({"anything": 1.0}, model) == []


class TestValidatePredictionsGeojsonGeometryType:
    def _polygon_model(self):
        m = _model(keywords=["building", "semantic-segmentation", "polygon"])
        return m

    def test_polygon_keyword_accepts_polygon(self):
        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]},
                    "properties": {},
                }
            ],
        }
        assert validate_predictions_geojson(geojson, self._polygon_model()) == []

    def test_polygon_keyword_accepts_multipolygon(self):
        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "MultiPolygon",
                        "coordinates": [[[[0, 0], [1, 0], [1, 1], [0, 0]]]],
                    },
                    "properties": {},
                }
            ],
        }
        assert validate_predictions_geojson(geojson, self._polygon_model()) == []

    def test_polygon_keyword_rejects_point(self):
        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [0, 0]},
                    "properties": {},
                }
            ],
        }
        errors = validate_predictions_geojson(geojson, self._polygon_model())
        assert any("not allowed" in e for e in errors)

    def test_no_base_model_skips_geometry_check(self):
        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [0, 0]},
                    "properties": {},
                }
            ],
        }
        assert validate_predictions_geojson(geojson) == []


class TestValidateHyperparameters:
    def _model_with_spec(self):
        m = _model()
        m.properties["fair:hyperparameters_spec"] = [
            {
                "key": "epochs",
                "type": "int",
                "default": 5,
                "min": 1,
                "max": 500,
                "description": "Number of training epochs",
            },
            {
                "key": "learning_rate",
                "type": "float",
                "default": 0.001,
                "min": 1e-7,
                "max": 1.0,
                "description": "Optimizer learning rate",
            },
            {
                "key": "optimizer",
                "type": "str",
                "default": "AdamW",
                "values": ["AdamW", "Adam", "SGD"],
                "description": "Optimizer algorithm",
            },
            {
                "key": "freeze_backbone",
                "type": "bool",
                "default": False,
                "description": "Whether to freeze backbone weights",
            },
        ]
        return m

    def test_valid_hyperparameters(self):
        m = self._model_with_spec()
        hp = {"epochs": 10, "learning_rate": 0.01, "optimizer": "Adam", "freeze_backbone": True}
        assert validate_hyperparameters(hp, m) == []

    def test_wrong_type_int(self):
        m = self._model_with_spec()
        errors = validate_hyperparameters({"epochs": "five"}, m)
        assert any("expected type 'int'" in e for e in errors)

    def test_wrong_type_float(self):
        m = self._model_with_spec()
        errors = validate_hyperparameters({"learning_rate": "fast"}, m)
        assert any("expected type 'float'" in e for e in errors)

    def test_int_accepted_as_float(self):
        m = self._model_with_spec()
        assert validate_hyperparameters({"learning_rate": 1}, m) == []

    def test_wrong_type_bool(self):
        m = self._model_with_spec()
        errors = validate_hyperparameters({"freeze_backbone": 1}, m)
        assert any("expected type 'bool'" in e for e in errors)

    def test_invalid_enum_value(self):
        m = self._model_with_spec()
        errors = validate_hyperparameters({"optimizer": "RMSprop"}, m)
        assert any("not in allowed values" in e for e in errors)

    def test_unknown_key_ignored(self):
        m = self._model_with_spec()
        assert validate_hyperparameters({"custom_param": 42}, m) == []

    def test_no_spec_skips_validation(self):
        m = _model()
        assert validate_hyperparameters({"anything": "goes"}, m) == []

    def test_below_minimum(self):
        m = self._model_with_spec()
        errors = validate_hyperparameters({"epochs": 0}, m)
        assert any("below minimum" in e for e in errors)

    def test_above_maximum(self):
        m = self._model_with_spec()
        errors = validate_hyperparameters({"epochs": 999}, m)
        assert any("above maximum" in e for e in errors)

    def test_float_below_minimum(self):
        m = self._model_with_spec()
        errors = validate_hyperparameters({"learning_rate": 1e-9}, m)
        assert any("below minimum" in e for e in errors)

    def test_float_above_maximum(self):
        m = self._model_with_spec()
        errors = validate_hyperparameters({"learning_rate": 5.0}, m)
        assert any("above maximum" in e for e in errors)

    def test_at_boundary_values(self):
        m = self._model_with_spec()
        assert validate_hyperparameters({"epochs": 1}, m) == []
        assert validate_hyperparameters({"epochs": 500}, m) == []
        assert validate_hyperparameters({"learning_rate": 1e-7}, m) == []
        assert validate_hyperparameters({"learning_rate": 1.0}, m) == []


class TestHyperparametersSpecCoverage:
    _FULL_SPEC: ClassVar[list[dict]] = [
        {"key": "epochs", "type": "int", "default": 10, "description": "Number of training epochs"},
        {"key": "batch_size", "type": "int", "default": 4, "description": "Samples per batch"},
        {"key": "learning_rate", "type": "float", "default": 0.001, "description": "Optimizer LR"},
        {"key": "confidence_threshold", "type": "float", "default": 0.5, "description": "Inference threshold"},
    ]

    def _item_with_full_spec(self) -> pystac.Item:
        item = _valid_base_model()
        item.properties["fair:hyperparameters_spec"] = copy.deepcopy(self._FULL_SPEC)
        return item

    def test_valid_coverage_passes(self):
        item = self._item_with_full_spec()
        errors = validate_item(item)
        assert not any("has no matching entry in fair:hyperparameters_spec" in e for e in errors)
        assert not any("missing required fields" in e for e in errors)

    def test_training_key_missing_from_spec(self):
        item = self._item_with_full_spec()
        item.properties["mlm:hyperparameters"]["training.custom_param"] = 99
        errors = validate_item(item)
        assert any("training.custom_param" in e and "has no matching entry" in e for e in errors)

    def test_inference_key_missing_from_spec(self):
        item = self._item_with_full_spec()
        item.properties["mlm:hyperparameters"]["inference.new_param"] = 0.9
        errors = validate_item(item)
        assert any("inference.new_param" in e and "has no matching entry" in e for e in errors)

    def test_spec_entry_missing_type(self):
        item = self._item_with_full_spec()
        del item.properties["fair:hyperparameters_spec"][0]["type"]
        errors = validate_item(item)
        assert any("missing required fields" in e and "type" in e for e in errors)

    def test_spec_entry_missing_default(self):
        item = self._item_with_full_spec()
        del item.properties["fair:hyperparameters_spec"][0]["default"]
        errors = validate_item(item)
        assert any("missing required fields" in e and "default" in e for e in errors)

    def test_spec_entry_missing_description(self):
        item = self._item_with_full_spec()
        del item.properties["fair:hyperparameters_spec"][0]["description"]
        errors = validate_item(item)
        assert any("missing required fields" in e and "description" in e for e in errors)

    def test_no_spec_skips_coverage_check(self):
        item = _valid_base_model()
        item.properties.pop("fair:hyperparameters_spec", None)
        errors = validate_item(item)
        assert not any("has no matching entry in fair:hyperparameters_spec" in e for e in errors)
