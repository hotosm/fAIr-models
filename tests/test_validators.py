from __future__ import annotations

import json
from datetime import UTC, datetime
from unittest.mock import patch

import pystac
import pytest

from fair_models.stac.builders import build_base_model_item, build_dataset_item
from fair_models.stac.validators import validate_compatibility, validate_mlm_schema


def _model(keywords=None, tasks=None):
    return build_base_model_item(
        item_id="m",
        dt=datetime(2024, 1, 1, tzinfo=UTC),
        geometry={"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]},
        mlm_name="m",
        mlm_architecture="UNet",
        mlm_tasks=tasks or ["semantic-segmentation"],
        mlm_framework="pytorch",
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
