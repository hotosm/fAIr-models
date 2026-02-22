from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock, patch

import pystac
import pytest

from fair_models.stac.builders import build_base_model_item
from fair_models.stac.catalog_manager import StacCatalogManager
from fair_models.stac.collections import initialize_catalog
from fair_models.zenml.promotion import (
    archive_model_version,
    delete_model,
    delete_model_version,
    promote_model_version,
    publish_promoted_model,
)


@pytest.fixture()
def cm(tmp_path) -> StacCatalogManager:
    path = str(tmp_path / "catalog.json")
    initialize_catalog(path)
    mgr = StacCatalogManager(path)
    # Seed a base model every test needs
    mgr.publish_item("base-models", _base_model_item())
    return mgr


def _base_model_item() -> pystac.Item:
    return build_base_model_item(
        item_id="example-unet",
        dt=datetime(2024, 1, 1, tzinfo=UTC),
        geometry={"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]},
        mlm_name="example-unet",
        mlm_architecture="UNet",
        mlm_tasks=["semantic-segmentation"],
        mlm_framework="pytorch",
        mlm_framework_version="2.1.0",
        mlm_input=[],
        mlm_output=[],
        mlm_hyperparameters={"epochs": 15, "batch_size": 4},
        keywords=["building", "semantic-segmentation"],
        model_href="s3://weights/unet.pt",
        model_artifact_type="pt",
        mlm_pretrained=True,
        mlm_pretrained_source="OAM-TCD",
        source_code_href="https://github.com/example",
        source_code_entrypoint="mod:train",
        training_runtime_href="ghcr.io/hotosm/fair-unet:v1",
        inference_runtime_href="ghcr.io/hotosm/fair-unet:v1",
    )


def _mock_mv(params: dict[str, Any] | None = None, *, weights_found: bool = True) -> MagicMock:
    mv = MagicMock()
    mv.id = "fake-uuid"
    step = MagicMock()
    step.config.parameters = params or {}
    run = MagicMock()
    run.steps.get.return_value = step
    mv.pipeline_runs = {"run-1": run}
    if weights_found:
        art = MagicMock()
        art.load.return_value = "artifacts/finetuned_weights.pth"
        mv.get_artifact.return_value = art
    else:
        mv.get_artifact.return_value = None
    return mv


def _publish(cm, version=1, **kw):
    """Shortcut for publish_promoted_model with common defaults."""
    defaults = {
        "model_name": "unet-finetuned-banepa",
        "version": version,
        "catalog_manager": cm,
        "base_model_item_id": "example-unet",
        "dataset_item_id": "buildings-banepa",
        "keywords": ["building"],
    }
    defaults.update(kw)
    return publish_promoted_model(**defaults)


@patch("fair_models.zenml.promotion.Client")
def test_promote_sets_stage(mock_cls):
    mv = MagicMock()
    mock_cls.return_value.get_model_version.return_value = mv
    promote_model_version("model", 3)
    mv.set_stage.assert_called_once()


@patch("fair_models.zenml.promotion.Client")
def test_publish_and_deprecate_previous(mock_cls, cm):
    mock_cls.return_value.get_model_version.return_value = _mock_mv({"epochs": 1})

    v1 = _publish(cm, version=1)
    assert v1.id == "unet-finetuned-banepa-v1"
    assert v1.properties["mlm:architecture"] == "UNet"

    _publish(cm, version=2)
    assert cm.get_item("local-models", "unet-finetuned-banepa-v1").extra_fields["deprecated"] is True


@patch("fair_models.zenml.promotion.Client")
def test_missing_weights_raises(mock_cls, cm):
    mock_cls.return_value.get_model_version.return_value = _mock_mv(weights_found=False)
    with pytest.raises(RuntimeError, match="No weights artifact"):
        _publish(cm)


@patch("fair_models.zenml.promotion.Client")
def test_archive(mock_cls, cm):
    mock_cls.return_value.get_model_version.return_value = _mock_mv()
    _publish(cm, version=1)

    mv = MagicMock()
    mock_cls.return_value.get_model_version.return_value = mv
    result = archive_model_version("unet-finetuned-banepa", 1, cm)
    assert result.extra_fields["deprecated"] is True
    mv.set_stage.assert_called_once()


@patch("fair_models.zenml.promotion.Client")
def test_archive_missing_raises(mock_cls, cm):
    mock_cls.return_value.get_model_version.return_value = MagicMock()
    with pytest.raises(KeyError):
        archive_model_version("nope", 99, cm)


@patch("fair_models.zenml.promotion.Client")
def test_delete_version(mock_cls, cm):
    mock_cls.return_value.get_model_version.return_value = _mock_mv()
    _publish(cm, version=1)

    mv = MagicMock()
    mv.id = "fake-uuid"
    mock_cls.return_value.get_model_version.return_value = mv
    delete_model_version("unet-finetuned-banepa", 1, cm)
    with pytest.raises(KeyError):
        cm.get_item("local-models", "unet-finetuned-banepa-v1")


@patch("fair_models.zenml.promotion.Client")
def test_delete_model(mock_cls, cm):
    mock_cls.return_value.get_model_version.return_value = _mock_mv()
    for v in (1, 2):
        _publish(cm, version=v)

    delete_model("unet-finetuned-banepa", cm)
    remaining = [i for i in cm.list_items("local-models") if i.id.startswith("unet-finetuned-banepa-v")]
    assert len(remaining) == 0
    mock_cls.return_value.delete_model.assert_called_once_with("unet-finetuned-banepa")
