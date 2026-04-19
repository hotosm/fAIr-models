from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock, patch

import pystac
import pytest

from fair.stac.builders import build_base_model_item, build_dataset_item
from fair.stac.catalog_manager import StacCatalogManager
from fair.stac.collections import initialize_catalog
from fair.zenml.promotion import (
    archive_model_version,
    delete_model,
    delete_model_version,
    promote_model_version,
    publish_promoted_model,
)


@pytest.fixture(autouse=True)
def _skip_url_validation(monkeypatch):
    monkeypatch.setattr("fair.zenml.promotion.validate_model_asset_urls", lambda *_a, **_kw: [])


@pytest.fixture(autouse=True)
def _stub_artifact_materializers(monkeypatch):
    monkeypatch.setattr("fair.zenml.promotion._materialize_onnx_bytes", lambda _a: b"onnx-bytes")
    monkeypatch.setattr("fair.zenml.promotion._materialize_checkpoint_bytes", lambda _a: b"pt-bytes")


@pytest.fixture()
def cm(tmp_path) -> StacCatalogManager:
    path = str(tmp_path / "catalog.json")
    initialize_catalog(path)
    mgr = StacCatalogManager(path)
    mgr.publish_item("base-models", _base_model_item())
    mgr.publish_item("datasets", _dataset_item())
    return mgr


def _base_model_item() -> pystac.Item:
    return build_base_model_item(
        item_id="example-unet",
        geometry={"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]},
        mlm_name="example-unet",
        mlm_architecture="UNet",
        mlm_tasks=["semantic-segmentation"],
        mlm_framework="pytorch",
        mlm_framework_version="2.1.0",
        mlm_input=[],
        mlm_output=[],
        mlm_hyperparameters={
            "training.epochs": 15,
            "training.batch_size": 4,
            "training.learning_rate": 0.0001,
            "inference.confidence_threshold": 0.5,
        },
        keywords=["building", "semantic-segmentation"],
        checkpoint_href="s3://weights/unet.pt",
        checkpoint_artifact_type="torch.save",
        mlm_pretrained=True,
        mlm_pretrained_source="OAM-TCD",
        source_code_href="https://github.com/example",
        source_code_entrypoint="mod:train",
        training_runtime_href="ghcr.io/hotosm/fair-unet:v1",
        inference_runtime_href="ghcr.io/hotosm/fair-unet:v1",
        title="Example UNet",
        description="UNet for building segmentation.",
        fair_metrics_spec=[{"name": "accuracy", "description": "Pixel accuracy", "higher_is_better": True}],
        providers=[{"name": "HOTOSM", "roles": ["producer"], "url": "https://www.hotosm.org"}],
    )


def _dataset_item() -> pystac.Item:
    return build_dataset_item(
        item_id="dataset-fixed-uuid",
        label_type="vector",
        label_tasks=["segmentation"],
        label_classes=[{"name": "building", "classes": ["building"]}],
        keywords=["building"],
        chips_href="chips/",
        labels_href="labels.geojson",
        title="buildings-banepa",
        description="Test dataset.",
        user_id="osm-test",
        providers=[{"name": "osm-test", "roles": ["producer"]}],
        geometry={
            "type": "Polygon",
            "coordinates": [[[85.51, 27.63], [85.53, 27.63], [85.53, 27.64], [85.51, 27.64], [85.51, 27.63]]],
        },
        bbox=[85.51, 27.63, 85.53, 27.64],
    )


def _mock_mv(params: dict[str, Any] | None = None, *, weights_found: bool = True) -> tuple[MagicMock, MagicMock]:
    mv = MagicMock()
    mv.id = "fake-uuid"
    mv.run_metadata = {}

    step = MagicMock()
    step.config.parameters = params or {}
    step.start_time = datetime(2024, 6, 1, 12, 0, tzinfo=UTC)
    step.end_time = datetime(2024, 6, 1, 13, 30, tzinfo=UTC)
    run = MagicMock()
    run.steps.get.return_value = step
    run.config.parameters = params or {}

    client = MagicMock()
    run_link = MagicMock()
    run_link.pipeline_run = run
    page = MagicMock()
    page.items = [run_link]
    client.list_model_version_pipeline_run_links.return_value = page

    if weights_found:
        weights_art = MagicMock()
        weights_art.uri = "s3://artifact-store/model/output/abc123"
        weights_art.id = "artifact-version-uuid-001"
        onnx_art = MagicMock()
        onnx_art.uri = "s3://artifact-store/model/output/abc123-onnx"
        onnx_art.id = "artifact-version-uuid-002"
        mv.get_artifact.side_effect = lambda name: {"trained_model": weights_art, "onnx_model": onnx_art}.get(name)
    else:
        mv.get_artifact.return_value = None
    return mv, client


def _publish(cm: StacCatalogManager, version: int = 1, **kw: Any) -> pystac.Item:
    return publish_promoted_model(
        model_name=kw.get("model_name", "unet-finetuned-banepa"),
        version=kw.get("version", version),
        catalog_manager=kw.get("catalog_manager", cm),
        base_model_item_id=kw.get("base_model_item_id", "example-unet"),
        dataset_item_id=kw.get("dataset_item_id", "dataset-fixed-uuid"),
        user_id=kw.get("user_id", "osm-test"),
        description=kw.get("description", "Test model."),
        keywords=kw.get("keywords", ["building"]),
        geometry=kw.get("geometry"),
    )


@patch("fair.zenml.promotion.Client")
def test_promote_sets_stage(mock_cls):
    mv = MagicMock()
    mock_cls.return_value.get_model_version.return_value = mv
    promote_model_version("model", 3)
    mv.set_stage.assert_called_once()


@patch("fair.zenml.promotion.Client")
def test_publish_and_deprecate_previous(mock_cls, cm):
    mv, client = _mock_mv({"epochs": 1})
    mock_cls.return_value = client
    client.get_model_version.return_value = mv

    v1 = _publish(cm, version=1)
    assert v1.id == "fake-uuid"
    assert v1.properties["mlm:architecture"] == "UNet"

    mv2, client2 = _mock_mv({"epochs": 2})
    mv2.id = "fake-uuid-v2"
    mv2.run_metadata = {}
    mock_cls.return_value = client2
    client2.get_model_version.return_value = mv2
    _publish(cm, version=2)
    deprecated = cm.get_item("local-models", "fake-uuid")
    assert deprecated.properties["deprecated"] is True
    latest_links = [lnk for lnk in deprecated.links if lnk.rel == "latest-version"]
    assert len(latest_links) == 0
    successor_links = [lnk for lnk in deprecated.links if lnk.rel == "successor-version"]
    assert len(successor_links) == 1


@patch("fair.zenml.promotion.Client")
def test_publish_stores_training_timing(mock_cls, cm):
    mv, client = _mock_mv({"epochs": 1})
    mock_cls.return_value = client
    client.get_model_version.return_value = mv
    item = _publish(cm, version=1)
    assert "fair:training_started_at" in item.properties
    assert "fair:training_ended_at" in item.properties


@patch("fair.zenml.promotion.Client")
def test_publish_uses_wall_time_for_duration(mock_cls, cm):
    mv, client = _mock_mv({"epochs": 1})
    mv.run_metadata = {"fair/training_wall_seconds": 42.5}
    mock_cls.return_value = client
    client.get_model_version.return_value = mv
    item = _publish(cm, version=1)
    assert item.properties["fair:training_duration_seconds"] == 42.5


@patch("fair.zenml.promotion.Client")
def test_publish_stores_fair_metrics(mock_cls, cm):
    mv, client = _mock_mv({"epochs": 1})
    mv.run_metadata = {"fair/accuracy": 0.95, "fair/mean_iou": 0.80, "other_key": "ignored"}
    mock_cls.return_value = client
    client.get_model_version.return_value = mv
    item = _publish(cm, version=1)
    assert item.properties["fair:accuracy"] == 0.95
    assert item.properties["fair:mean_iou"] == 0.80
    assert "other_key" not in item.properties


@patch("fair.zenml.promotion.Client")
def test_publish_stores_artifact_metadata(mock_cls, cm):
    """Checkpoint and model assets must carry materialized hrefs and the ZenML artifact version ID."""
    mv, client = _mock_mv({"epochs": 1})
    mock_cls.return_value = client
    client.get_model_version.return_value = mv
    item = _publish(cm, version=1)
    checkpoint = item.assets["checkpoint"]
    assert checkpoint.href.endswith(".pt")
    assert checkpoint.extra_fields["zenml:artifact_version_id"] == "artifact-version-uuid-001"
    onnx = item.assets["model"]
    assert onnx.href.endswith(".onnx")
    assert onnx.extra_fields["zenml:artifact_version_id"] == "artifact-version-uuid-001"


@patch("fair.zenml.promotion.Client")
def test_publish_stores_training_metrics_asset(mock_cls, cm):
    mv, client = _mock_mv({"epochs": 2})
    mv.run_metadata = {
        "fair/loss_history": {"train_loss": [0.5, 0.3], "val_loss": [0.6, 0.4]},
    }
    mock_cls.return_value = client
    client.get_model_version.return_value = mv
    item = _publish(cm, version=1)
    assert "training-metrics" in item.assets
    asset = item.assets["training-metrics"]
    assert asset.media_type == "application/json"
    assert asset.href.endswith(".json")


@patch("fair.zenml.promotion.Client")
def test_publish_omits_training_metrics_when_absent(mock_cls, cm):
    mv, client = _mock_mv({"epochs": 1})
    mock_cls.return_value = client
    client.get_model_version.return_value = mv
    item = _publish(cm, version=1)
    assert "training-metrics" not in item.assets


@patch("fair.zenml.promotion.Client")
def test_missing_weights_raises(mock_cls, cm):
    mv, client = _mock_mv(weights_found=False)
    mock_cls.return_value = client
    client.get_model_version.return_value = mv
    with pytest.raises(RuntimeError, match="No 'trained_model' artifact"):
        _publish(cm)


@patch("fair.zenml.promotion.Client")
def test_archive(mock_cls, cm):
    mv, client = _mock_mv()
    mock_cls.return_value = client
    client.get_model_version.return_value = mv
    _publish(cm, version=1)

    mv2 = MagicMock()
    mv2.id = "fake-uuid"
    mock_cls.return_value.get_model_version.return_value = mv2
    result = archive_model_version("unet-finetuned-banepa", 1, cm)
    assert result.properties["deprecated"] is True
    mv2.set_stage.assert_called_once()


@patch("fair.zenml.promotion.Client")
def test_archive_missing_raises(mock_cls, cm):
    mock_cls.return_value.get_model_version.return_value = MagicMock()
    with pytest.raises(KeyError):
        archive_model_version("nope", 99, cm)


@patch("fair.zenml.promotion.Client")
def test_delete_version(mock_cls, cm):
    mv, client = _mock_mv()
    mock_cls.return_value = client
    client.get_model_version.return_value = mv
    _publish(cm, version=1)

    mv2 = MagicMock()
    mv2.id = "fake-uuid"
    mock_cls.return_value.get_model_version.return_value = mv2
    delete_model_version("unet-finetuned-banepa", 1, cm)
    with pytest.raises(KeyError):
        cm.get_item("local-models", "fake-uuid")


@patch("fair.zenml.promotion.Client")
def test_duplicate_promotion_returns_existing(mock_cls, cm):
    mv, client = _mock_mv({"epochs": 1})
    mock_cls.return_value = client
    client.get_model_version.return_value = mv
    first = _publish(cm, version=1)
    second = _publish(cm, version=1)
    assert second.id == first.id


@patch("fair.zenml.promotion.Client")
def test_delete_model(mock_cls, cm):
    mv, client = _mock_mv()
    mock_cls.return_value = client
    client.get_model_version.return_value = mv
    _publish(cm, version=1)

    mv2, client2 = _mock_mv()
    mv2.id = "fake-uuid-v2"
    mv2.run_metadata = {}
    mock_cls.return_value = client2
    client2.get_model_version.return_value = mv2
    _publish(cm, version=2)

    delete_model("unet-finetuned-banepa", cm)
    remaining = [i for i in cm.list_items("local-models") if i.properties.get("mlm:name") == "unet-finetuned-banepa"]
    assert len(remaining) == 0
    mock_cls.return_value.delete_model.assert_called_once_with("unet-finetuned-banepa")
