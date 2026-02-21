"""Tests for fair_models.zenml.promotion -- ZenML + STAC sync operations.

All ZenML Client calls are mocked. StacCatalogManager uses a real tmp_path
catalog so STAC side-effects are fully exercised.
"""

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

# -- fixtures


@pytest.fixture()
def catalog_manager(tmp_path) -> StacCatalogManager:
    path = str(tmp_path / "catalog.json")
    initialize_catalog(path)
    return StacCatalogManager(path)


def _base_model_item() -> pystac.Item:
    return build_base_model_item(
        item_id="example-unet",
        geometry={"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]},
        dt=datetime(2024, 1, 1, tzinfo=UTC),
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
        source_code_href="https://github.com/hotosm/fAIr-models/tree/main/models/example_unet",
        source_code_entrypoint="models.example_unet.pipeline:train",
        training_runtime_href="ghcr.io/hotosm/fair-unet:v1",
        inference_runtime_href="ghcr.io/hotosm/fair-unet:v1",
    )


def _seed_base_model(cm: StacCatalogManager) -> pystac.Item:
    """Insert base model into catalog so promotion can read it."""
    item = _base_model_item()
    return cm.publish_item("base-models", item)


def _mock_model_version(run_metadata: dict[str, Any] | None = None) -> MagicMock:
    mv = MagicMock()
    mv.id = "fake-uuid-1234"
    mv.run_metadata = {}
    if run_metadata:
        for k, v in run_metadata.items():
            m = MagicMock()
            m.value = v
            mv.run_metadata[k] = m
    mv.get_model_artifact.return_value = None
    return mv


# -- tests


class TestPromoteModelVersion:
    @patch("fair_models.zenml.promotion.Client")
    def test_sets_stage_production(self, mock_client_cls):
        mock_client = mock_client_cls.return_value
        mv = MagicMock()
        mock_client.get_model_version.return_value = mv

        promote_model_version("unet-finetuned-banepa", 3)

        mock_client.get_model_version.assert_called_once_with("unet-finetuned-banepa", 3)
        mv.set_stage.assert_called_once()


class TestPublishPromotedModel:
    @patch("fair_models.zenml.promotion.Client")
    def test_publishes_new_item(self, mock_client_cls, catalog_manager):
        mock_client = mock_client_cls.return_value
        mock_client.get_model_version.return_value = _mock_model_version(
            {"mlm:hyperparameters": {"epochs": 1, "batch_size": 4}}
        )
        _seed_base_model(catalog_manager)

        item = publish_promoted_model(
            model_name="unet-finetuned-banepa",
            version=1,
            catalog_manager=catalog_manager,
            base_model_item_id="example-unet",
            dataset_item_id="buildings-banepa",
            keywords=["building"],
        )

        assert item.id == "unet-finetuned-banepa-v1"
        retrieved = catalog_manager.get_item("local-models", "unet-finetuned-banepa-v1")
        assert retrieved.properties["version"] == "1"

    @patch("fair_models.zenml.promotion.Client")
    def test_deprecates_previous_version(self, mock_client_cls, catalog_manager):
        mock_client = mock_client_cls.return_value
        mock_client.get_model_version.return_value = _mock_model_version()
        _seed_base_model(catalog_manager)

        # Publish v1 first
        publish_promoted_model(
            model_name="unet-finetuned-banepa",
            version=1,
            catalog_manager=catalog_manager,
            base_model_item_id="example-unet",
            dataset_item_id="buildings-banepa",
            keywords=["building"],
        )

        # Publish v2 -- should deprecate v1
        publish_promoted_model(
            model_name="unet-finetuned-banepa",
            version=2,
            catalog_manager=catalog_manager,
            base_model_item_id="example-unet",
            dataset_item_id="buildings-banepa",
            keywords=["building"],
        )

        v1 = catalog_manager.get_item("local-models", "unet-finetuned-banepa-v1")
        assert v1.extra_fields.get("deprecated") is True

    @patch("fair_models.zenml.promotion.Client")
    def test_copies_mlm_fields_from_base(self, mock_client_cls, catalog_manager):
        mock_client = mock_client_cls.return_value
        mock_client.get_model_version.return_value = _mock_model_version()
        _seed_base_model(catalog_manager)

        item = publish_promoted_model(
            model_name="unet-finetuned-banepa",
            version=1,
            catalog_manager=catalog_manager,
            base_model_item_id="example-unet",
            dataset_item_id="buildings-banepa",
        )

        assert item.properties["mlm:architecture"] == "UNet"
        assert item.properties["mlm:framework"] == "pytorch"


class TestArchiveModelVersion:
    @patch("fair_models.zenml.promotion.Client")
    def test_archives_and_deprecates(self, mock_client_cls, catalog_manager):
        mock_client = mock_client_cls.return_value
        mv = MagicMock()
        mock_client.get_model_version.return_value = mv

        # Seed a local-model item manually
        _seed_base_model(catalog_manager)
        pub_mv = _mock_model_version()
        mock_client.get_model_version.return_value = pub_mv
        with patch("fair_models.zenml.promotion.Client", return_value=mock_client):
            publish_promoted_model(
                model_name="unet-finetuned-banepa",
                version=1,
                catalog_manager=catalog_manager,
                base_model_item_id="example-unet",
                dataset_item_id="buildings-banepa",
                keywords=["building"],
            )

        # Now archive
        mock_client.get_model_version.return_value = mv
        result = archive_model_version("unet-finetuned-banepa", 1, catalog_manager)

        assert result is not None
        assert result.extra_fields.get("deprecated") is True
        mv.set_stage.assert_called_once()

    @patch("fair_models.zenml.promotion.Client")
    def test_archive_missing_stac_item_raises(self, mock_client_cls, catalog_manager):
        mv = MagicMock()
        mock_client_cls.return_value.get_model_version.return_value = mv

        with pytest.raises(KeyError):
            archive_model_version("nonexistent-model", 99, catalog_manager)


class TestDeleteModelVersion:
    @patch("fair_models.zenml.promotion.Client")
    def test_deletes_zenml_and_stac(self, mock_client_cls, catalog_manager):
        mock_client = mock_client_cls.return_value
        mv = MagicMock()
        mv.id = "fake-uuid"
        mock_client.get_model_version.return_value = mv

        # Seed item
        _seed_base_model(catalog_manager)
        pub_mv = _mock_model_version()
        mock_client.get_model_version.return_value = pub_mv
        with patch("fair_models.zenml.promotion.Client", return_value=mock_client):
            publish_promoted_model(
                model_name="unet-finetuned-banepa",
                version=1,
                catalog_manager=catalog_manager,
                base_model_item_id="example-unet",
                dataset_item_id="buildings-banepa",
                keywords=["building"],
            )

        mock_client.get_model_version.return_value = mv
        delete_model_version("unet-finetuned-banepa", 1, catalog_manager)

        mock_client.delete_model_version.assert_called_once_with("fake-uuid")
        with pytest.raises(KeyError):
            catalog_manager.get_item("local-models", "unet-finetuned-banepa-v1")


class TestDeleteModel:
    @patch("fair_models.zenml.promotion.Client")
    def test_deletes_all_items_and_zenml_model(self, mock_client_cls, catalog_manager):
        mock_client = mock_client_cls.return_value
        mock_client.get_model_version.return_value = _mock_model_version()
        _seed_base_model(catalog_manager)

        # Publish v1 and v2
        for v in (1, 2):
            publish_promoted_model(
                model_name="unet-finetuned-banepa",
                version=v,
                catalog_manager=catalog_manager,
                base_model_item_id="example-unet",
                dataset_item_id="buildings-banepa",
                keywords=["building"],
            )

        delete_model("unet-finetuned-banepa", catalog_manager)

        mock_client.delete_model.assert_called_once_with("unet-finetuned-banepa")
        items = catalog_manager.list_items("local-models")
        matching = [i for i in items if i.id.startswith("unet-finetuned-banepa-v")]
        assert len(matching) == 0
