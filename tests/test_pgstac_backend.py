"""Unit tests for PgStacBackend with mocked pypgstac and pystac-client."""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pystac
import pytest

# Inject mock modules before importing pgstac_backend
_mock_pypgstac_db = MagicMock()
_mock_pypgstac_load = MagicMock()
_mock_pystac_client = MagicMock()
sys.modules.setdefault("pypgstac", MagicMock())
sys.modules.setdefault("pypgstac.db", _mock_pypgstac_db)
sys.modules.setdefault("pypgstac.load", _mock_pypgstac_load)
sys.modules.setdefault("pystac_client", _mock_pystac_client)


def _make_item(item_id: str = "test-item") -> pystac.Item:
    return pystac.Item(
        item_id,
        geometry={"type": "Point", "coordinates": [0, 0]},
        bbox=[0, 0, 0, 0],
        datetime=datetime(2024, 1, 1, tzinfo=UTC),
        properties={},
    )


@pytest.fixture()
def _mock_deps():
    """Set up mock PgstacDB and Loader for each test."""
    mock_db_instance = MagicMock()
    mock_db_instance.__enter__ = MagicMock(return_value=mock_db_instance)
    mock_db_instance.__exit__ = MagicMock(return_value=False)

    mock_loader_instance = MagicMock()

    _mock_pypgstac_db.PgstacDB.return_value = mock_db_instance
    _mock_pypgstac_load.Loader.return_value = mock_loader_instance
    # Expose Methods.upsert for assertions
    _mock_pypgstac_load.Methods = MagicMock()
    _mock_pypgstac_load.Methods.upsert = "upsert"

    yield mock_db_instance, mock_loader_instance
    _mock_pypgstac_db.reset_mock()
    _mock_pypgstac_load.reset_mock()


@pytest.fixture()
def backend(_mock_deps):
    from fair.stac.pgstac_backend import PgStacBackend

    return PgStacBackend(dsn="postgresql://u:p@localhost/db", stac_api_url="http://localhost:8082")


class TestBootstrap:
    def test_loads_three_collections(self, _mock_deps, backend):
        _, mock_loader = _mock_deps
        # __init__ calls _bootstrap_collections -> loader.load_collections
        assert mock_loader.load_collections.call_count == 1
        call_args = mock_loader.load_collections.call_args
        assert call_args[1].get("insert_mode") == "upsert" or call_args[0][1] == "upsert"


class TestPublishItem:
    def test_upserts_item(self, _mock_deps, backend):
        _, mock_loader = _mock_deps
        mock_loader.reset_mock()

        item = _make_item()
        result = backend.publish_item("base-models", item)

        assert result is item
        assert mock_loader.load_items.call_count == 1

    def test_sets_collection_on_item_dict(self, _mock_deps, backend):
        _, mock_loader = _mock_deps
        mock_loader.reset_mock()

        item = _make_item()
        backend.publish_item("datasets", item)

        # Verify the iterator passed to load_items produces dicts with correct collection
        call_args = mock_loader.load_items.call_args
        items_iter = call_args[0][0]
        loaded = list(items_iter)
        assert loaded[0]["collection"] == "datasets"


class TestGetItem:
    @patch("fair.stac.pgstac_backend.StacClient")
    def test_returns_item(self, mock_client_cls, _mock_deps, backend):
        expected = _make_item("found")
        mock_collection = MagicMock()
        mock_collection.get_item.return_value = expected
        mock_client = MagicMock()
        mock_client.get_collection.return_value = mock_collection
        mock_client_cls.open.return_value = mock_client

        result = backend.get_item("base-models", "found")
        assert result.id == "found"

    @patch("fair.stac.pgstac_backend.StacClient")
    def test_raises_on_missing(self, mock_client_cls, _mock_deps, backend):
        mock_collection = MagicMock()
        mock_collection.get_item.return_value = None
        mock_client = MagicMock()
        mock_client.get_collection.return_value = mock_collection
        mock_client_cls.open.return_value = mock_client

        with pytest.raises(KeyError, match="not found"):
            backend.get_item("base-models", "missing")


class TestListItems:
    @patch("fair.stac.pgstac_backend.StacClient")
    def test_returns_list(self, mock_client_cls, _mock_deps, backend):
        items = [_make_item("a"), _make_item("b")]
        mock_search = MagicMock()
        mock_search.items.return_value = iter(items)
        mock_client = MagicMock()
        mock_client.search.return_value = mock_search
        mock_client_cls.open.return_value = mock_client

        result = backend.list_items("base-models")
        assert len(result) == 2


class TestDeprecateItem:
    @patch("fair.stac.pgstac_backend.StacClient")
    def test_sets_deprecated_and_upserts(self, mock_client_cls, _mock_deps, backend):
        _, mock_loader = _mock_deps

        item = _make_item("dep")
        mock_collection = MagicMock()
        mock_collection.get_item.return_value = item
        mock_client = MagicMock()
        mock_client.get_collection.return_value = mock_collection
        mock_client_cls.open.return_value = mock_client

        mock_loader.reset_mock()
        result = backend.deprecate_item("base-models", "dep")

        assert result.properties["deprecated"] is True
        assert mock_loader.load_items.call_count == 1


class TestDeleteItem:
    def test_calls_delete_function(self, _mock_deps, backend):
        mock_db, _ = _mock_deps
        mock_db.reset_mock()

        backend.delete_item("base-models", "del-me")

        mock_db.query_one.assert_called_once()
        sql_arg = mock_db.query_one.call_args[0][0]
        assert "delete_item" in sql_arg
        assert mock_db.query_one.call_args[0][1] == ["del-me", "base-models"]
