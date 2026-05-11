from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from fair.client import FairClient, UserScopedFairClient


def _client_with_stub_backend() -> tuple[FairClient, MagicMock]:
    client = FairClient.__new__(FairClient)
    client._zenml_store_url = None
    client._stac_api_url = None
    client._stac_api_key = None
    client._dsn = None
    client._catalog_path = "stac_catalog/catalog.json"
    client.user_id = "system"
    client._upload_artifacts = False
    backend_mock = MagicMock()
    client._cached_backend = backend_mock
    return client, backend_mock


def test_with_user_returns_proxy_with_correct_user() -> None:
    client, _ = _client_with_stub_backend()
    proxy = client.with_user("alice")
    assert isinstance(proxy, UserScopedFairClient)
    assert proxy.user_id == "alice"


def test_proxy_register_dataset_stamps_user_id() -> None:
    client, _ = _client_with_stub_backend()
    register_dataset = MagicMock(return_value="ds-1")
    client.register_dataset = register_dataset  # type: ignore[method-assign]
    proxy = client.with_user("alice")

    proxy.register_dataset("path/to/item.json")

    register_dataset.assert_called_once_with("path/to/item.json", user_id="alice", paths=None)


def test_proxy_submit_finetune_delegates() -> None:
    client, _ = _client_with_stub_backend()
    submit_finetune = MagicMock(return_value="run-id")
    client.submit_finetune = submit_finetune  # type: ignore[method-assign]
    proxy = client.with_user("bob")

    proxy.submit_finetune(
        base_model_id="b-1",
        dataset_id="d-1",
        model_name="m-1",
        overrides={"epochs": 3},
    )

    submit_finetune.assert_called_once_with(
        base_model_id="b-1",
        dataset_id="d-1",
        model_name="m-1",
        overrides={"epochs": 3},
        config_dir=None,
    )


def test_proxy_submit_predict_delegates() -> None:
    client, _ = _client_with_stub_backend()
    submit_predict = MagicMock(return_value="run-id")
    client.submit_predict = submit_predict  # type: ignore[method-assign]
    proxy = client.with_user("carol")

    proxy.submit_predict(local_model_id="lm-1", image_path="s3://bucket/img")

    submit_predict.assert_called_once_with(
        local_model_id="lm-1",
        image_path="s3://bucket/img",
        config_dir=None,
    )


def test_proxy_promote_stamps_user_id() -> None:
    client, _ = _client_with_stub_backend()
    promote = MagicMock(return_value="lm-1")
    client.promote = promote  # type: ignore[method-assign]
    proxy = client.with_user("dave")

    proxy.promote("ft-1", description="Looks good")

    promote.assert_called_once_with(
        "ft-1",
        base_model_id=None,
        dataset_id=None,
        description="Looks good",
        title=None,
        keywords=None,
        user_id="dave",
        pipeline_run_id=None,
        paths=None,
    )


def test_proxy_register_base_model_does_not_stamp() -> None:
    client, _ = _client_with_stub_backend()
    register_base_model = MagicMock(return_value="bm-1")
    client.register_base_model = register_base_model  # type: ignore[method-assign]
    proxy = client.with_user("ignored")

    proxy.register_base_model("path/to/base.json")

    register_base_model.assert_called_once_with("path/to/base.json")


def test_proxy_get_item_delegates_to_backend() -> None:
    client, backend_mock = _client_with_stub_backend()
    backend_mock.get_item.return_value = "item-stub"
    proxy = client.with_user("alice")

    result = proxy.get_item("datasets", "ds-1")

    assert result == "item-stub"
    backend_mock.get_item.assert_called_once_with("datasets", "ds-1")


def test_proxy_list_items_passes_limit() -> None:
    client, backend_mock = _client_with_stub_backend()
    backend_mock.list_items.return_value = []
    proxy = client.with_user("alice")

    proxy.list_items("base-models", limit=5)

    backend_mock.list_items.assert_called_once_with("base-models", limit=5)


def test_get_backend_caches_first_call(monkeypatch: pytest.MonkeyPatch) -> None:
    backend_init_count = {"calls": 0}

    class _SpyBackend:
        def __init__(self, *_: object, **__: object) -> None:
            backend_init_count["calls"] += 1

    monkeypatch.setattr("fair.stac.api_backend.StacApiBackend", _SpyBackend)

    client = FairClient.__new__(FairClient)
    client._zenml_store_url = None
    client._stac_api_url = "https://stac.example/api"
    client._stac_api_key = None
    client._dsn = None
    client._catalog_path = "stac_catalog/catalog.json"
    client.user_id = "system"
    client._upload_artifacts = False
    client._cached_backend = None

    first = client._get_backend()
    second = client._get_backend()
    third = client._get_backend()

    assert first is second is third
    assert backend_init_count["calls"] == 1
