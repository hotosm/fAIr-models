from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import httpx
import pystac

from fair.client import FairClient
from fair.infra.knative import public_predict_url
from fair.stac.constants import BASE_MODELS_COLLECTION


class _StubBackend:
    def __init__(self, item: pystac.Item) -> None:
        self._item = item

    def get_item(self, _collection: str, _item_id: str) -> pystac.Item:
        return self._item


class _StubResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return {"status": "ok"}


def _build_base_model_item(*, with_endpoint: bool = True) -> pystac.Item:
    item = pystac.Item(
        id="resnet18-classification",
        geometry={
            "type": "Polygon",
            "coordinates": [[[-180, -90], [180, -90], [180, 90], [-180, 90], [-180, -90]]],
        },
        bbox=[-180, -90, 180, 90],
        datetime=datetime.now(UTC),
        properties={
            "mlm:name": "resnet18-classification",
            "mlm:hyperparameters": {"inference.confidence_threshold": 0.5},
        },
    )
    item.add_asset("model", pystac.Asset(href="https://example.com/model.onnx"))
    if with_endpoint:
        item.add_asset(
            "mlm:inference-endpoint",
            pystac.Asset(
                href="https://resnet18-classification.predict.fair.example.com/predict",
                media_type="application/json",
                roles=["mlm:inference-endpoint"],
            ),
        )
    return item


def test_public_predict_url_matches_cluster_routing_convention() -> None:
    # Pins the URL shape served by infra/manifests/ingress.yaml.gotmpl +
    # KnativeServing config-domain. Changing either side without the other
    # will route predictions to a nonexistent host.
    assert (
        public_predict_url("resnet18-classification", "fair.example.com")
        == "https://resnet18-classification.predict.fair.example.com/predict"
    )


def test_predict_live_reads_endpoint_from_stac(monkeypatch) -> None:
    item = _build_base_model_item()
    client = FairClient(stac_api_url="https://stac.example.com", dsn="postgresql://example")
    monkeypatch.setattr(client, "_get_backend", lambda: _StubBackend(item))
    monkeypatch.setenv("ZENML_STORE_VERIFY_SSL", "false")

    captured: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: Any) -> _StubResponse:
        captured["url"] = url
        captured["kwargs"] = kwargs
        return _StubResponse()

    monkeypatch.setattr(httpx, "post", fake_post)

    result = client.predict_live(
        "resnet18-classification",
        image_uri="https://tiles.openaerialmap.org/abc/{z}/{x}/{y}",
        bbox=[85.5, 27.6, 85.52, 27.63],
        zoom=18,
        collection=BASE_MODELS_COLLECTION,
    )

    assert result == {"status": "ok"}
    assert captured["url"] == "https://resnet18-classification.predict.fair.example.com/predict"
    assert captured["kwargs"]["verify"] is False
    sent = captured["kwargs"]["json"]
    assert sent["params"] == {"confidence_threshold": 0.5}
    assert sent["image_uri"] == "https://tiles.openaerialmap.org/abc/{z}/{x}/{y}"
    assert sent["bbox"] == [85.5, 27.6, 85.52, 27.63]
    assert sent["zoom"] == 18
    assert sent["model_uri"] == "https://example.com/model.onnx"


def test_predict_live_explicit_override_wins_over_stac(monkeypatch) -> None:
    item = _build_base_model_item()
    client = FairClient(stac_api_url="https://stac.example.com", dsn="postgresql://example")
    monkeypatch.setattr(client, "_get_backend", lambda: _StubBackend(item))
    monkeypatch.setenv("ZENML_STORE_VERIFY_SSL", "false")

    captured: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: Any) -> _StubResponse:
        captured["url"] = url
        return _StubResponse()

    monkeypatch.setattr(httpx, "post", fake_post)

    client.predict_live(
        "resnet18-classification",
        image_uri="https://tiles.openaerialmap.org/abc/{z}/{x}/{y}",
        bbox=[85.5, 27.6, 85.52, 27.63],
        zoom=18,
        predict_base_url="http://127.0.0.1:8080/predict",
        collection=BASE_MODELS_COLLECTION,
    )
    assert captured["url"] == "http://127.0.0.1:8080/predict"


def test_predict_live_missing_endpoint_asset_raises(monkeypatch) -> None:
    import pytest

    from fair.client import FairClientError

    item = _build_base_model_item(with_endpoint=False)
    client = FairClient(stac_api_url="https://stac.example.com", dsn="postgresql://example")
    monkeypatch.setattr(client, "_get_backend", lambda: _StubBackend(item))

    with pytest.raises(FairClientError, match="mlm:inference-endpoint"):
        client.predict_live(
            "resnet18-classification",
            image_uri="https://tiles.openaerialmap.org/abc/{z}/{x}/{y}",
            bbox=[85.5, 27.6, 85.52, 27.63],
            zoom=18,
            collection=BASE_MODELS_COLLECTION,
        )
