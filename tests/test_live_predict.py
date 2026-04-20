from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import httpx
import pystac

from fair.client import FairClient
from fair.infra.knative import build_predict_gateway_config
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


def _build_base_model_item() -> pystac.Item:
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
    return item


def test_build_predict_gateway_config_routes_each_service() -> None:
    config = build_predict_gateway_config(["resnet18-classification", "unet-segmentation"])

    assert "location ~ ^/resnet18-classification(/|$)(.*)$" in config
    assert "proxy_pass http://resnet18-classification.predict.svc.cluster.local;" in config
    assert "location ~ ^/unet-segmentation(/|$)(.*)$" in config
    assert "location = /health" in config
    assert "location = /models" in config
    assert "Available models" in config
    assert "/resnet18-classification/health" in config
    assert '"name": "resnet18-classification"' in config


def test_predict_live_routes_with_knative_host_header(monkeypatch) -> None:
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
        image_path="data/sample/predict/oam",
        predict_base_url="https://predict.example.com",
        collection=BASE_MODELS_COLLECTION,
    )

    assert result == {"status": "ok"}
    assert captured["url"] == "https://predict.example.com/resnet18-classification/predict"
    assert "headers" not in captured["kwargs"]
    assert captured["kwargs"]["verify"] is False
    assert captured["kwargs"]["json"]["params"] == {"confidence_threshold": 0.5}


def test_predict_live_uses_public_model_domain_when_configured(monkeypatch) -> None:
    item = _build_base_model_item()
    client = FairClient(stac_api_url="https://stac.example.com", dsn="postgresql://example")
    monkeypatch.setattr(client, "_get_backend", lambda: _StubBackend(item))
    monkeypatch.setenv("FAIR_LABEL_DOMAIN", "fair.example.com")
    monkeypatch.setenv("ZENML_STORE_VERIFY_SSL", "false")

    captured: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: Any) -> _StubResponse:
        captured["url"] = url
        captured["kwargs"] = kwargs
        return _StubResponse()

    monkeypatch.setattr(httpx, "post", fake_post)

    result = client.predict_live(
        "resnet18-classification",
        image_path="data/sample/predict/oam",
        collection=BASE_MODELS_COLLECTION,
    )

    assert result == {"status": "ok"}
    assert captured["url"] == "https://predict.fair.example.com/resnet18-classification/predict"
    assert "headers" not in captured["kwargs"]
