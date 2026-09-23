from __future__ import annotations

import copy
from datetime import UTC, datetime
from typing import Any

import pystac
import pytest
from kubernetes.client.exceptions import ApiException

import fair.infra.knative as kn
from fair.client import FairClient
from fair.stac.constants import BASE_MODELS_COLLECTION

LIVE = kn.OWNER_LABEL_PREFIX + kn.LIVE_OWNER
STAGING = kn.OWNER_LABEL_PREFIX + "staging"


def _item(name: str = "demo-model", image: str = "ghcr.io/hotosm/demo:v1", **props: Any) -> pystac.Item:
    item = pystac.Item(
        id=name,
        geometry=None,
        bbox=None,
        datetime=datetime.now(UTC),
        properties={"mlm:name": name, **props},
    )
    item.add_asset("mlm:inference", pystac.Asset(href=image))
    item.add_asset(
        "source-code",
        pystac.Asset(href="https://example.com/src", extra_fields={"mlm:entrypoint": "models.demo.pipeline:predict"}),
    )
    return item


class FakeKnative:
    """Implement merge patches: mappings merge, lists replace, and None deletes."""

    def __init__(self) -> None:
        self.services: dict[str, dict[str, Any]] = {}
        self.revisions: dict[str, dict[str, Any]] = {}
        self.patches: list[dict[str, Any]] = []
        self.deleted: list[str] = []

    def _not_found(self) -> ApiException:
        exc = ApiException(status=404)
        exc.status = 404
        return exc

    def _merge(self, target: dict[str, Any], patch: dict[str, Any]) -> None:
        for key, value in patch.items():
            if value is None:
                target.pop(key, None)
            elif isinstance(value, dict) and isinstance(target.get(key), dict):
                self._merge(target[key], value)
            else:
                target[key] = copy.deepcopy(value)

    def _roll(self, name: str) -> None:
        """Simulate revision creation and route resolution."""
        svc = self.services[name]
        template = svc["spec"].get("template")
        latest = svc.setdefault("status", {}).get("latestReadyRevisionName")
        if template is not None and (latest is None or self.revisions[latest]["template"] != template):
            latest = f"{name}-{len([r for r in self.revisions if r.startswith(name)]) + 1:05d}"
            self.revisions[latest] = {
                "service": name,
                "template": copy.deepcopy(template),
                "metadata": {"name": latest, "annotations": dict(template["metadata"].get("annotations") or {})},
            }
        svc["status"]["latestReadyRevisionName"] = latest
        resolved = []
        for route in svc["spec"].get("traffic") or [{"latestRevision": True, "percent": 100}]:
            rev = latest if route.get("latestRevision") else route["revisionName"]
            resolved.append({k: v for k, v in {**route, "revisionName": rev}.items() if k != "latestRevision"})
        svc["status"]["traffic"] = resolved
        svc["status"]["conditions"] = [{"type": "Ready", "status": "True"}]

    def get_namespaced_custom_object(self, *, name: str, **_: Any) -> dict[str, Any]:
        if name not in self.services:
            raise self._not_found()
        return copy.deepcopy(self.services[name])

    def create_namespaced_custom_object(self, *, body: dict[str, Any], **_: Any) -> None:
        self.services[body["metadata"]["name"]] = copy.deepcopy(body)
        self._roll(body["metadata"]["name"])

    def patch_namespaced_custom_object(self, *, name: str, body: dict[str, Any], **_: Any) -> None:
        self.patches.append(copy.deepcopy(body))
        self._merge(self.services[name], body)
        self._roll(name)

    def delete_namespaced_custom_object(self, *, name: str, **_: Any) -> None:
        if name not in self.services:
            raise self._not_found()
        del self.services[name]
        self.deleted.append(name)

    def list_namespaced_custom_object(self, *, plural: str, label_selector: str = "", **_: Any) -> dict[str, Any]:
        if plural == kn.KNATIVE_REVISIONS:
            service = label_selector.split("=", 1)[1]
            return {"items": [{"metadata": r["metadata"]} for r in self.revisions.values() if r["service"] == service]}
        return {"items": [copy.deepcopy(s) for s in self.services.values()]}

    def routes(self, name: str) -> list[tuple[str | None, str, int]]:
        return [(t.get("tag"), t["revisionName"], t["percent"]) for t in self.services[name]["status"]["traffic"]]

    def image(self, revision: str) -> str:
        return self.revisions[revision]["template"]["spec"]["containers"][0]["image"]


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> FakeKnative:
    api = FakeKnative()
    monkeypatch.setattr(kn, "_custom_objects_api", lambda: api)
    monkeypatch.setattr(kn, "_knative_serving_installed", lambda: True)
    monkeypatch.delenv("FAIR_KNATIVE_TAG", raising=False)
    monkeypatch.setenv("FAIR_KNATIVE_VERIFY_TIMEOUT", "1")
    return api


def _as(monkeypatch: pytest.MonkeyPatch, tag: str | None) -> None:
    if tag:
        monkeypatch.setenv("FAIR_KNATIVE_TAG", tag)
    else:
        monkeypatch.delenv("FAIR_KNATIVE_TAG", raising=False)


def test_public_url_includes_tag() -> None:
    assert kn.public_predict_url("Demo_Model", "ai.example.org") == "https://demo-model.predict.ai.example.org/predict"
    assert (
        kn.public_predict_url("Demo_Model", "ai.example.org", "staging")
        == "https://staging-demo-model.predict.ai.example.org/predict"
    )


def test_staging_candidate_does_not_move_live_then_release_promotes(fake: FakeKnative, monkeypatch) -> None:
    _as(monkeypatch, None)
    kn.ensure_knative_service(_item(image="img:v1"))
    [(tag, live_rev, pct)] = fake.routes("demo-model")
    assert (tag, pct) == (None, 100)

    _as(monkeypatch, "staging")
    kn.ensure_knative_service(_item(image="img:v2"))
    routes = fake.routes("demo-model")
    assert (None, live_rev, 100) in routes
    [(_, rc_rev, rc_pct)] = [r for r in routes if r[0] == "staging"]
    assert rc_pct == 0 and fake.image(rc_rev) == "img:v2"
    assert set(fake.services["demo-model"]["metadata"]["labels"]) >= {LIVE, STAGING}

    # An unchanged live item must not create a revision or change routes.
    _as(monkeypatch, None)
    revisions = len(fake.revisions)
    kn.ensure_knative_service(_item(image="img:v1"))
    assert len(fake.revisions) == revisions
    assert (None, live_rev, 100) in fake.routes("demo-model")
    assert ("staging", rc_rev, 0) in fake.routes("demo-model")

    # Live registration reuses the candidate revision.
    kn.ensure_knative_service(_item(image="img:v2"))
    assert len(fake.revisions) == revisions
    assert (None, rc_rev, 100) in fake.routes("demo-model")


def test_staging_only_model_serves_its_revision(fake: FakeKnative, monkeypatch) -> None:
    _as(monkeypatch, "staging")
    kn.ensure_knative_service(_item(image="img:v1"))
    routes = fake.routes("demo-model")
    assert {r[0] for r in routes} == {None, "staging"}
    assert all(fake.image(r[1]) == "img:v1" for r in routes)


def test_release_drops_only_own_route_and_deletes_when_unowned(fake: FakeKnative, monkeypatch) -> None:
    _as(monkeypatch, None)
    kn.ensure_knative_service(_item(image="img:v1"))
    _as(monkeypatch, "staging")
    kn.ensure_knative_service(_item(image="img:v2"))

    assert kn.release_knative_service("demo-model") is True
    assert [r[0] for r in fake.routes("demo-model")] == [None]
    assert STAGING not in fake.services["demo-model"]["metadata"]["labels"]
    assert kn.release_knative_service("demo-model") is False

    _as(monkeypatch, None)
    assert kn.release_knative_service("demo-model") is True
    assert fake.deleted == ["demo-model"]


def test_live_release_hands_traffic_to_remaining_tag(fake: FakeKnative, monkeypatch) -> None:
    _as(monkeypatch, None)
    kn.ensure_knative_service(_item(image="img:v1"))
    _as(monkeypatch, "staging")
    kn.ensure_knative_service(_item(image="img:v2"))
    _as(monkeypatch, None)
    assert kn.release_knative_service("demo-model") is True
    [(tag, rev, pct)] = fake.routes("demo-model")
    assert (tag, pct) == ("staging", 100) and fake.image(rev) == "img:v2"


def test_unlabelled_ksvc_belongs_to_live(fake: FakeKnative, monkeypatch) -> None:
    fake.services["legacy"] = {"metadata": {"name": "legacy", "labels": {}}, "spec": {}, "status": {}}
    _as(monkeypatch, "staging")
    assert kn.release_knative_service("legacy") is False
    _as(monkeypatch, None)
    assert kn.release_knative_service("legacy") is True
    assert fake.deleted == ["legacy"]


def test_reconcile_applies_prunes_and_reports_failures(fake: FakeKnative, monkeypatch) -> None:
    _as(monkeypatch, None)
    kn.ensure_knative_service(_item("gone-model"))
    broken = _item("broken-model")
    del broken.assets["mlm:inference"]

    result = kn.reconcile_knative_services([_item("kept-model"), broken], prune=True)
    assert result["applied"] == ["kept-model"]
    assert result["removed"] == ["gone-model"]
    assert result["failed"][0].startswith("broken-model")
    assert "gone-model" in fake.deleted

    again = kn.reconcile_knative_services([_item("kept-model")], prune=True)
    assert again == {"applied": ["kept-model"], "removed": [], "failed": []}


def test_reconcile_skips_prune_without_items(fake: FakeKnative, monkeypatch) -> None:
    _as(monkeypatch, None)
    kn.ensure_knative_service(_item())
    assert kn.reconcile_knative_services([], prune=True)["removed"] == []
    assert "demo-model" in fake.services


def test_client_reconcile_uses_active_base_models(monkeypatch) -> None:
    active, deprecated = _item("active-model"), _item("old-model", deprecated=True)

    class Backend:
        def list_items(self, collection_id: str, **_: Any) -> list[pystac.Item]:
            assert collection_id == BASE_MODELS_COLLECTION
            return [active, deprecated]

    captured: dict[str, Any] = {}

    def fake_reconcile(items, **kwargs):
        captured["ids"] = [i.id for i in items]
        captured.update(kwargs)
        return {"applied": captured["ids"], "removed": [], "failed": []}

    monkeypatch.setattr(kn, "reconcile_knative_services", fake_reconcile)
    client = FairClient(stac_api_url="https://stac.example.org")
    monkeypatch.setattr(client, "_get_backend", lambda: Backend())
    result = client.with_user("u").reconcile_knative(knative_template="t.yaml", prune=True)
    assert captured["ids"] == ["active-model"]
    assert captured["template_path"] == "t.yaml" and captured["prune"] is True
    assert result["applied"] == ["active-model"]
