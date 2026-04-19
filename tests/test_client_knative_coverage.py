from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import httpx
import pystac
import pytest
from kubernetes.client.exceptions import ApiException

import fair.client as client_module
import fair.infra.knative as knative_module
from fair.client import FairClient, FairClientError
from fair.stac.constants import BASE_MODELS_COLLECTION, DATASETS_COLLECTION, LOCAL_MODELS_COLLECTION


class DummyBackend:
    def __init__(self, items: dict[tuple[str, str], pystac.Item] | None = None) -> None:
        self.items = items or {}
        self.published: list[tuple[str, pystac.Item]] = []

    def get_item(self, collection: str, item_id: str) -> pystac.Item:
        key = (collection, item_id)
        if key not in self.items:
            raise KeyError(item_id)
        return self.items[key]

    def publish_item(self, collection: str, item: pystac.Item) -> pystac.Item:
        self.published.append((collection, item))
        return item

    def item_href(self, collection: str, item_id: str) -> str:
        return f"https://example.com/{collection}/{item_id}.json"


class DummyResponse:
    def __init__(self, payload: dict[str, Any] | None = None) -> None:
        self._payload = payload or {"status": "ok"}

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._payload


class DummyOutput:
    def __init__(self, value: Any) -> None:
        self._value = value

    def load(self) -> Any:
        return self._value


class DummyPipeline:
    def __init__(self, run: Any) -> None:
        self._run = run
        self.config_path: str | None = None

    def with_options(self, *, config_path: str, enable_cache: bool) -> Any:
        self.config_path = config_path

        def _runner() -> Any:
            return self._run

        return _runner


class DummyApiException(ApiException):
    def __init__(self, status: int) -> None:
        super().__init__(status=status)
        self.status = status


class DummyCustomObjectsApi:
    def __init__(self, items: list[dict[str, Any]] | None = None, missing_names: set[str] | None = None) -> None:
        self.items = items or []
        self.missing_names = missing_names or set()
        self.created: list[dict[str, Any]] = []
        self.patched: list[dict[str, Any]] = []
        self.deleted: list[str] = []

    def get_namespaced_custom_object(self, *, name: str, **_: Any) -> dict[str, Any]:
        if name in self.missing_names:
            raise DummyApiException(status=404)
        return {"metadata": {"name": name}}

    def create_namespaced_custom_object(self, *, body: dict[str, Any], **_: Any) -> dict[str, Any]:
        self.created.append(body)
        return body

    def patch_namespaced_custom_object(self, *, body: dict[str, Any], **_: Any) -> dict[str, Any]:
        self.patched.append(body)
        return body

    def list_namespaced_custom_object(self, **_: Any) -> dict[str, Any]:
        return {"items": self.items}

    def delete_namespaced_custom_object(self, *, name: str, **_: Any) -> None:
        if name == "missing-service":
            raise DummyApiException(status=404)
        if name == "broken-service":
            raise DummyApiException(status=500)
        self.deleted.append(name)


def _build_item(item_id: str, properties: dict[str, Any] | None = None) -> pystac.Item:
    return pystac.Item(
        id=item_id,
        geometry={
            "type": "Polygon",
            "coordinates": [[[-180, -90], [180, -90], [180, 90], [-180, 90], [-180, -90]]],
        },
        bbox=[-180, -90, 180, 90],
        datetime=datetime.now(UTC),
        properties=properties or {},
    )


def _build_base_model_item(item_id: str = "resnet18-classification") -> pystac.Item:
    item = _build_item(
        item_id,
        {
            "mlm:name": item_id,
            "mlm:hyperparameters": {"inference.confidence_threshold": 0.5},
            "version": "1",
        },
    )
    item.add_asset("checkpoint", pystac.Asset(href="https://example.com/checkpoint.pt"))
    item.add_asset("model", pystac.Asset(href="https://example.com/model.onnx"))
    item.add_asset("mlm:inference", pystac.Asset(href="https://example.com/model.onnx"))
    item.add_asset(
        "source-code",
        pystac.Asset(
            href="https://example.com/source.tar.gz",
            extra_fields={"mlm:entrypoint": "models.resnet.pipeline:train"},
        ),
    )
    return item


def _build_dataset_source_item(chips_href: str, labels_href: str) -> pystac.Item:
    item = _build_item(
        "dataset-source",
        {
            "title": "Dataset Source",
            "description": "demo",
            "label:type": "vector",
            "label:tasks": ["segmentation"],
            "label:classes": [{"name": "building"}],
            "keywords": ["demo"],
            "providers": [{"name": "provider"}],
            "license": "CC-BY-4.0",
        },
    )
    item.add_asset("chips", pystac.Asset(href=chips_href))
    item.add_asset("labels", pystac.Asset(href=labels_href))
    item.add_link(pystac.Link(rel="source", target="https://example.com/imagery.tif"))
    return item


def test_client_helper_methods_cover_core_branches(monkeypatch) -> None:
    client = FairClient(upload_artifacts=True)

    good_stack = SimpleNamespace(active_stack=SimpleNamespace(artifact_store=SimpleNamespace(path="s3://bucket/path/")))
    monkeypatch.setattr(client_module, "Client", lambda: good_stack)
    assert client._artifact_store_prefix() == "s3://bucket/path"

    local_stack = SimpleNamespace(active_stack=SimpleNamespace(artifact_store=SimpleNamespace(path="/tmp/store")))
    monkeypatch.setattr(client_module, "Client", lambda: local_stack)
    assert client._artifact_store_prefix() is None

    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(client, "_artifact_store_prefix", lambda: "s3://bucket")
    monkeypatch.setattr(
        client_module,
        "upload_item_assets",
        lambda item, prefix, collection: calls.append((prefix, collection)),
    )
    client._upload_assets_if_remote(_build_item("x"), BASE_MODELS_COLLECTION)
    assert calls == [("s3://bucket", BASE_MODELS_COLLECTION)]

    assert client._dataset_providers_from_properties({"providers": [{"name": "ok"}]}, "item-1") == [{"name": "ok"}]
    with pytest.raises(FairClientError):
        client._dataset_providers_from_properties({}, "item-1")
    with pytest.raises(FairClientError):
        client._dataset_providers_from_properties({"providers": ["bad"]}, "item-1")

    assert client._predict_base_url("https://predict.example.com") == "https://predict.example.com"
    monkeypatch.delenv("FAIR_PREDICT_BASE_URL", raising=False)
    monkeypatch.delenv("FAIR_LABEL_DOMAIN", raising=False)
    with pytest.raises(FairClientError):
        client._predict_base_url()
    monkeypatch.setenv("FAIR_PREDICT_BASE_URL", "https://predict.env.example.com")
    assert client._predict_base_url() == "https://predict.env.example.com"
    monkeypatch.delenv("FAIR_PREDICT_BASE_URL", raising=False)
    monkeypatch.setenv("FAIR_LABEL_DOMAIN", "fair.example.com")
    assert client._predict_base_url() == "https://predict.fair.example.com"

    monkeypatch.delenv("FAIR_PREDICT_VERIFY_SSL", raising=False)
    monkeypatch.delenv("ZENML_STORE_VERIFY_SSL", raising=False)
    assert client._predict_verify_ssl() is True
    monkeypatch.setenv("FAIR_PREDICT_VERIFY_SSL", "false")
    assert client._predict_verify_ssl() is False


def test_pipeline_module_and_base_model_resolution(monkeypatch) -> None:
    client = FairClient(stac_api_url="https://stac.example.com", dsn="postgresql://example")
    base_item = _build_base_model_item()
    assert client._pipeline_module_from_item(base_item) == "models.resnet.pipeline"

    missing_source = _build_item("missing-source")
    with pytest.raises(FairClientError):
        client._pipeline_module_from_item(missing_source)

    empty_entrypoint = _build_item("bad-entrypoint")
    empty_entrypoint.add_asset(
        "source-code",
        pystac.Asset(
            href="https://example.com/src",
            extra_fields={"mlm:entrypoint": ""},
        ),
    )
    with pytest.raises(FairClientError):
        client._pipeline_module_from_item(empty_entrypoint)

    local_model = _build_item("local-model", {"fair:base_model_id": base_item.id})
    backend = DummyBackend(
        {
            (BASE_MODELS_COLLECTION, base_item.id): base_item,
            (LOCAL_MODELS_COLLECTION, local_model.id): local_model,
        }
    )
    monkeypatch.setattr(client, "_get_backend", lambda: backend)

    assert client._base_model_name_for_live_service(base_item, BASE_MODELS_COLLECTION) == "resnet18-classification"
    assert client._base_model_name_for_live_service(local_model, LOCAL_MODELS_COLLECTION) == "resnet18-classification"

    broken_local = _build_item("broken-local")
    with pytest.raises(FairClientError):
        client._base_model_name_for_live_service(broken_local, LOCAL_MODELS_COLLECTION)


def test_mirror_asset_to_artifact_store_updates_asset_href(monkeypatch) -> None:
    client = FairClient(upload_artifacts=True)
    item = _build_base_model_item()

    monkeypatch.setattr(client, "_artifact_store_prefix", lambda: "s3://bucket")

    class DummyUPath:
        storage: ClassVar[dict[str, bytes]] = {
            "https://example.com/checkpoint.pt": b"weights",
        }

        def __init__(self, path: str) -> None:
            self.path = path

        @property
        def name(self) -> str:
            return Path(self.path).name

        def read_bytes(self) -> bytes:
            return self.storage[self.path]

        def write_bytes(self, data: bytes) -> None:
            self.storage[self.path] = data

    import upath

    monkeypatch.setattr(upath, "UPath", DummyUPath)
    monkeypatch.setattr(
        client_module,
        "s3_uri_to_http_url",
        lambda path: f"https://cdn.example.com/{path.split('://', 1)[1]}",
    )

    client._mirror_asset_to_artifact_store(item, "checkpoint", BASE_MODELS_COLLECTION)
    assert (
        item.assets["checkpoint"].href
        == "https://cdn.example.com/bucket/base-models/resnet18-classification/checkpoint/checkpoint.pt"
    )

    monkeypatch.setattr(client, "_artifact_store_prefix", lambda: None)
    with pytest.raises(FairClientError):
        client._mirror_asset_to_artifact_store(item, "model", BASE_MODELS_COLLECTION)


def test_setup_register_base_model_and_register_dataset_paths(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    init_calls: list[str] = []
    monkeypatch.setattr(client_module.subprocess, "run", lambda cmd, check, capture_output: init_calls.append(cmd[-1]))
    monkeypatch.setattr(client_module, "initialize_catalog", lambda path: init_calls.append(path))

    client = FairClient(catalog_path="catalog.json")
    client.setup()
    assert "init" in init_calls
    assert "catalog.json" in init_calls
    assert (tmp_path / "artifacts").exists()

    backend = DummyBackend()
    item = _build_base_model_item()
    previous = _build_base_model_item("older-model")
    previous.properties["version"] = "1"
    archived: list[str] = []
    ensured: list[str] = []

    monkeypatch.setattr(client, "_get_backend", lambda: backend)
    monkeypatch.setattr(client_module.pystac.Item, "from_file", lambda _: item)
    monkeypatch.setattr(client_module, "validate_item", lambda _: [])
    monkeypatch.setattr(client_module, "validate_model_asset_urls", lambda *args, **kwargs: [])
    monkeypatch.setattr(client_module, "find_previous_active_item", lambda *args, **kwargs: previous)
    monkeypatch.setattr(client_module, "archive_previous_version", lambda *args, **kwargs: archived.append("done"))
    monkeypatch.setattr(client, "_upload_assets_if_remote", lambda *args, **kwargs: None)
    monkeypatch.setattr(knative_module, "ensure_knative_service", lambda published: ensured.append(published.id))

    assert client.register_base_model("base.json") == "resnet18-classification"
    assert item.properties["version"] == "2"
    assert archived == ["done"]
    assert ensured == ["resnet18-classification"]

    monkeypatch.setattr(
        knative_module,
        "ensure_knative_service",
        lambda published: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    assert client.register_base_model("base.json") == "resnet18-classification"

    source_labels = tmp_path / "labels.geojson"
    source_labels.write_text("{}")
    source_chips = tmp_path / "chips"
    source_chips.mkdir()
    source_item = _build_dataset_source_item(str(source_chips), str(source_labels))
    published_dataset = _build_item("published-dataset", {"version": "1"})
    previous_dataset = _build_item("previous-dataset", {"version": "3"})

    monkeypatch.setattr(client_module.pystac.Item, "from_file", lambda _: source_item)
    monkeypatch.setattr(client_module, "count_chips", lambda _: 4)
    monkeypatch.setattr(client_module, "create_dataset_archive", lambda **kwargs: None)
    monkeypatch.setattr(client_module, "find_previous_active_item", lambda *args, **kwargs: previous_dataset)
    monkeypatch.setattr(client_module, "build_dataset_item", lambda **kwargs: published_dataset)
    monkeypatch.setattr(client_module, "validate_item", lambda _: [])

    assert client._register_dataset_from_item("dataset.json") == "published-dataset"
    assert client.register_dataset("dataset.json") == "published-dataset"

    monkeypatch.setattr(
        client_module,
        "build_dataset_item",
        lambda **kwargs: _build_item("params-dataset", {"version": "1"}),
    )
    params = client_module.DatasetItemParams(
        label_type="vector",
        label_tasks=["segmentation"],
        label_classes=[{"name": "building"}],
        keywords=["demo"],
        chips_href=str(source_chips),
        labels_href=str(source_labels),
        title="Demo Dataset",
        description="demo",
        user_id="tester",
        providers=[{"name": "provider"}],
    )
    assert client._register_dataset_from_params(params) == "params-dataset"
    assert client.register_dataset(params) == "params-dataset"


def test_finetune_promote_and_predict_workflows(monkeypatch, tmp_path) -> None:
    base_item = _build_base_model_item()
    dataset_item = _build_item("dataset-1", {"version": "1"})
    local_model = _build_item("local-model", {"fair:base_model_id": base_item.id})
    local_model.add_asset("model", pystac.Asset(href="https://example.com/local-model.onnx"))

    backend = DummyBackend(
        {
            (BASE_MODELS_COLLECTION, base_item.id): base_item,
            (DATASETS_COLLECTION, dataset_item.id): dataset_item,
            (LOCAL_MODELS_COLLECTION, local_model.id): local_model,
        }
    )

    client = FairClient(
        stac_api_url="https://stac.example.com",
        dsn="postgresql://example",
        config_dir=str(tmp_path / "config"),
        upload_artifacts=True,
    )
    monkeypatch.setattr(client, "_get_backend", lambda: backend)
    monkeypatch.setattr(client, "_artifact_store_prefix", lambda: "s3://bucket")
    monkeypatch.setattr(client, "_pipeline_module_from_item", lambda _: "models.demo.pipeline")
    monkeypatch.setattr(client_module, "validate_compatibility", lambda *_: [])
    monkeypatch.setattr(
        client_module,
        "generate_training_config",
        lambda *args, **kwargs: {"steps": {"train_model": {}}},
    )
    monkeypatch.setattr(
        client_module,
        "generate_inference_config",
        lambda *args, **kwargs: {"steps": {"predict": {}}},
    )

    tracker = SimpleNamespace(name="mlflow")
    zenml_client = SimpleNamespace(
        active_stack_model=SimpleNamespace(components={client_module.StackComponentType.EXPERIMENT_TRACKER: [tracker]}),
        list_model_versions=lambda **kwargs: [SimpleNamespace(number=7, id="mv-1")],
        list_model_version_pipeline_run_links=lambda **kwargs: SimpleNamespace(
            items=[
                SimpleNamespace(
                    pipeline_run=SimpleNamespace(
                        steps={
                            "train_model": SimpleNamespace(
                                config=SimpleNamespace(
                                    parameters={
                                        "base_model_id": base_item.id,
                                        "dataset_id": dataset_item.id,
                                    }
                                )
                            )
                        },
                        config=SimpleNamespace(parameters={}),
                    )
                )
            ]
        ),
    )
    monkeypatch.setattr(client_module, "Client", lambda: zenml_client)

    training_run = SimpleNamespace(id="train-run", status="completed")
    prediction_run = SimpleNamespace(
        id="predict-run",
        status="completed",
        steps={"predict": SimpleNamespace(outputs={"predictions": [DummyOutput({"ok": True})]})},
    )
    training_pipeline = DummyPipeline(training_run)
    inference_pipeline = DummyPipeline(prediction_run)
    fake_module = SimpleNamespace(training_pipeline=training_pipeline, inference_pipeline=inference_pipeline)
    monkeypatch.setattr(client_module.importlib, "import_module", lambda _: fake_module)
    monkeypatch.setattr(client_module, "promote_model_version", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        client_module,
        "publish_promoted_model",
        lambda **kwargs: _build_item("promoted-model", {"version": "7"}),
    )
    upload_calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        client_module,
        "upload_local_directory",
        lambda path, remote: upload_calls.append((str(path), remote)),
    )

    assert (
        client.finetune(base_model_id=base_item.id, dataset_id=dataset_item.id, model_name="demo-model") == "demo-model"
    )
    assert client.promote("demo-model") == "promoted-model"
    assert client.predict(local_model.id, str(tmp_path)) == {"ok": True}
    assert upload_calls[-1][1] == f"s3://bucket/predict/{local_model.id}/input"

    broken_backend = DummyBackend({(BASE_MODELS_COLLECTION, base_item.id): base_item})
    monkeypatch.setattr(client, "_get_backend", lambda: broken_backend)
    with pytest.raises(FairClientError):
        client.finetune(base_model_id=base_item.id, dataset_id="missing", model_name="demo-model")


def test_predict_live_error_paths_and_success(monkeypatch, tmp_path) -> None:
    base_item = _build_base_model_item()
    local_model = _build_item("local-model", {"fair:base_model_id": base_item.id})
    local_model.add_asset("model", pystac.Asset(href="https://example.com/local-model.onnx"))

    backend = DummyBackend(
        {
            (BASE_MODELS_COLLECTION, base_item.id): base_item,
            (LOCAL_MODELS_COLLECTION, local_model.id): local_model,
        }
    )
    client = FairClient(stac_api_url="https://stac.example.com", dsn="postgresql://example", upload_artifacts=True)
    monkeypatch.setattr(client, "_get_backend", lambda: backend)
    monkeypatch.setattr(client, "_artifact_store_prefix", lambda: "s3://bucket")

    upload_calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        client_module,
        "upload_local_directory",
        lambda path, remote: upload_calls.append((str(path), remote)),
    )

    captured: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: Any) -> DummyResponse:
        captured["url"] = url
        captured["kwargs"] = kwargs
        return DummyResponse({"ok": True})

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setenv("FAIR_PREDICT_VERIFY_SSL", "no")

    result = client.predict_live(
        local_model.id,
        str(tmp_path),
        collection=LOCAL_MODELS_COLLECTION,
        predict_base_url="https://predict.example.com",
    )
    assert result == {"ok": True}
    assert upload_calls[-1][1] == f"s3://bucket/predict/{local_model.id}/input"
    assert captured["url"] == "https://predict.example.com/resnet18-classification/predict"
    assert captured["kwargs"]["verify"] is False

    missing_model_backend = DummyBackend()
    monkeypatch.setattr(client, "_get_backend", lambda: missing_model_backend)
    with pytest.raises(FairClientError):
        client.predict_live("missing", str(tmp_path), predict_base_url="https://predict.example.com")

    no_asset_item = _build_item("no-asset")
    no_asset_backend = DummyBackend({(LOCAL_MODELS_COLLECTION, no_asset_item.id): no_asset_item})
    monkeypatch.setattr(client, "_get_backend", lambda: no_asset_backend)
    with pytest.raises(FairClientError):
        client.predict_live(no_asset_item.id, str(tmp_path), predict_base_url="https://predict.example.com")


def test_knative_helpers_and_gateway_management(monkeypatch) -> None:
    assert knative_module.knative_service_name("Model_Name") == "model-name"
    assert knative_module.knative_service_host("model-name") == "model-name.predict.svc.cluster.local"
    assert knative_module._module_from_entrypoint("pkg.module:run") == "pkg.module"
    with pytest.raises(ValueError):
        knative_module._module_from_entrypoint("pkg.module")

    empty_config = knative_module.build_predict_gateway_config([])
    assert "No models are currently registered." in empty_config

    manifest = knative_module.build_knative_manifest(_build_base_model_item())
    assert manifest["metadata"]["name"] == "resnet18-classification"
    assert manifest["spec"]["template"]["spec"]["containers"][0]["image"] == "https://example.com/model.onnx"

    missing_inference = _build_item("missing-inference")
    missing_inference.add_asset(
        "source-code",
        pystac.Asset(href="https://example.com/src", extra_fields={"mlm:entrypoint": "pkg.module:run"}),
    )
    with pytest.raises(KeyError):
        knative_module.build_knative_manifest(missing_inference)

    missing_source = _build_item("missing-source")
    missing_source.add_asset("mlm:inference", pystac.Asset(href="https://example.com/model.onnx"))
    with pytest.raises(KeyError):
        knative_module.build_knative_manifest(missing_source)

    created: list[str] = []
    patched: list[str] = []
    knative_module._upsert_resource(
        read=lambda: (_ for _ in ()).throw(DummyApiException(status=404)),
        create=lambda: created.append("created"),
        patch=lambda: patched.append("patched"),
    )
    knative_module._upsert_resource(
        read=lambda: {"ok": True},
        create=lambda: created.append("should-not-create"),
        patch=lambda: patched.append("patched"),
    )
    assert created == ["created"]
    assert patched == ["patched"]

    api = DummyCustomObjectsApi(
        items=[{"metadata": {"name": "resnet18-classification"}}],
        missing_names={"resnet18-classification"},
    )
    knative_module._upsert_knative_service(api, manifest, knative_module.DEFAULT_NAMESPACE)
    assert api.created[0]["metadata"]["name"] == "resnet18-classification"
    api.missing_names.clear()
    knative_module._upsert_knative_service(api, manifest, knative_module.DEFAULT_NAMESPACE)
    assert api.patched[-1]["metadata"]["name"] == "resnet18-classification"
    assert knative_module._list_knative_service_names(
        api,
        knative_module.DEFAULT_NAMESPACE,
    ) == ["resnet18-classification"]

    class DummyNetworkingApi:
        def __init__(self) -> None:
            self.deleted: list[str] = []

        def delete_namespaced_ingress(self, name: str, namespace: str) -> None:
            if name == "missing-public":
                raise DummyApiException(status=404)
            self.deleted.append(f"{namespace}:{name}")

    class DummyCoreApi:
        def __init__(self) -> None:
            self.created: list[str] = []
            self.patched: list[str] = []

        def read_namespaced_config_map(self, *args: Any) -> Any:
            raise DummyApiException(status=404)

        def create_namespaced_config_map(self, namespace: str, body: Any) -> None:
            self.created.append(f"config:{namespace}:{body.metadata.name}")

        def patch_namespaced_config_map(self, name: str, namespace: str, body: Any) -> None:
            self.patched.append(f"config:{namespace}:{name}")

        def read_namespaced_service(self, *args: Any) -> Any:
            raise DummyApiException(status=404)

        def create_namespaced_service(self, namespace: str, body: Any) -> None:
            self.created.append(f"service:{namespace}:{body.metadata.name}")

        def patch_namespaced_service(self, name: str, namespace: str, body: Any) -> None:
            self.patched.append(f"service:{namespace}:{name}")

    class DummyAppsApi:
        def __init__(self) -> None:
            self.created: list[str] = []

        def read_namespaced_deployment(self, *args: Any) -> Any:
            raise DummyApiException(status=404)

        def create_namespaced_deployment(self, namespace: str, body: Any) -> None:
            self.created.append(f"deployment:{namespace}:{body.metadata.name}")

        def patch_namespaced_deployment(self, name: str, namespace: str, body: Any) -> None:
            self.created.append(f"deployment-patch:{namespace}:{name}")

    import kubernetes.client as kube_client

    networking_api = DummyNetworkingApi()
    core_api = DummyCoreApi()
    apps_api = DummyAppsApi()
    monkeypatch.setattr(kube_client, "NetworkingV1Api", lambda: networking_api)
    monkeypatch.setattr(kube_client, "CoreV1Api", lambda: core_api)
    monkeypatch.setattr(kube_client, "AppsV1Api", lambda: apps_api)

    gateway_api = DummyCustomObjectsApi(
        items=[
            {"metadata": {"name": "resnet18-classification"}},
            {"metadata": {"name": "missing"}},
        ]
    )
    knative_module._ensure_predict_gateway(gateway_api, knative_module.DEFAULT_NAMESPACE)
    assert any(entry.startswith("config:") for entry in core_api.created)
    assert any(entry.startswith("deployment:") for entry in apps_api.created)
    assert any(entry.startswith("service:") for entry in core_api.created)

    monkeypatch.setattr(knative_module, "_custom_objects_api", lambda: gateway_api)
    upserted: list[str] = []
    gateway_calls: list[str] = []
    monkeypatch.setattr(knative_module, "_upsert_knative_service", lambda *args: upserted.append("service"))
    monkeypatch.setattr(knative_module, "_ensure_predict_gateway", lambda *args: gateway_calls.append("gateway"))

    monkeypatch.delenv("FAIR_LABEL_DOMAIN", raising=False)
    knative_module.ensure_knative_service(_build_base_model_item())
    monkeypatch.setenv("FAIR_LABEL_DOMAIN", "fair.example.com")
    knative_module.ensure_knative_service(_build_base_model_item())
    assert upserted == ["service", "service"]
    assert gateway_calls == ["gateway"]

    delete_api = DummyCustomObjectsApi()
    monkeypatch.setattr(knative_module, "_custom_objects_api", lambda: delete_api)
    knative_module.delete_knative_service("missing-service")
    knative_module.delete_knative_service("good-service")
    assert delete_api.deleted == ["good-service"]
    with pytest.raises(DummyApiException):
        knative_module.delete_knative_service("broken-service")
