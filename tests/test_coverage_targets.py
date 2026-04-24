from __future__ import annotations

import ast
import builtins
import importlib
import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pystac
import pytest

import fair._patch_zenml as patch_module
import fair.client as client_module
from fair.client import FairClient, FairClientError
from fair.stac.builders import (
    _infer_source_code_media_type,
    _raster_bands_from_model_input,
    _validate_providers,
    geometry_and_bbox_from_geojson,
)
from fair.utils.model_validator import _return_annotation_name, validate_model


def _make_item(item_id: str, properties: dict[str, Any] | None = None) -> pystac.Item:
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


def _write_model_scaffold(model_dir: Path, *, model_href: str = "https://example.com/model.onnx") -> None:
    (model_dir / "tests").mkdir(parents=True, exist_ok=True)
    (model_dir / "README.md").write_text("# Demo\n", encoding="utf-8")
    (model_dir / "stac-item.json").write_text(
        json.dumps(
            {
                "assets": {
                    "checkpoint": {"href": "https://example.com/checkpoint.pt"},
                    "model": {"href": model_href},
                }
            }
        ),
        encoding="utf-8",
    )
    (model_dir / "pipeline.py").write_text(
        """
from zenml import pipeline, step

@pipeline
def training_pipeline():
    pass

@pipeline
def inference_pipeline():
    pass

@step
def split_dataset() -> bytes:
    return b''
""".strip(),
        encoding="utf-8",
    )
    (model_dir / "tests" / "test_steps.py").write_text(
        """
def test_split_dataset():
    assert True

def test_train_model():
    assert True

def test_evaluate_model():
    assert True

def test_export_onnx():
    assert True
""".strip(),
        encoding="utf-8",
    )


def test_client_helper_error_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    client = FairClient(stac_api_url="https://stac.example.com", config_dir=str(tmp_path))

    monkeypatch.setattr(
        client_module,
        "Client",
        lambda: SimpleNamespace(active_stack=SimpleNamespace(artifact_store=SimpleNamespace(path="/tmp/artifacts"))),
    )

    assert client._artifact_store_prefix() is None

    with pytest.raises(FairClientError, match="non-empty 'providers' list"):
        client._dataset_providers_from_properties({}, "dataset-1")

    with pytest.raises(FairClientError, match="invalid provider entry"):
        client._dataset_providers_from_properties({"providers": ["bad"]}, "dataset-1")

    missing_source = _make_item("base-model")
    with pytest.raises(FairClientError, match="has no source-code asset"):
        client._pipeline_module_from_item(missing_source)

    empty_entrypoint = _make_item("base-model")
    empty_entrypoint.add_asset("source-code", pystac.Asset(href="https://example.com/pipeline.py"))
    with pytest.raises(FairClientError, match="Cannot determine pipeline module"):
        client._pipeline_module_from_item(empty_entrypoint)

    model_item = _make_item("base-model")
    model_item.add_asset("model", pystac.Asset(href="https://example.com/model.onnx"))
    with pytest.raises(FairClientError, match="Artifact store prefix is not configured"):
        client._mirror_asset_to_artifact_store(model_item, "model", "base-models")


def test_predict_raises_when_inference_pipeline_returns_no_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_item = _make_item("local-model", {"fair:base_model_id": "base-1"})
    model_item.add_asset("model", pystac.Asset(href="https://example.com/model.onnx"))
    model_item.add_asset(
        "source-code",
        pystac.Asset(
            href="https://example.com/pipeline.py",
            extra_fields={"mlm:entrypoint": "demo.pipeline:inference_pipeline"},
        ),
    )

    backend = SimpleNamespace(get_item=lambda collection, item_id: model_item)
    client = FairClient(config_dir=str(tmp_path), upload_artifacts=True)

    uploads: list[tuple[str, str]] = []

    monkeypatch.setattr(client, "_get_backend", lambda: backend)
    monkeypatch.setattr(client, "_artifact_store_prefix", lambda: "s3://bucket")
    monkeypatch.setattr(
        client_module,
        "upload_local_directory",
        lambda path, remote_path: uploads.append((str(path), remote_path)),
    )

    class _InferencePipeline:
        def with_options(self, *, config_path: str, enable_cache: bool) -> Any:
            def _runner() -> None:
                return None

            return _runner

    module = SimpleNamespace(inference_pipeline=_InferencePipeline())
    monkeypatch.setattr(client_module.importlib, "import_module", lambda _: module)

    with pytest.raises(RuntimeError, match="Inference pipeline returned no run"):
        client.predict("local-model", str(tmp_path))

    assert uploads == [(str(tmp_path), "s3://bucket/predict/local-model/input")]


def test_artifact_store_setup_and_registration_error_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client = FairClient(config_dir=str(tmp_path), catalog_path="catalog.json")

    monkeypatch.setattr(
        client_module,
        "Client",
        lambda: SimpleNamespace(active_stack=SimpleNamespace(artifact_store=SimpleNamespace(path="s3://bucket/root/"))),
    )
    assert client._artifact_store_prefix() == "s3://bucket/root"

    upload_calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        client_module,
        "upload_item_assets",
        lambda item, prefix, collection_id: upload_calls.append((prefix, collection_id)),
    )
    client._upload_assets_if_remote(_make_item("skipped"), "base-models")
    assert upload_calls == []

    remote_client = FairClient(config_dir=str(tmp_path), upload_artifacts=True)
    monkeypatch.setattr(remote_client, "_artifact_store_prefix", lambda: "s3://bucket")
    remote_client._upload_assets_if_remote(_make_item("uploaded"), "base-models")
    assert upload_calls == [("s3://bucket", "base-models")]

    monkeypatch.setattr(client_module, "Client", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    assert client._artifact_store_prefix() is None

    catalog_calls: list[str] = []
    subprocess_calls: list[list[str]] = []

    monkeypatch.setattr(client_module, "initialize_catalog", lambda path: catalog_calls.append(path))
    monkeypatch.setattr(
        client_module.subprocess,
        "run",
        lambda args, check, capture_output: subprocess_calls.append(list(args)),
    )

    client.setup()

    assert subprocess_calls
    assert catalog_calls == ["catalog.json"]

    item_path = tmp_path / "base-model.json"
    _make_item("demo-base").save_object(dest_href=str(item_path))

    monkeypatch.setattr(client, "_get_backend", lambda: SimpleNamespace())
    monkeypatch.setattr(client_module, "validate_item", lambda item: ["broken-schema"])

    with pytest.raises(FairClientError, match="Schema validation failed"):
        client.register_base_model(str(item_path))

    monkeypatch.setattr(client_module, "validate_item", lambda item: [])
    monkeypatch.setattr(client_module, "validate_model_asset_urls", lambda *args, **kwargs: ["bad-asset"])

    with pytest.raises(FairClientError, match="Asset URL validation failed"):
        client.register_base_model(str(item_path))


def test_backend_and_dataclass_registration_branches(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    catalog_path = tmp_path / "stac_catalog" / "catalog.json"
    client_module.initialize_catalog(str(catalog_path))

    local_client = FairClient(config_dir=str(tmp_path), catalog_path=str(catalog_path))
    assert local_client._get_backend().__class__.__name__ == "StacCatalogManager"

    real_import = builtins.__import__

    class _DummyPgStacBackend:
        def __init__(self, dsn: str, stac_api_url: str) -> None:
            self.dsn = dsn
            self.stac_api_url = stac_api_url

    def _fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "fair.stac.pgstac_backend":
            return SimpleNamespace(PgStacBackend=_DummyPgStacBackend)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    with pytest.raises(FairClientError, match="dsn is required"):
        FairClient(stac_api_url="https://stac.example.com", config_dir=str(tmp_path))._get_backend()

    remote_backend = FairClient(
        stac_api_url="https://stac.example.com",
        dsn="postgresql://user:pass@localhost/db",
        config_dir=str(tmp_path),
    )._get_backend()
    assert isinstance(remote_backend, _DummyPgStacBackend)

    base_geometry = _make_item("base").geometry
    assert base_geometry is not None

    params = client_module.BaseModelItemParams(
        item_id="demo-base",
        geometry=base_geometry,
        mlm_name="demo-base",
        mlm_architecture="resnet18",
        mlm_tasks=["classification"],
        mlm_framework="PyTorch",
        mlm_framework_version="2.0",
        mlm_input=[{"bands": ["red", "green"]}],
        mlm_output=[{"name": "class"}],
        mlm_hyperparameters={},
        keywords=["buildings"],
        checkpoint_href="https://example.com/checkpoint.pt",
        checkpoint_artifact_type="torch.save",
        mlm_pretrained=True,
        mlm_pretrained_source=None,
        source_code_href="https://example.com/pipeline.py",
        source_code_entrypoint="demo.pipeline:training_pipeline",
        training_runtime_href="docker.io/example/train:latest",
        inference_runtime_href="docker.io/example/predict:latest",
        title="Demo Base",
        description="Demo model",
        fair_metrics_spec=[],
        providers=[{"name": "HOT", "roles": ["producer"]}],
        onnx_href="https://example.com/model.onnx",
    )

    registering_client = FairClient(config_dir=str(tmp_path), upload_artifacts=True)
    monkeypatch.setattr(registering_client, "_get_backend", lambda: SimpleNamespace())
    monkeypatch.setattr(client_module, "validate_item", lambda item: ["broken-schema"])

    with pytest.raises(FairClientError, match="Schema validation failed"):
        registering_client.register_base_model(params)

    validation_calls = {"count": 0}

    def _validate_urls(*args: Any, **kwargs: Any) -> list[str]:
        validation_calls["count"] += 1
        return [] if validation_calls["count"] == 1 else ["mirrored-bad"]

    monkeypatch.setattr(client_module, "validate_item", lambda item: [])
    monkeypatch.setattr(client_module, "validate_model_asset_urls", _validate_urls)
    monkeypatch.setattr(client_module, "find_previous_active_item", lambda *args, **kwargs: None)
    monkeypatch.setattr(registering_client, "_mirror_asset_to_artifact_store", lambda *args, **kwargs: None)
    monkeypatch.setattr(registering_client, "_upload_assets_if_remote", lambda *args, **kwargs: None)

    with pytest.raises(FairClientError, match="Mirrored asset URL validation failed"):
        registering_client.register_base_model(params)

    plain_client = FairClient(config_dir=str(tmp_path))
    empty_item = _make_item("no-model-asset")
    plain_client._mirror_asset_to_artifact_store(empty_item, "model", "base-models")


def test_dataset_and_prediction_error_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    client = FairClient(config_dir=str(tmp_path))
    monkeypatch.setattr(client, "_get_backend", lambda: SimpleNamespace())

    missing_chips = _make_item("dataset-no-chips")
    missing_chips_path = tmp_path / "dataset-no-chips.json"
    missing_chips.save_object(dest_href=str(missing_chips_path))

    with pytest.raises(FairClientError, match="has no 'chips' asset"):
        client.register_dataset(str(missing_chips_path))

    missing_labels = _make_item("dataset-no-labels")
    missing_labels.add_asset("chips", pystac.Asset(href="https://example.com/chips"))
    missing_labels_path = tmp_path / "dataset-no-labels.json"
    missing_labels.save_object(dest_href=str(missing_labels_path))

    with pytest.raises(FairClientError, match="has no 'labels' asset"):
        client.register_dataset(str(missing_labels_path))

    schema_error_item = _make_item(
        "dataset-invalid-schema",
        {
            "title": "Dataset",
            "description": "Dataset",
            "providers": [{"name": "HOT", "roles": ["producer"]}],
        },
    )
    schema_error_item.add_asset("chips", pystac.Asset(href="https://example.com/chips"))
    schema_error_item.add_asset("labels", pystac.Asset(href="https://example.com/labels.geojson"))
    schema_error_path = tmp_path / "dataset-invalid-schema.json"
    schema_error_item.save_object(dest_href=str(schema_error_path))

    monkeypatch.setattr(client_module, "count_chips", lambda href: 1)
    monkeypatch.setattr(client_module, "find_previous_active_item", lambda *args, **kwargs: None)
    monkeypatch.setattr(client_module, "create_dataset_archive", lambda **kwargs: None)
    monkeypatch.setattr(client_module, "build_dataset_item", lambda **kwargs: _make_item("built-dataset"))
    monkeypatch.setattr(client_module, "validate_item", lambda item: ["broken-dataset"])

    with pytest.raises(FairClientError, match="Schema validation failed"):
        client.register_dataset(str(schema_error_path))

    missing_backend = SimpleNamespace(get_item=lambda collection, item_id: (_ for _ in ()).throw(KeyError(item_id)))
    monkeypatch.setattr(client, "_get_backend", lambda: missing_backend)

    with pytest.raises(FairClientError, match=r"Run promote\(\) first"):
        client.predict("missing-model", str(tmp_path))

    with pytest.raises(FairClientError, match="not found in 'local-models'"):
        client.predict_live(
            "missing-model",
            image_uri="https://tiles.example.com/{z}/{x}/{y}",
            bbox=[0.0, 0.0, 1.0, 1.0],
            zoom=18,
        )


def test_dataset_param_and_finetune_validation_error_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client = FairClient(config_dir=str(tmp_path))

    dataset_params = client_module.DatasetItemParams(
        label_type="vector",
        label_tasks=["detection"],
        label_classes=[{"name": "building"}],
        keywords=["buildings"],
        chips_href=str(tmp_path / "chips"),
        labels_href=str(tmp_path / "labels.geojson"),
        title="Demo dataset",
        description="Demo dataset",
        user_id="tester",
        providers=[{"name": "HOT", "roles": ["producer"]}],
        geometry=_make_item("dataset").geometry,
        bbox=_make_item("dataset").bbox,
    )

    monkeypatch.setattr(client, "_get_backend", lambda: SimpleNamespace())
    monkeypatch.setattr(client_module, "count_chips", lambda href: 0)
    monkeypatch.setattr(client_module, "find_previous_active_item", lambda *args, **kwargs: None)
    monkeypatch.setattr(client_module, "create_dataset_archive", lambda **kwargs: None)
    monkeypatch.setattr(client_module, "validate_item", lambda item: ["broken-dataset"])

    with pytest.raises(FairClientError, match="Schema validation failed"):
        client.register_dataset(dataset_params)

    base_item = _make_item("base-model", {"mlm:name": "demo-model"})
    base_item.add_asset(
        "source-code",
        pystac.Asset(
            href="https://example.com/pipeline.py",
            extra_fields={"mlm:entrypoint": "demo.pipeline:training_pipeline"},
        ),
    )
    dataset_item = _make_item("dataset")
    incompatible_backend = SimpleNamespace(
        get_item=lambda collection, item_id: base_item if collection == "base-models" else dataset_item
    )
    monkeypatch.setattr(client, "_get_backend", lambda: incompatible_backend)
    monkeypatch.setattr(client_module, "validate_compatibility", lambda base, ds: ["task mismatch"])

    with pytest.raises(FairClientError, match="Incompatible"):
        client.finetune(base_model_id="base-model", dataset_id="dataset", model_name="demo")


def test_finetune_and_promote_error_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    client = FairClient(config_dir=str(tmp_path))

    base_item = _make_item("base-model", {"mlm:name": "demo-model"})
    base_item.add_asset(
        "source-code",
        pystac.Asset(
            href="https://example.com/pipeline.py",
            extra_fields={"mlm:entrypoint": "demo.pipeline:training_pipeline"},
        ),
    )
    dataset_item = _make_item("dataset")

    missing_dataset_backend = SimpleNamespace(
        get_item=lambda collection, item_id: (
            base_item if collection == "base-models" else (_ for _ in ()).throw(KeyError(item_id))
        )
    )
    monkeypatch.setattr(client, "_get_backend", lambda: missing_dataset_backend)

    with pytest.raises(FairClientError, match="register_dataset first"):
        client.finetune(base_model_id="base-model", dataset_id="missing-dataset", model_name="demo")

    ready_backend = SimpleNamespace(
        get_item=lambda collection, item_id: base_item if collection == "base-models" else dataset_item
    )
    monkeypatch.setattr(client, "_get_backend", lambda: ready_backend)
    monkeypatch.setattr(client_module, "validate_compatibility", lambda base, ds: [])
    monkeypatch.setattr(
        client_module,
        "Client",
        lambda: SimpleNamespace(active_stack_model=SimpleNamespace(components={})),
    )
    monkeypatch.setattr(client_module, "generate_training_config", lambda *args, **kwargs: {"steps": {}})

    class _TrainingPipeline:
        def with_options(self, *, config_path: str, enable_cache: bool) -> Any:
            def _runner() -> None:
                return None

            return _runner

    monkeypatch.setattr(
        client_module.importlib,
        "import_module",
        lambda _: SimpleNamespace(training_pipeline=_TrainingPipeline()),
    )

    with pytest.raises(RuntimeError, match="Training pipeline returned no run"):
        client.finetune(base_model_id="base-model", dataset_id="dataset", model_name="demo")

    monkeypatch.setattr(
        client_module,
        "Client",
        lambda: SimpleNamespace(list_model_versions=lambda **kwargs: []),
    )

    with pytest.raises(FairClientError, match="Run finetune first"):
        client.promote("demo")

    version = SimpleNamespace(id="version-1", number="1")
    monkeypatch.setattr(
        client_module,
        "Client",
        lambda: SimpleNamespace(
            list_model_versions=lambda **kwargs: [version],
            list_model_version_pipeline_run_links=lambda **kwargs: SimpleNamespace(items=[]),
        ),
    )

    with pytest.raises(FairClientError, match="Cannot resolve base_model_id"):
        client.promote("demo")

    with pytest.raises(FairClientError, match="Cannot resolve dataset_id"):
        client.promote("demo", base_model_id="base-model")


def test_builder_helpers_cover_media_and_geojson(tmp_path: Path) -> None:
    assert _infer_source_code_media_type("model.py") == "text/x-python"
    assert _infer_source_code_media_type("widget.js") == "text/javascript"
    assert _infer_source_code_media_type("https://github.com/hotosm/fAIr-models") == "text/html"
    assert _infer_source_code_media_type("README.txt") == "text/plain"

    assert _raster_bands_from_model_input(
        [
            {"bands": ["red", {"name": "green"}, "red"]},
            {"bands": [{"name": "blue"}]},
        ]
    ) == [{"name": "red"}, {"name": "green"}, {"name": "blue"}]
    assert _raster_bands_from_model_input([{"bands": []}]) is None

    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    (labels_dir / "a.geojson").write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
                        },
                        "properties": {},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (labels_dir / "b.geojson").write_text(
        json.dumps(
            {
                "type": "Polygon",
                "coordinates": [[[2, 2], [3, 2], [3, 3], [2, 3], [2, 2]]],
            }
        ),
        encoding="utf-8",
    )

    geometry, bbox = geometry_and_bbox_from_geojson(str(labels_dir))
    assert geometry["type"] == "Polygon"
    assert bbox == [0, 0, 3, 3]

    with pytest.raises(ValueError, match="providers is required"):
        _validate_providers([])

    with pytest.raises(ValueError, match="each provider"):
        _validate_providers([{"name": "HOT"}])

    empty_labels_dir = tmp_path / "empty-labels"
    empty_labels_dir.mkdir()
    with pytest.raises(ValueError, match=r"No \.geojson files found"):
        geometry_and_bbox_from_geojson(str(empty_labels_dir))

    no_coordinates = tmp_path / "no-coordinates.geojson"
    no_coordinates.write_text(json.dumps({"type": "FeatureCollection", "features": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="No coordinates found"):
        geometry_and_bbox_from_geojson(str(no_coordinates))


def test_model_validator_covers_return_annotations_and_url_validation(tmp_path: Path) -> None:
    plain_name = ast.parse("def fn() -> str:\n    return ''\n").body[0]
    quoted_name = ast.parse('def fn() -> "bytes":\n    return b""\n').body[0]
    annotated = ast.parse("import typing\ndef fn() -> typing.Annotated[str, 'label']:\n    return ''\n").body[1]

    assert isinstance(plain_name, ast.FunctionDef)
    assert isinstance(quoted_name, ast.FunctionDef)
    assert isinstance(annotated, ast.FunctionDef)

    assert _return_annotation_name(plain_name) == "str"
    assert _return_annotation_name(quoted_name) == "bytes"
    assert _return_annotation_name(annotated) == "str"

    model_dir = tmp_path / "demo_model"
    _write_model_scaffold(model_dir, model_href="model.onnx")

    errors = validate_model(model_dir)
    assert f"{model_dir.name}: model asset href must be an https URL" in errors


def test_model_validator_reports_unreadable_test_steps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model_dir = tmp_path / "broken_model"
    _write_model_scaffold(model_dir)

    original_read_text = Path.read_text
    blocked_path = model_dir / "tests" / "test_steps.py"

    def _patched_read_text(path: Path, *args: Any, **kwargs: Any) -> str:
        if path == blocked_path:
            raise OSError("permission denied")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", _patched_read_text)

    errors = validate_model(model_dir)
    assert any("cannot read tests/test_steps.py" in error for error in errors)


def test_apply_zenml_patch_tolerates_missing_server_models(monkeypatch: pytest.MonkeyPatch) -> None:
    reloaded = importlib.reload(patch_module)
    assert hasattr(reloaded, "_apply")

    real_import = builtins.__import__

    def _fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "zenml.models.v2.misc.server_models":
            raise ModuleNotFoundError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    monkeypatch.delenv("FAIR_SKIP_ZENML_PATCH", raising=False)

    patch_module._apply()
