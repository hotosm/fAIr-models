from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import MagicMock

import pystac
import pytest

from fair.stac.builders import build_base_model_item, build_dataset_item, build_local_model_item
from fair.stac.catalog_manager import StacCatalogManager
from fair.stac.collections import initialize_catalog
from fair.stac.constants import BASE_MODELS_COLLECTION, DATASETS_COLLECTION
from fair.stac.validators import validate_model_asset_urls, validate_model_weight_href, validate_predictions_geojson
from fair.utils.data import (
    count_chips,
    http_url_to_s3_uri,
    resolve_directory,
    s3_uri_to_http_url,
    upload_item_assets,
    upload_local_directory,
)
from fair.utils.model_validator import validate_model
from fair.zenml.config import (
    _extract_class_names,
    _extract_num_classes,
    _force_cpu_mode,
    _normalize_container_href,
    generate_inference_config,
    generate_training_config,
)
from fair.zenml.promotion import (
    _materialize_checkpoint_bytes,
    _materialize_onnx_bytes,
    _upload_model_artifacts,
    _upload_training_metrics,
    publish_promoted_model,
)

_NOW = datetime(2024, 1, 1, tzinfo=UTC)


class _Response:
    def __init__(self, status: int) -> None:
        self.status = status

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_args: object) -> None:
        return None


def _item(item_id: str = "item", *, keywords: list[str] | None = None) -> pystac.Item:
    return pystac.Item(
        id=item_id,
        geometry={"type": "Point", "coordinates": [0, 0]},
        bbox=[0, 0, 0, 0],
        datetime=_NOW,
        properties={"keywords": keywords or ["building", "polygon"]},
    )


def _base_model() -> pystac.Item:
    return build_base_model_item(
        item_id="example-model",
        geometry={"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]},
        mlm_name="example-model",
        mlm_architecture="UNet",
        mlm_tasks=["semantic-segmentation"],
        mlm_framework="pytorch",
        mlm_framework_version="2.1.0",
        mlm_input=[],
        mlm_output=[
            {
                "classification:classes": [{"name": "background"}, {"name": "building"}],
            }
        ],
        mlm_hyperparameters={
            "training.epochs": 10,
            "training.batch_size": 2,
            "inference.confidence_threshold": 0.5,
        },
        keywords=["building", "polygon", "semantic-segmentation"],
        checkpoint_href="https://example.com/checkpoint.pt",
        onnx_href="https://example.com/model.onnx",
        checkpoint_artifact_type="torch.save",
        mlm_pretrained=False,
        mlm_pretrained_source=None,
        source_code_href="https://example.com/repo",
        source_code_entrypoint="mod:train",
        training_runtime_href="ghcr.io/hotosm/example:train",
        inference_runtime_href="ghcr.io/hotosm/example:infer",
        title="Example Model",
        description="Example model for coverage tests.",
        fair_metrics_spec=[{"name": "accuracy", "description": "Accuracy", "higher_is_better": True}],
        providers=[{"name": "HOTOSM", "roles": ["producer"]}],
    )


def _dataset_item(tmp_path: Path) -> pystac.Item:
    label_path = tmp_path / "labels.geojson"
    label_path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[[85.0, 27.0], [85.1, 27.0], [85.1, 27.1], [85.0, 27.1], [85.0, 27.0]]],
                        },
                        "properties": {},
                    }
                ],
            }
        )
    )
    return build_dataset_item(
        item_id="dataset-item",
        label_type="vector",
        label_tasks=["segmentation"],
        label_classes=[{"name": "building", "classes": ["building"]}],
        keywords=["building", "polygon", "semantic-segmentation"],
        chips_href="data/chips",
        labels_href=str(label_path),
        title="Dataset",
        description="Coverage dataset.",
        user_id="osm-test",
        providers=[{"name": "osm-test", "roles": ["producer"]}],
    )


def _local_model() -> pystac.Item:
    return build_local_model_item(
        base_model_item=_base_model(),
        item_id="local-model",
        checkpoint_href="s3://bucket/checkpoint.pt",
        onnx_href="s3://bucket/model.onnx",
        mlm_hyperparameters={
            "training.epochs": 10,
            "training.batch_size": 2,
            "inference.confidence_threshold": 0.5,
        },
        keywords=["building", "polygon", "semantic-segmentation"],
        base_model_href="https://example.com/base.json",
        dataset_href="https://example.com/dataset.json",
        version="1",
        title="Local Model",
        description="Finetuned model.",
        user_id="osm-test",
        providers=[{"name": "osm-test", "roles": ["producer"]}],
    )


def _catalog(tmp_path: Path) -> StacCatalogManager:
    catalog_path = tmp_path / "catalog.json"
    initialize_catalog(str(catalog_path))
    manager = StacCatalogManager(str(catalog_path))
    manager.publish_item(BASE_MODELS_COLLECTION, _base_model())
    manager.publish_item(DATASETS_COLLECTION, _dataset_item(tmp_path))
    return manager


def test_validator_paths_cover_assets_and_geojson(monkeypatch: pytest.MonkeyPatch) -> None:
    item = _item()
    item.add_asset("checkpoint", pystac.Asset(href="weights.pt"))

    errors = validate_model_asset_urls(item, required_keys=("checkpoint", "model"), optional_keys=())
    assert any("href must be an http(s) URL" in error for error in errors)
    assert any("Missing required asset 'model'" in error for error in errors)
    assert validate_model_weight_href(item)

    remote_item = _item("remote")
    remote_item.add_asset("checkpoint", pystac.Asset(href="https://example.com/checkpoint.pt"))
    remote_item.add_asset("model", pystac.Asset(href="https://example.com/model.onnx"))

    monkeypatch.setattr("urllib.request.urlopen", lambda *_args, **_kwargs: _Response(500))
    errors = validate_model_asset_urls(remote_item, required_keys=("checkpoint",), optional_keys=("model",))
    assert any("HTTP 500" in error for error in errors)

    monkeypatch.setattr("urllib.request.urlopen", lambda *_args, **_kwargs: _Response(200))
    assert validate_model_asset_urls(remote_item, required_keys=("checkpoint",), optional_keys=()) == []

    def _raise_url_error(*_args: object, **_kwargs: object) -> None:
        raise OSError("offline")

    monkeypatch.setattr("urllib.request.urlopen", _raise_url_error)
    errors = validate_model_asset_urls(remote_item, required_keys=("checkpoint",), optional_keys=())
    assert any("URL not accessible" in error for error in errors)

    model = _item("model", keywords=["building", "polygon"])
    bad_geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "NotFeature",
                "geometry": {"type": "Point", "coordinates": [0, 0]},
            }
        ],
    }
    geo_errors = validate_predictions_geojson(bad_geojson, model)
    assert any("must be 'Feature'" in error for error in geo_errors)
    assert any("not allowed by keyword geometry type" in error for error in geo_errors)
    assert any("missing 'properties'" in error for error in geo_errors)

    assert validate_predictions_geojson({"type": "Wrong"})
    assert validate_predictions_geojson({"type": "FeatureCollection", "features": "bad"})
    assert validate_predictions_geojson(bad_geojson, _item("plain", keywords=["building"])) == [
        "features[0].type must be 'Feature'",
        "features[0] missing 'properties'",
    ]


def test_data_helpers_cover_conversions_counts_and_uploads(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("AWS_ENDPOINT_URL", "https://minio.example.com")
    assert s3_uri_to_http_url("s3://bucket/path/file.tif") == "https://minio.example.com/bucket/path/file.tif"
    assert http_url_to_s3_uri("https://minio.example.com/bucket/path/file.tif") == "s3://bucket/path/file.tif"

    monkeypatch.delenv("AWS_ENDPOINT_URL", raising=False)
    monkeypatch.setenv("AWS_REGION", "eu-west-1")
    assert s3_uri_to_http_url("/tmp/local-file.tif") == "/tmp/local-file.tif"
    assert s3_uri_to_http_url("s3://bucket/path/file.tif") == "https://bucket.s3.eu-west-1.amazonaws.com/path/file.tif"
    assert http_url_to_s3_uri("https://bucket.s3.eu-west-1.amazonaws.com/path/file.tif") == "s3://bucket/path/file.tif"
    assert http_url_to_s3_uri("not-a-url") == "not-a-url"

    for name in ("a.tif", "b.tiff", "c.png", "d.jpg", "e.jpeg"):
        (tmp_path / name).write_bytes(b"x")
    (tmp_path / "ignore.txt").write_text("x")
    assert count_chips(str(tmp_path)) == 5

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "nested").mkdir()
    (source_dir / "nested" / "tile.tif").write_bytes(b"tile")

    class DummyRemotePath:
        storage: ClassVar[dict[str, bytes]] = {}

        def __init__(self, path: str) -> None:
            self.path = path

        def __truediv__(self, other: Path) -> DummyRemotePath:
            return DummyRemotePath(f"{self.path.rstrip('/')}/{other.as_posix()}")

        def write_bytes(self, data: bytes) -> None:
            self.storage[self.path] = data

    monkeypatch.setattr("fair.utils.data.UPath", DummyRemotePath)
    upload_local_directory(source_dir, "s3://bucket/prefix")
    assert DummyRemotePath.storage["s3://bucket/prefix/nested/tile.tif"] == b"tile"

    missing_item = _item("missing")
    missing_item.add_asset("missing", pystac.Asset(href=str(tmp_path / "does-not-exist.bin")))
    upload_item_assets(missing_item, "s3://bucket/data", "datasets")
    assert missing_item.assets["missing"].href.endswith("does-not-exist.bin")

    monkeypatch.setattr("fair.utils.data.list_files", lambda *_args, **_kwargs: ["s3://bucket/file.tif"])
    monkeypatch.setattr("fair.utils.data.resolve_path", lambda *_args, **_kwargs: SimpleNamespace(parent=None))
    with pytest.raises(ValueError, match="dest_dir is still None"):
        resolve_directory("s3://bucket/path")


def test_model_validator_covers_return_types_and_stac_branches(tmp_path: Path) -> None:
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tmp_path / "README.md").write_text("# Model\n")
    (tests_dir / "test_steps.py").write_text(
        "def test_split_dataset(): ...\n"
        "def test_train_model(): ...\n"
        "def test_evaluate_model(): ...\n"
        "def test_export_onnx(): ...\n"
    )
    (tmp_path / "pipeline.py").write_text(
        "from zenml import pipeline, step\n\n"
        "@step\n"
        "def split_dataset():\n    return None\n\n"
        "@step\n"
        "def export_onnx() -> str:\n    return 'model.onnx'\n\n"
        "@step\n"
        "def train_model() -> str:\n    return 'bad'\n\n"
        "@pipeline\n"
        "def training_pipeline():\n    return None\n\n"
        "@pipeline\n"
        "def inference_pipeline():\n    return None\n"
    )

    errors = validate_model(tmp_path)
    assert any("missing stac-item.json" in error for error in errors)

    (tmp_path / "stac-item.json").write_text("{not-json")
    errors = validate_model(tmp_path)
    assert any("export_onnx must return bytes" in error for error in errors)
    assert any("train_model must not return str" in error for error in errors)

    (tmp_path / "stac-item.json").write_text(json.dumps({"assets": {"checkpoint": {"href": "weights.pt"}}}))
    errors = validate_model(tmp_path)
    assert any("checkpoint asset href must be an https URL" in error for error in errors)
    assert any("model asset href is required" in error for error in errors)


def test_config_helpers_cover_edge_cases(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    assert _normalize_container_href("prefix/ghcr.io/hotosm/fair-model:latest") == "ghcr.io/hotosm/fair-model:latest"
    assert (
        _normalize_container_href("registry-not-recognized/example:latest") == "registry-not-recognized/example:latest"
    )
    assert _extract_num_classes([]) is None
    assert _extract_class_names([]) is None
    assert _extract_num_classes([{"classification:classes": [{"name": "a"}, {"name": "b"}]}]) == 2
    assert _extract_class_names([{"classification:classes": [{"name": "a"}, {"value": 2}]}]) == ["a"]

    monkeypatch.setenv("FAIR_FORCE_CPU", "yes")
    assert _force_cpu_mode() is True
    monkeypatch.setenv("FAIR_FORCE_CPU", "no")
    assert _force_cpu_mode() is False

    tracker_config = generate_training_config(
        _base_model(),
        _dataset_item(tmp_path),
        model_name="tracked",
        experiment_tracker="mlflow_tracker",
    )
    assert tracker_config["steps"]["train_model"]["experiment_tracker"] == "mlflow_tracker"
    assert tracker_config["settings"]["experiment_tracker.mlflow"]["experiment_name"] == "tracked"

    base_item = _base_model()
    dataset_item = _dataset_item(tmp_path)
    del dataset_item.assets["chips"]
    with pytest.raises(KeyError, match="missing 'chips' asset"):
        generate_training_config(base_item, dataset_item, model_name="demo")

    dataset_item = _dataset_item(tmp_path)
    del dataset_item.assets["labels"]
    with pytest.raises(KeyError, match="missing 'labels' asset"):
        generate_training_config(base_item, dataset_item, model_name="demo")

    base_item = _base_model()
    del base_item.assets["checkpoint"]
    with pytest.raises(KeyError, match="missing 'checkpoint' asset"):
        generate_training_config(base_item, _dataset_item(tmp_path), model_name="demo")

    local_model = _local_model()
    del local_model.assets["model"]
    with pytest.raises(KeyError, match=r"missing 'model' \(ONNX\) asset"):
        generate_inference_config(local_model, "/images")


def test_base_model_stac_items_publish_onnx_assets() -> None:
    model_roots = {
        Path("models/resnet18_classification/stac-item.json"): Path(
            "models/resnet18_classification/artifacts/resnet18_classification.onnx"
        ),
        Path("models/unet_segmentation/stac-item.json"): Path(
            "models/unet_segmentation/artifacts/unet_segmentation.onnx"
        ),
        Path("models/yolo11n_detection/stac-item.json"): Path(
            "models/yolo11n_detection/artifacts/yolo11n_detection.onnx"
        ),
    }

    for stac_path, artifact_path in model_roots.items():
        item = json.loads(stac_path.read_text())
        model_asset = item["assets"].get("model")
        assert model_asset is not None, f"{stac_path} is missing a model asset"
        assert artifact_path.exists(), f"{artifact_path} is missing"
        assert model_asset["href"].startswith("https://")
        assert artifact_path.name in model_asset["href"]
        assert "mlm:model" in model_asset.get("roles", [])


def test_promotion_helpers_cover_materialization_and_storage(monkeypatch: pytest.MonkeyPatch) -> None:
    onnx_art = MagicMock()
    onnx_art.load.return_value = b"onnx"
    assert _materialize_onnx_bytes(onnx_art) == b"onnx"

    onnx_art.load.return_value = bytearray(b"abc")
    assert _materialize_onnx_bytes(onnx_art) == b"abc"

    onnx_art.load.return_value = memoryview(b"xyz")
    assert _materialize_onnx_bytes(onnx_art) == b"xyz"

    onnx_art.load.return_value = "bad"
    with pytest.raises(TypeError):
        _materialize_onnx_bytes(onnx_art)

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(save=lambda _model, buffer: buffer.write(b"checkpoint")),
    )

    weights_art = MagicMock()
    weights_art.load.return_value = {"weights": [1, 2, 3]}
    assert _materialize_checkpoint_bytes(weights_art) == b"checkpoint"

    class DummyRemotePath:
        storage: ClassVar[dict[str, bytes | str]] = {}

        def __init__(self, path: str) -> None:
            self.path = path

        def write_bytes(self, data: bytes) -> None:
            self.storage[self.path] = data

        def write_text(self, data: str) -> None:
            self.storage[self.path] = data

    monkeypatch.setattr("upath.UPath", DummyRemotePath)
    monkeypatch.setattr("fair.zenml.promotion.s3_uri_to_http_url", lambda path: f"https://cdn.example/{path[5:]}")

    checkpoint_path, onnx_path = _upload_model_artifacts(weights_art, MagicMock(load=lambda: b"onnx"), "demo", None)
    assert Path(checkpoint_path).exists()
    assert Path(onnx_path).exists()

    remote_checkpoint, remote_onnx = _upload_model_artifacts(
        weights_art,
        MagicMock(load=lambda: b"onnx"),
        "demo",
        "s3://bucket",
    )
    assert remote_checkpoint == "https://cdn.example/bucket/local-models/demo/checkpoint/demo.pt"
    assert remote_onnx == "https://cdn.example/bucket/local-models/demo/model/demo.onnx"

    local_metrics = _upload_training_metrics({"train_loss": [1.0], "val_loss": [2.0]}, "demo", None)
    assert local_metrics is not None and Path(local_metrics).exists()
    assert json.loads(Path(local_metrics).read_text())["epochs"] == [1]

    remote_metrics = _upload_training_metrics({"train_loss": [1.0], "val_loss": [2.0]}, "demo", "s3://bucket")
    assert remote_metrics == "https://cdn.example/bucket/local-models/demo/training-metrics/demo.json"


def test_publish_promoted_model_covers_missing_onnx_and_local_store(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(save=lambda _model, buffer: buffer.write(b"checkpoint")),
    )
    manager = _catalog(tmp_path)

    run = MagicMock()
    run.steps.get.return_value = None
    run.config.parameters = {}
    run_page = MagicMock()
    run_page.items = []

    mv = MagicMock()
    mv.id = "mv-1"
    mv.run_metadata = {}
    weights_art = MagicMock()
    weights_art.id = "weights-artifact"
    weights_art.load.return_value = {"weights": [1]}
    mv.get_artifact.side_effect = lambda name: {"trained_model": weights_art}.get(name)

    client = MagicMock()
    client.get_model_version.return_value = mv
    client.list_model_version_pipeline_run_links.return_value = run_page
    monkeypatch.setattr("fair.zenml.promotion.Client", lambda: client)

    with pytest.raises(RuntimeError, match="No 'onnx_model' artifact"):
        publish_promoted_model(
            model_name="demo-model",
            version=1,
            catalog_manager=manager,
            base_model_item_id="example-model",
            dataset_item_id="dataset-item",
            user_id="osm-test",
            description="demo",
        )

    onnx_art = MagicMock()
    onnx_art.id = "onnx-artifact"
    onnx_art.load.return_value = b"onnx"
    mv.id = "mv-2"
    mv.get_artifact.side_effect = lambda name: {"trained_model": weights_art, "onnx_model": onnx_art}.get(name)
    monkeypatch.setattr(
        "fair.zenml.promotion.validate_model_asset_urls",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("should not run for local files")),
    )

    item = publish_promoted_model(
        model_name="demo-model",
        version=2,
        catalog_manager=manager,
        base_model_item_id="example-model",
        dataset_item_id="dataset-item",
        user_id="osm-test",
        description="demo",
        artifact_store_prefix=None,
    )
    assert item.id == "mv-2"

    monkeypatch.setattr("fair.zenml.promotion.validate_model_asset_urls", lambda *_args, **_kwargs: ["bad"])
    monkeypatch.setattr(
        "fair.zenml.promotion._upload_model_artifacts",
        lambda **_kwargs: ("https://example.com/checkpoint.pt", "https://example.com/model.onnx"),
    )
    mv.id = "mv-3"
    with pytest.raises(RuntimeError, match="Asset URL validation failed"):
        publish_promoted_model(
            model_name="demo-model",
            version=3,
            catalog_manager=manager,
            base_model_item_id="example-model",
            dataset_item_id="dataset-item",
            user_id="osm-test",
            description="demo",
            artifact_store_prefix="s3://bucket",
            geometry={"type": "Point", "coordinates": [0, 0]},
        )
