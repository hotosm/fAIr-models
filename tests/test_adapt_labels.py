import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pystac
import pytest

import fair.client as client_module
from fair.client import FairClient, FairClientError

_GEOMETRY = {"type": "Point", "coordinates": [0.5, 0.5]}


def _item(item_id: str, properties: dict, assets: dict[str, str] | None = None) -> pystac.Item:
    item = pystac.Item(item_id, _GEOMETRY, [0, 0, 1, 1], datetime(2026, 1, 1, tzinfo=UTC), properties)
    for key, href in (assets or {}).items():
        item.add_asset(key, pystac.Asset(href=href))
    return item


_TREES = _item(
    "trees",
    {
        "mlm:output": [
            {
                "name": "mask",
                "classification:classes": [
                    {"name": "background", "value": 0},
                    {"name": "tree", "value": 1, "fair:osm_tags": {"natural": ["tree"]}},
                ],
            }
        ]
    },
)


def _dataset(tmp_path: Path, features: list[dict], values: list[str]) -> pystac.Item:
    labels = tmp_path / "dataset" / "labels"
    labels.mkdir(parents=True)
    (labels / "labels.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": features}))
    properties = {"label:properties": ["natural"], "label:classes": [{"name": "natural", "classes": values}]}
    return _item("trees-sample", properties, {"labels": str(labels)})


def _tree(natural: str) -> dict:
    return {
        "type": "Feature",
        "properties": {"natural": natural, "tags": {"natural": natural, "height": "4"}},
        "geometry": _GEOMETRY,
    }


def test_local_labels_are_written_next_to_the_config(tmp_path) -> None:
    client = FairClient(catalog_path=str(tmp_path / "catalog.json"))
    dataset = _dataset(tmp_path, [_tree("tree")], ["tree"])

    href = client._adapt_labels(_TREES, dataset, "my-trees", None, tmp_path / "config")

    assert Path(href).parent == tmp_path / "config" / "labels" / "my-trees"
    features = json.loads((Path(href) / "labels.geojson").read_text())["features"]
    assert features[0]["properties"] == {"natural": "tree", "tags": {"natural": "tree", "height": "4"}, "label": 1}
    assert client._adapt_labels(_TREES, dataset, "my-trees", None, tmp_path / "config") == href


def test_class_map_changes_the_file(tmp_path) -> None:
    client = FairClient(catalog_path=str(tmp_path / "catalog.json"))
    dataset = _dataset(tmp_path, [_tree("palm")], ["palm"])

    href = client._adapt_labels(_TREES, dataset, "my-trees", {"natural=palm": "tree"}, tmp_path / "config")

    assert json.loads((Path(href) / "labels.geojson").read_text())["features"][0]["properties"]["label"] == 1
    with pytest.raises(FairClientError, match=r"does not fit base model 'trees'.*natural=palm"):
        client._adapt_labels(_TREES, dataset, "my-trees", None, tmp_path / "config")


def test_remote_labels_are_uploaded_to_the_artifact_store(tmp_path, monkeypatch) -> None:
    client = FairClient(catalog_path=str(tmp_path / "catalog.json"))
    dataset = _dataset(tmp_path, [_tree("tree")], ["tree"])
    local_labels = dataset.assets["labels"].href
    dataset.assets["labels"].href = "s3://bucket/datasets/trees-sample/v1/labels"
    monkeypatch.setattr(client_module, "resolve_directory", lambda href, pattern: Path(local_labels))
    monkeypatch.setattr(client, "_artifact_store_prefix", lambda: "s3://store")
    mirrored: list[tuple[Path, str]] = []
    monkeypatch.setattr(client_module, "mirror", lambda src, dest: mirrored.append((src, dest)))

    href = client._adapt_labels(_TREES, dataset, "my-trees", None, tmp_path / "config")

    assert href.startswith("s3://store/training-labels/my-trees/")
    assert mirrored == [(tmp_path / "config" / "labels" / "my-trees" / href.rsplit("/", 1)[1], href)]


def test_finetune_passes_the_class_map_to_the_label_step(tmp_path, monkeypatch) -> None:
    client = FairClient(catalog_path=str(tmp_path / "catalog.json"), config_dir=str(tmp_path / "config"))
    dataset = _dataset(tmp_path, [_tree("palm")], ["palm"])
    base = _TREES.clone()
    base.add_asset("mlm:training", pystac.Asset(href="ghcr.io/hotosm/trees:latest"))
    backend = SimpleNamespace(get_item=lambda collection, item_id: base if collection == "base-models" else dataset)
    monkeypatch.setattr(client, "_get_backend", lambda: backend)
    monkeypatch.setattr(client_module, "validate_compatibility", lambda base, ds: [])
    monkeypatch.setattr(client, "_pipeline_module_from_item", lambda item: "pipeline")
    monkeypatch.setattr(
        client_module, "Client", lambda: SimpleNamespace(active_stack_model=SimpleNamespace(components={}))
    )
    configs: list[str | None] = []
    monkeypatch.setattr(
        client_module,
        "generate_training_config",
        lambda *args, labels_href=None, **kwargs: configs.append(labels_href) or {},
    )
    monkeypatch.setattr(client_module.importlib, "import_module", lambda name: SimpleNamespace())

    client._prepare_training_pipeline(
        base_model_id="trees",
        dataset_id="trees-sample",
        model_name="m",
        overrides=None,
        class_map={"natural=palm": "tree"},
    )

    assert configs[0] is not None
    assert json.loads((Path(configs[0]) / "labels.geojson").read_text())["features"][0]["properties"]["label"] == 1


def test_a_dataset_without_labels_is_rejected(tmp_path) -> None:
    client = FairClient(catalog_path=str(tmp_path / "catalog.json"))
    dataset = _dataset(tmp_path, [_tree("tree")], ["tree"])
    del dataset.assets["labels"]

    with pytest.raises(FairClientError, match="no 'labels' asset"):
        client._adapt_labels(_TREES, dataset, "my-trees", None, tmp_path / "config")
