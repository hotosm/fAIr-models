from __future__ import annotations

import json
from typing import Any

import pystac
import pytest

from fair.stac.builders import build_base_model_item, build_dataset_item
from fair.zenml.config import (
    LABEL_DOMAIN,
    _scheduling_settings,
    _workload_selector,
    _workload_toleration,
    generate_inference_config,
    generate_training_config,
)


def _base_model(**overrides: Any):
    defaults: dict[str, Any] = {
        "item_id": "unet",
        "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]},
        "mlm_name": "unet",
        "mlm_architecture": "UNet",
        "mlm_tasks": ["semantic-segmentation"],
        "mlm_framework": "pytorch",
        "mlm_framework_version": "2.1.0",
        "mlm_input": [
            {
                "name": "image",
                "bands": [{"name": "red"}, {"name": "green"}, {"name": "blue"}],
                "input": {
                    "shape": [-1, 3, 512, 512],
                    "dim_order": ["batch", "bands", "height", "width"],
                    "data_type": "float32",
                },
                "pre_processing_function": {"format": "python", "expression": "mod:preprocess"},
            }
        ],
        "mlm_output": [
            {
                "name": "mask",
                "tasks": ["semantic-segmentation"],
                "result": {
                    "shape": [-1, 2, 512, 512],
                    "dim_order": ["batch", "channel", "height", "width"],
                    "data_type": "float32",
                },
                "classification:classes": [{"name": "bg", "value": 0}, {"name": "building", "value": 1}],
                "post_processing_function": {"format": "python", "expression": "mod:postprocess"},
            }
        ],
        "mlm_hyperparameters": {
            "training.epochs": 15,
            "training.batch_size": 4,
            "training.learning_rate": 0.0001,
            "inference.confidence_threshold": 0.5,
            "inference.min_class_value": 1,
        },
        "keywords": ["building"],
        "checkpoint_href": "https://example.com/weights.pt",
        "onnx_href": "https://example.com/model.onnx",
        "checkpoint_artifact_type": "torch.save",
        "mlm_pretrained": True,
        "mlm_pretrained_source": "OAM-TCD",
        "source_code_href": "https://github.com/example",
        "source_code_entrypoint": "mod:train",
        "training_runtime_href": "ghcr.io/hotosm/fair-unet:v1",
        "inference_runtime_href": "ghcr.io/hotosm/fair-unet:v1",
        "title": "UNet test model",
        "description": "Config test base model.",
        "fair_metrics_spec": [{"name": "accuracy", "description": "Pixel accuracy", "higher_is_better": True}],
        "providers": [{"name": "HOTOSM", "roles": ["producer"]}],
    }
    defaults.update(overrides)
    item = build_base_model_item(**defaults)
    item.properties["fair:hyperparameters_spec"] = [
        {"key": "epochs", "type": "int", "default": 15},
        {"key": "batch_size", "type": "int", "default": 4},
        {"key": "learning_rate", "type": "float", "default": 0.0001},
        {"key": "confidence_threshold", "type": "float", "default": 0.5},
        {"key": "min_class_value", "type": "int", "default": 1},
    ]
    return item


def _dataset(tmp_path, **kw: Any):
    geojson = {
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
    path = tmp_path / "labels.geojson"
    path.write_text(json.dumps(geojson))
    kw.setdefault("chips_hrefs", {"image": "data/oam/"})
    return build_dataset_item(
        item_id="ds",
        label_type="vector",
        label_tasks=["segmentation"],
        label_classes=[{"name": "building", "classes": ["building"]}],
        keywords=["building"],
        labels_href=str(path),
        title="Test Dataset",
        description="Config test dataset.",
        user_id="osm-test",
        providers=[{"name": "osm-test", "roles": ["producer"]}],
        **kw,
    )


def _with_inputs(item: pystac.Item, *names: str) -> pystac.Item:
    first = item.properties["mlm:input"][0]
    item.properties["mlm:input"] = [{**first, "name": name} for name in names]
    return item


def _local_model(**kw: Any):
    from fair.stac.builders import build_local_model_item

    defaults: dict[str, Any] = {
        "base_model_item": _base_model(),
        "item_id": "local-v1",
        "checkpoint_href": "s3://store/model/abc.pt",
        "onnx_href": "s3://store/model/abc.onnx",
        "mlm_hyperparameters": {
            "training.epochs": 15,
            "training.batch_size": 4,
            "training.learning_rate": 0.0001,
            "inference.confidence_threshold": 0.5,
            "inference.min_class_value": 1,
        },
        "keywords": ["building"],
        "base_model_href": "../base-models/unet/unet.json",
        "dataset_href": "../datasets/ds-1/ds-1.json",
        "version": "1",
        "title": "Local UNet",
        "description": "Finetuned.",
        "user_id": "osm-test",
        "providers": [{"name": "osm-test", "roles": ["producer"]}],
    }
    defaults.update(kw)
    return build_local_model_item(**defaults)


def test_training_config(tmp_path):
    cfg = generate_training_config(_base_model(), _dataset(tmp_path), model_name="finetuned")
    p = cfg["parameters"]
    hp = p["hyperparameters"]
    assert cfg["model"]["name"] == "finetuned"
    assert hp["epochs"] == 15 and hp["batch_size"] == 4
    assert "inference.min_class_value" not in hp
    assert p["dataset_chips"] == {"image": "data/oam/"}
    assert p["num_classes"] == 2
    assert p["base_model_weights"] == "https://example.com/weights.pt"
    assert cfg["settings"]["docker"]["parent_image"] == "ghcr.io/hotosm/fair-unet:v1"
    assert cfg["steps"]["train_model"]["parameters"]["model_name"] == "finetuned"
    assert cfg["steps"]["train_model"]["parameters"]["base_model_id"] == "unet"
    assert cfg["steps"]["evaluate_model"]["parameters"]["class_names"] == ["bg", "building"]


def test_training_config_passes_the_chips_of_each_model_input_by_name(tmp_path):
    dataset = _dataset(tmp_path, chips_hrefs={"post": "data/post/", "pre": "data/pre/"})
    cfg = generate_training_config(_with_inputs(_base_model(), "post", "pre"), dataset, model_name="damage")
    assert cfg["parameters"]["dataset_chips"] == {"post": "data/post/", "pre": "data/pre/"}


def test_training_config_reads_a_legacy_chips_asset_for_a_single_input_model(tmp_path):
    dataset = _dataset(tmp_path)
    dataset.add_asset("chips", dataset.assets.pop("chips-image"))
    cfg = generate_training_config(_base_model(), dataset, model_name="buildings")
    assert cfg["parameters"]["dataset_chips"] == {"image": "data/oam/"}


def test_training_config_raises_when_the_dataset_lacks_chips_for_a_model_input(tmp_path):
    dataset = _dataset(tmp_path, chips_hrefs={"post": "data/post/"})
    with pytest.raises(KeyError, match="pre"):
        generate_training_config(_with_inputs(_base_model(), "post", "pre"), dataset, model_name="damage")


def test_training_overrides(tmp_path):
    cfg = generate_training_config(
        _base_model(),
        _dataset(tmp_path),
        model_name="t",
        overrides={"epochs": 1},
    )
    hp = cfg["parameters"]["hyperparameters"]
    assert hp["epochs"] == 1
    assert hp["batch_size"] == 4


def test_inference_config():
    cfg = generate_inference_config(_local_model(), {"image": "/data/input/"})
    p = cfg["parameters"]
    assert p["model_uri"] == "s3://store/model/abc.onnx"
    assert p["input_images"] == {"image": "/data/input/"}
    assert p["inference_params"] == {"confidence_threshold": 0.5, "min_class_value": 1}
    assert cfg["settings"]["docker"]["parent_image"] == "ghcr.io/hotosm/fair-unet:v1"


def test_inference_config_keys_each_input_chips_by_name():
    model = _with_inputs(_local_model(), "post", "pre")
    cfg = generate_inference_config(model, {"pre": "s3://chips/pre", "post": "s3://chips/post"})
    assert cfg["parameters"]["input_images"] == {"post": "s3://chips/post", "pre": "s3://chips/pre"}


def test_inference_config_rejects_imagery_that_misses_a_model_input():
    model = _with_inputs(_local_model(), "post", "pre")
    with pytest.raises(ValueError, match=r"\['post', 'pre'\], got \['post'\]"):
        generate_inference_config(model, {"post": "s3://chips/post"})


def test_inference_config_applies_declared_overrides():
    cfg = generate_inference_config(_local_model(), {"image": "/data/input/"}, {"confidence_threshold": 0.8})
    assert cfg["parameters"]["inference_params"] == {"confidence_threshold": 0.8, "min_class_value": 1}


def test_inference_config_rejects_undeclared_overrides():
    with pytest.raises(ValueError, match="Unknown inference parameter"):
        generate_inference_config(_local_model(), {"image": "/data/input/"}, {"epochs": 3})


def test_inference_config_rejects_overrides_without_a_spec_entry():
    model = _local_model()
    model.properties["fair:hyperparameters_spec"] = [
        entry for entry in model.properties["fair:hyperparameters_spec"] if entry["key"] != "min_class_value"
    ]
    with pytest.raises(ValueError, match="min_class_value"):
        generate_inference_config(model, {"image": "/data/input/"}, {"min_class_value": 2})


def test_inference_config_rejects_overrides_of_the_wrong_type():
    with pytest.raises(ValueError, match="expected type 'float'"):
        generate_inference_config(_local_model(), {"image": "/data/input/"}, {"confidence_threshold": "high"})


# --- scheduling settings tests ---


def _item_with_accelerator(accelerator: str | None = None, count: int | None = None) -> pystac.Item:
    item = _base_model()
    if accelerator is not None:
        item.properties["mlm:accelerator"] = accelerator
    if count is not None:
        item.properties["mlm:accelerator_count"] = count
    return item


def test_k8s_settings_cuda():
    item = _item_with_accelerator("cuda", 2)
    settings = _scheduling_settings(item, "training")
    pod = settings["orchestrator.kubernetes"]["pod_settings"]
    assert pod["resources"]["requests"]["nvidia.com/gpu"] == "2"
    assert pod["resources"]["limits"]["nvidia.com/gpu"] == "2"
    assert len(pod["tolerations"]) == 2
    assert _workload_toleration("training") in pod["tolerations"]
    assert pod["node_selectors"] == _workload_selector("training")


def test_k8s_settings_cpu_returns_workload_toleration():
    item = _item_with_accelerator()
    settings = _scheduling_settings(item, "training")
    pod = settings["orchestrator.kubernetes"]["pod_settings"]
    assert pod["tolerations"] == [_workload_toleration("training")]
    assert pod["node_selectors"] == _workload_selector("training")
    assert pod["resources"]["requests"]["memory"] == "2Gi"
    assert pod["resources"]["limits"]["memory"] == "2.5Gi"


def test_k8s_settings_explicit_cpu():
    settings = _scheduling_settings(_item_with_accelerator("cpu"), "inference")
    pod = settings["orchestrator.kubernetes"]["pod_settings"]
    assert pod["tolerations"] == [_workload_toleration("inference")]
    assert pod["node_selectors"] == _workload_selector("inference")
    assert pod["resources"]["requests"]["memory"] == "1Gi"
    assert pod["resources"]["limits"]["memory"] == "2Gi"


def test_k8s_settings_default_count():
    item = _item_with_accelerator("cuda")
    settings = _scheduling_settings(item, "training")
    assert settings["orchestrator.kubernetes"]["pod_settings"]["resources"]["limits"]["nvidia.com/gpu"] == "1"


def test_k8s_resources_global_env_override(monkeypatch):
    monkeypatch.setenv("FAIR_K8S_MEMORY_REQUEST", "8Gi")
    monkeypatch.setenv("FAIR_K8S_MEMORY_LIMIT", "12Gi")
    settings = _scheduling_settings(_item_with_accelerator(), "training")
    resources = settings["orchestrator.kubernetes"]["pod_settings"]["resources"]
    assert resources["requests"]["memory"] == "8Gi"
    assert resources["limits"]["memory"] == "12Gi"


def test_k8s_resources_workload_env_overrides_global(monkeypatch):
    monkeypatch.setenv("FAIR_K8S_MEMORY_REQUEST", "8Gi")
    monkeypatch.setenv("FAIR_TRAINING_MEMORY_REQUEST", "16Gi")
    settings = _scheduling_settings(_item_with_accelerator(), "training")
    assert settings["orchestrator.kubernetes"]["pod_settings"]["resources"]["requests"]["memory"] == "16Gi"


def test_k8s_resources_inference_defaults():
    settings = _scheduling_settings(_item_with_accelerator(), "inference")
    resources = settings["orchestrator.kubernetes"]["pod_settings"]["resources"]
    assert resources["requests"]["memory"] == "1Gi"
    assert resources["limits"]["memory"] == "2Gi"


def test_k8s_resources_cpu_env(monkeypatch):
    monkeypatch.setenv("FAIR_TRAINING_CPU_REQUEST", "2")
    monkeypatch.setenv("FAIR_TRAINING_CPU_LIMIT", "4")
    settings = _scheduling_settings(_item_with_accelerator(), "training")
    resources = settings["orchestrator.kubernetes"]["pod_settings"]["resources"]
    assert resources["requests"]["cpu"] == "2"
    assert resources["limits"]["cpu"] == "4"


def test_k8s_settings_force_cpu_env(monkeypatch):
    monkeypatch.setenv("FAIR_FORCE_CPU", "1")
    settings = _scheduling_settings(_item_with_accelerator("cuda", 2), "training")
    pod = settings["orchestrator.kubernetes"]["pod_settings"]
    assert pod["tolerations"] == [_workload_toleration("training")]
    assert pod["node_selectors"] == _workload_selector("training")
    assert pod["resources"] == {"requests": {"memory": "2Gi"}, "limits": {"memory": "2.5Gi"}}


def test_workload_selectors_use_label_domain():
    selector = _workload_selector("training")
    assert f"{LABEL_DOMAIN}/training" in selector
    toleration = _workload_toleration("inference")
    assert toleration["key"] == f"{LABEL_DOMAIN}/inference"


def test_training_config_includes_k8s_settings(tmp_path):
    item = _item_with_accelerator("cuda", 1)
    ds = _dataset(tmp_path)
    cfg = generate_training_config(item, ds, model_name="gpu-test")
    assert "orchestrator.kubernetes" in cfg["steps"]["train_model"]["settings"]
    assert "orchestrator.kubernetes" in cfg["steps"]["evaluate_model"]["settings"]
    assert "orchestrator.kubernetes" not in cfg.get("settings", {})


def test_inference_config_includes_k8s_settings():
    local = _local_model()
    local.properties["mlm:accelerator"] = "cuda"
    cfg = generate_inference_config(local, {"image": "/images/"})
    assert "orchestrator.kubernetes" in cfg["steps"]["run_inference"]["settings"]
    assert "orchestrator.kubernetes" not in cfg.get("settings", {})


def test_a_lone_chips_asset_feeds_only_an_image_input(tmp_path):
    dataset = _dataset(tmp_path)
    dataset.assets["chips"] = dataset.assets.pop("chips-image")
    assert generate_training_config(_base_model(), dataset, model_name="m")["parameters"]["dataset_chips"] == {
        "image": "data/oam/"
    }
    with pytest.raises(KeyError, match="rgb"):
        generate_training_config(_with_inputs(_base_model(), "rgb"), dataset, model_name="m")


def test_inference_config_rejects_a_model_whose_input_names_are_not_slugs():
    with pytest.raises(ValueError, match="not slugs"):
        generate_inference_config(_with_inputs(_local_model(), "RGB chips"), {"RGB chips": "s3://chips"})
