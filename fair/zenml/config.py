from __future__ import annotations

import os
from typing import Any

import pystac

from fair.stac.constants import CONTAINER_REGISTRIES, OCI_IMAGE_INDEX_TYPE
from fair.utils.data import http_url_to_s3_uri

LABEL_DOMAIN = os.environ.get("FAIR_LABEL_DOMAIN", "fair.dev")


def _normalize_container_href(href: str) -> str:
    """Fix container registry URLs that PySTAC made relative."""
    for registry in CONTAINER_REGISTRIES:
        if (idx := href.find(registry)) != -1:
            return href[idx:]
    return href


def _extract_input_spec(mlm_input: list[dict[str, Any]]) -> dict[str, Any]:
    # Band names are model-internal, only chip_size is a pipeline param
    if not mlm_input:
        return {}
    shape = mlm_input[0].get("input", {}).get("shape", [])
    return {"chip_size": shape[-1]} if len(shape) == 4 else {}


def _extract_num_classes(mlm_output: list[dict[str, Any]]) -> int | None:
    if not mlm_output:
        return None
    classes = mlm_output[0].get("classification:classes", [])
    return len(classes) if classes else None


def _extract_class_names(mlm_output: list[dict[str, Any]]) -> list[str] | None:
    if not mlm_output:
        return None
    classes = mlm_output[0].get("classification:classes", [])
    return [c["name"] for c in classes if "name" in c] or None


def _force_cpu_mode() -> bool:
    return os.environ.get("FAIR_FORCE_CPU", "").lower() in {"1", "true", "yes", "on"}


def _workload_selector(workload: str) -> dict[str, str]:
    return {f"{LABEL_DOMAIN}/{workload}": "true"}


def _workload_toleration(workload: str) -> dict[str, str]:
    return {"key": f"{LABEL_DOMAIN}/{workload}", "operator": "Equal", "value": "true", "effect": "NoSchedule"}


_WORKLOAD_RESOURCE_DEFAULTS: dict[str, dict[str, str]] = {
    "training": {"memory_request": "4Gi", "memory_limit": "6Gi"},
    "inference": {"memory_request": "2Gi", "memory_limit": "4Gi"},
}


def _resource_value(workload: str, resource: str) -> str | None:
    upper_workload = workload.upper()
    upper_resource = resource.upper()
    return (
        os.environ.get(f"FAIR_{upper_workload}_{upper_resource}")
        or os.environ.get(f"FAIR_K8S_{upper_resource}")
        or _WORKLOAD_RESOURCE_DEFAULTS.get(workload, {}).get(resource)
    )


def _cpu_resources(workload: str) -> dict[str, dict[str, str]]:
    resources: dict[str, dict[str, str]] = {"requests": {}, "limits": {}}
    memory_request = _resource_value(workload, "memory_request")
    memory_limit = _resource_value(workload, "memory_limit")
    cpu_request = _resource_value(workload, "cpu_request")
    cpu_limit = _resource_value(workload, "cpu_limit")
    if memory_request:
        resources["requests"]["memory"] = memory_request
    if memory_limit:
        resources["limits"]["memory"] = memory_limit
    if cpu_request:
        resources["requests"]["cpu"] = cpu_request
    if cpu_limit:
        resources["limits"]["cpu"] = cpu_limit
    return resources


def _scheduling_settings(item: pystac.Item, workload: str) -> dict[str, Any]:
    cpu_resources = _cpu_resources(workload)

    if _force_cpu_mode():
        return {
            "orchestrator.kubernetes": {
                "pod_settings": {
                    "node_selectors": _workload_selector(workload),
                    "tolerations": [_workload_toleration(workload)],
                    "resources": cpu_resources,
                }
            }
        }

    accelerator = item.properties.get("mlm:accelerator")
    if not accelerator or accelerator in ("amd64", "cpu"):
        return {
            "orchestrator.kubernetes": {
                "pod_settings": {
                    "node_selectors": _workload_selector(workload),
                    "tolerations": [_workload_toleration(workload)],
                    "resources": cpu_resources,
                }
            }
        }
    count = str(item.properties.get("mlm:accelerator_count", 1))
    return {
        "orchestrator.kubernetes": {
            "pod_settings": {
                "node_selectors": _workload_selector(workload),
                "resources": {"requests": {"nvidia.com/gpu": count}, "limits": {"nvidia.com/gpu": count}},
                "tolerations": [
                    _workload_toleration(workload),
                    {"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"},
                ],
            }
        }
    }


def generate_training_config(
    base_model_item: pystac.Item,
    dataset_item: pystac.Item,
    model_name: str,
    overrides: dict[str, Any] | None = None,
    experiment_tracker: str | None = None,
) -> dict[str, Any]:
    # ZenML config schema: https://docs.zenml.io/concepts/steps_and_pipelines/yaml_configuration
    props = base_model_item.properties

    hyperparams: dict[str, Any] = dict(props.get("mlm:hyperparameters", {}))
    input_spec = _extract_input_spec(props.get("mlm:input", []))
    num_classes = _extract_num_classes(props.get("mlm:output", []))
    class_names = _extract_class_names(props.get("mlm:output", []))

    ds_props = dataset_item.assets
    chips_asset = ds_props.get("chips")
    labels_asset = ds_props.get("labels")
    if chips_asset is None:
        msg = f"Dataset item '{dataset_item.id}' missing 'chips' asset"
        raise KeyError(msg)
    if labels_asset is None:
        msg = f"Dataset item '{dataset_item.id}' missing 'labels' asset"
        raise KeyError(msg)
    chips_href = http_url_to_s3_uri(chips_asset.href)
    labels_href = http_url_to_s3_uri(labels_asset.href)

    model_asset = base_model_item.assets.get("model")
    if model_asset is None:
        msg = f"Base model item '{base_model_item.id}' missing 'model' asset"
        raise KeyError(msg)

    hyperparams.update(input_spec)
    if overrides:
        hyperparams.update(overrides)

    parameters: dict[str, Any] = {
        "base_model_weights": http_url_to_s3_uri(model_asset.href),
        "dataset_chips": chips_href,
        "dataset_labels": labels_href,
        "hyperparameters": hyperparams,
    }
    if num_classes is not None:
        parameters["num_classes"] = num_classes

    train_step_params: dict[str, Any] = {
        "model_name": model_name,
        "base_model_id": base_model_item.id,
        "dataset_id": dataset_item.id,
    }
    eval_step_params: dict[str, Any] = {}
    if class_names is not None:
        eval_step_params["class_names"] = class_names

    config: dict[str, Any] = {
        "model": {"name": model_name},
        "parameters": parameters,
        "tags": list(
            dict.fromkeys(
                [
                    f"model:{model_name}",
                    f"base-model:{base_model_item.id}",
                    f"dataset:{dataset_item.id}",
                ]
            )
        ),
    }

    runtime = base_model_item.assets.get("mlm:training")
    if runtime and runtime.media_type == OCI_IMAGE_INDEX_TYPE:
        docker_cfg: dict[str, Any] = {
            "parent_image": _normalize_container_href(runtime.href),
            "skip_build": True,
        }
        config["settings"] = {"docker": docker_cfg}

    if experiment_tracker:
        config.setdefault("steps", {}).setdefault("train_model", {})["experiment_tracker"] = experiment_tracker
        config.setdefault("steps", {}).setdefault("evaluate_model", {})["experiment_tracker"] = experiment_tracker
        config.setdefault("settings", {})["experiment_tracker.mlflow"] = {
            "experiment_name": model_name,
            "run_name": f"train/{model_name}",
        }

    if train_step_params:
        config.setdefault("steps", {}).setdefault("train_model", {}).setdefault("parameters", {}).update(
            train_step_params
        )
    if eval_step_params:
        config.setdefault("steps", {}).setdefault("evaluate_model", {}).setdefault("parameters", {}).update(
            eval_step_params
        )

    k8s = _scheduling_settings(base_model_item, "training")
    if k8s:
        config.setdefault("settings", {}).update(k8s)

    return config


def generate_inference_config(
    model_item: pystac.Item,
    input_images_path: str,
) -> dict[str, Any]:
    # Works for both base-model and local-model items (same MLM structure)
    props = model_item.properties
    input_spec = _extract_input_spec(props.get("mlm:input", []))

    model_asset = model_item.assets.get("model")
    if model_asset is None:
        msg = f"Model item '{model_item.id}' missing 'model' asset"
        raise KeyError(msg)
    zenml_art_id = model_asset.extra_fields.get("zenml:artifact_version_id")

    parameters: dict[str, Any] = {
        "model_uri": http_url_to_s3_uri(model_asset.href),
        "input_images": http_url_to_s3_uri(input_images_path),
        **input_spec,
    }

    # Base model: no ZenML artifact, load from pretrained weights directly
    # Finetuned model: resolve from artifact store via ID (fast) or URI (fallback)
    if zenml_art_id:
        parameters["zenml_artifact_version_id"] = zenml_art_id
    else:
        parameters["use_base_model"] = True

    num_classes = _extract_num_classes(props.get("mlm:output", []))
    if num_classes is not None:
        parameters["num_classes"] = num_classes

    config: dict[str, Any] = {
        "parameters": parameters,
        "tags": [
            f"model:{model_item.id}",
        ],
    }

    runtime = model_item.assets.get("mlm:inference")
    if runtime and runtime.media_type == OCI_IMAGE_INDEX_TYPE:
        docker_cfg: dict[str, Any] = {
            "parent_image": _normalize_container_href(runtime.href),
            "skip_build": True,
        }
        config["settings"] = {"docker": docker_cfg}

    k8s = _scheduling_settings(model_item, "inference")
    if k8s:
        config.setdefault("settings", {}).update(k8s)

    return config
