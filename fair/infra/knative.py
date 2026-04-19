"""Dynamic KNative service management for live model serving.

KNative operator, Kourier, and the ingress bridge are provisioned once by
Terraform (infra/dok8s/services.tf). Per-model KNative services are created
on demand by `fair.client.register_base_model` via `ensure_knative_service`.
"""

from __future__ import annotations

from typing import Any

import pystac

KNATIVE_GROUP = "serving.knative.dev"
KNATIVE_VERSION = "v1"
KNATIVE_PLURAL = "services"
DEFAULT_NAMESPACE = "predict"


def knative_service_name(name: str) -> str:
    """Convert a model identifier to a DNS-1035 label accepted by KNative."""
    return str(name).lower().replace("_", "-")


def _module_from_entrypoint(entrypoint: str) -> str:
    if ":" not in entrypoint:
        msg = f"Invalid mlm:entrypoint '{entrypoint}', expected 'module.path:function'"
        raise ValueError(msg)
    return entrypoint.rsplit(":", 1)[0]


def _service_name(item: pystac.Item) -> str:
    return knative_service_name(item.properties.get("mlm:name") or item.id)


def build_knative_manifest(item: pystac.Item, namespace: str = DEFAULT_NAMESPACE) -> dict[str, Any]:
    inference_asset = item.assets.get("mlm:inference")
    if inference_asset is None:
        msg = f"Item '{item.id}' missing 'mlm:inference' asset"
        raise KeyError(msg)

    source_asset = item.assets.get("source-code")
    if source_asset is None:
        msg = f"Item '{item.id}' missing 'source-code' asset"
        raise KeyError(msg)
    entrypoint = source_asset.extra_fields.get("mlm:entrypoint")
    if not entrypoint:
        msg = f"Item '{item.id}' source-code asset missing 'mlm:entrypoint'"
        raise KeyError(msg)

    return {
        "apiVersion": f"{KNATIVE_GROUP}/{KNATIVE_VERSION}",
        "kind": "Service",
        "metadata": {
            "name": _service_name(item),
            "namespace": namespace,
        },
        "spec": {
            "template": {
                "metadata": {
                    "annotations": {
                        "autoscaling.knative.dev/min-scale": "0",
                        "autoscaling.knative.dev/max-scale": "5",
                        "autoscaling.knative.dev/scale-down-delay": "60s",
                    },
                },
                "spec": {
                    "containers": [
                        {
                            "image": inference_asset.href,
                            "ports": [{"containerPort": 8080}],
                            "env": [
                                {"name": "MODEL_MODULE", "value": _module_from_entrypoint(entrypoint)},
                            ],
                        }
                    ],
                },
            }
        },
    }


def _custom_objects_api() -> Any:
    from kubernetes import client, config

    try:
        config.load_incluster_config()
    except config.ConfigException:
        config.load_kube_config()
    return client.CustomObjectsApi()


def ensure_knative_service(item: pystac.Item, namespace: str = DEFAULT_NAMESPACE) -> None:
    from kubernetes.client.exceptions import ApiException

    manifest = build_knative_manifest(item, namespace=namespace)
    name = manifest["metadata"]["name"]
    api = _custom_objects_api()

    try:
        api.get_namespaced_custom_object(
            group=KNATIVE_GROUP,
            version=KNATIVE_VERSION,
            namespace=namespace,
            plural=KNATIVE_PLURAL,
            name=name,
        )
    except ApiException as exc:
        if exc.status != 404:
            raise
        api.create_namespaced_custom_object(
            group=KNATIVE_GROUP,
            version=KNATIVE_VERSION,
            namespace=namespace,
            plural=KNATIVE_PLURAL,
            body=manifest,
        )
        return

    api.patch_namespaced_custom_object(
        group=KNATIVE_GROUP,
        version=KNATIVE_VERSION,
        namespace=namespace,
        plural=KNATIVE_PLURAL,
        name=name,
        body=manifest,
    )


def delete_knative_service(model_name: str, namespace: str = DEFAULT_NAMESPACE) -> None:
    from kubernetes.client.exceptions import ApiException

    api = _custom_objects_api()
    name = knative_service_name(model_name)
    try:
        api.delete_namespaced_custom_object(
            group=KNATIVE_GROUP,
            version=KNATIVE_VERSION,
            namespace=namespace,
            plural=KNATIVE_PLURAL,
            name=name,
        )
    except ApiException as exc:
        if exc.status != 404:
            raise
