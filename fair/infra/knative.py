"""Dynamic KNative service management for live model serving.

KNative operator, Kourier, and the ingress bridge are provisioned once by
Terraform (infra/dok8s/services.tf). Per-model KNative services are created
on demand by `fair.client.register_base_model` via `ensure_knative_service`.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from typing import Any

import pystac

KNATIVE_GROUP = "serving.knative.dev"
KNATIVE_VERSION = "v1"
KNATIVE_PLURAL = "services"
DEFAULT_NAMESPACE = "predict"
PUBLIC_INGRESS_NAMESPACE = "knative-serving"
S3_CREDENTIALS_SECRET = "s3-credentials"
PREDICT_GATEWAY_NAME = "predict-gateway"
PREDICT_GATEWAY_LOCATION_TEMPLATE = """
    location ~ ^/{service_name}(/|$)(.*)$ {{
        rewrite ^/{service_name}(/|$)(.*)$ /$2 break;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_set_header Host {upstream_host};
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_pass http://{upstream_host};
    }}
""".strip("\n")
PREDICT_GATEWAY_MODEL_LINK_TEMPLATE = (
    "<li><strong>{service_name}</strong> "
    '· <a href="/{service_name}/health">health</a> '
    '· <a href="/{service_name}/predict">predict</a></li>'
)
PREDICT_GATEWAY_HOME_TEMPLATE = (
    '<!doctype html><html><head><meta charset="utf-8"><title>fAIr live models</title></head>'
    "<body><h1>Available models</h1><ul>{model_links}</ul></body></html>"
)
PREDICT_GATEWAY_SERVER_TEMPLATE = """
server {{
    listen 8080;
    client_max_body_size 512m;

{locations}
    location = /health {{
        default_type text/plain;
        return 200 'ok';
    }}

    location = /models {{
        default_type application/json;
        return 200 '{models_json}';
    }}

    location = / {{
        default_type text/html;
        return 200 '{homepage_html}';
    }}

    location / {{
        return 404;
    }}
}}
""".strip()


def knative_service_name(name: str) -> str:
    """Convert a model identifier to a DNS-1035 label accepted by KNative."""
    return str(name).lower().replace("_", "-")


def knative_service_host(name: str, namespace: str = DEFAULT_NAMESPACE) -> str:
    service_name = knative_service_name(name)
    return f"{service_name}.{namespace}.svc.cluster.local"


def _module_from_entrypoint(entrypoint: str) -> str:
    if ":" not in entrypoint:
        msg = f"Invalid mlm:entrypoint '{entrypoint}', expected 'module.path:function'"
        raise ValueError(msg)
    return entrypoint.rsplit(":", 1)[0]


def _service_name(item: pystac.Item) -> str:
    return knative_service_name(item.properties.get("mlm:name") or item.id)


def _build_predict_gateway_homepage(service_names: list[str]) -> str:
    unique_service_names = sorted(set(service_names))
    if not unique_service_names:
        return PREDICT_GATEWAY_HOME_TEMPLATE.format(model_links="<li>No models are currently registered.</li>")

    model_links = "".join(
        PREDICT_GATEWAY_MODEL_LINK_TEMPLATE.format(service_name=service_name) for service_name in unique_service_names
    )
    return PREDICT_GATEWAY_HOME_TEMPLATE.format(model_links=model_links)


def _build_predict_gateway_models_json(service_names: list[str]) -> str:
    models = [
        {
            "name": service_name,
            "health": f"/{service_name}/health",
            "predict": f"/{service_name}/predict",
        }
        for service_name in sorted(set(service_names))
    ]
    return json.dumps({"models": models})


def build_predict_gateway_config(service_names: list[str], service_namespace: str = DEFAULT_NAMESPACE) -> str:
    unique_service_names = sorted(set(service_names))
    locations = []
    for service_name in unique_service_names:
        upstream_host = knative_service_host(service_name, service_namespace)
        locations.append(
            PREDICT_GATEWAY_LOCATION_TEMPLATE.format(
                service_name=service_name,
                upstream_host=upstream_host,
            )
        )

    joined_locations = "\n\n".join(locations)
    homepage_html = _build_predict_gateway_homepage(unique_service_names)
    models_json = _build_predict_gateway_models_json(unique_service_names)
    return PREDICT_GATEWAY_SERVER_TEMPLATE.format(
        locations=joined_locations,
        homepage_html=homepage_html,
        models_json=models_json,
    )


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
                            "envFrom": [
                                {"secretRef": {"name": S3_CREDENTIALS_SECRET}},
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


def _upsert_resource(
    *,
    read: Callable[[], Any],
    create: Callable[[], Any],
    patch: Callable[[], Any],
) -> None:
    from kubernetes.client.exceptions import ApiException

    try:
        read()
    except ApiException as exc:
        if exc.status != 404:
            raise
        create()
        return

    patch()


def _upsert_knative_service(api: Any, manifest: dict[str, Any], namespace: str) -> None:
    name = manifest["metadata"]["name"]
    _upsert_resource(
        read=lambda: api.get_namespaced_custom_object(
            group=KNATIVE_GROUP,
            version=KNATIVE_VERSION,
            namespace=namespace,
            plural=KNATIVE_PLURAL,
            name=name,
        ),
        create=lambda: api.create_namespaced_custom_object(
            group=KNATIVE_GROUP,
            version=KNATIVE_VERSION,
            namespace=namespace,
            plural=KNATIVE_PLURAL,
            body=manifest,
        ),
        patch=lambda: api.patch_namespaced_custom_object(
            group=KNATIVE_GROUP,
            version=KNATIVE_VERSION,
            namespace=namespace,
            plural=KNATIVE_PLURAL,
            name=name,
            body=manifest,
        ),
    )


def _list_knative_service_names(api: Any, namespace: str) -> list[str]:
    items = api.list_namespaced_custom_object(
        group=KNATIVE_GROUP,
        version=KNATIVE_VERSION,
        namespace=namespace,
        plural=KNATIVE_PLURAL,
    ).get("items", [])
    return sorted(entry["metadata"]["name"] for entry in items)


def _gateway_labels() -> dict[str, str]:
    return {"app": PREDICT_GATEWAY_NAME}


def _build_gateway_config_map(nginx_config: str) -> Any:
    from kubernetes import client

    return client.V1ConfigMap(
        metadata=client.V1ObjectMeta(
            name=f"{PREDICT_GATEWAY_NAME}-config",
            namespace=PUBLIC_INGRESS_NAMESPACE,
        ),
        data={"default.conf": nginx_config},
    )


def _build_gateway_deployment() -> Any:
    from kubernetes import client

    labels = _gateway_labels()
    return client.V1Deployment(
        metadata=client.V1ObjectMeta(name=PREDICT_GATEWAY_NAME, namespace=PUBLIC_INGRESS_NAMESPACE),
        spec=client.V1DeploymentSpec(
            replicas=1,
            selector=client.V1LabelSelector(match_labels=labels),
            template=client.V1PodTemplateSpec(
                metadata=client.V1ObjectMeta(labels=labels),
                spec=client.V1PodSpec(
                    containers=[
                        client.V1Container(
                            name=PREDICT_GATEWAY_NAME,
                            image="nginx:1.27-alpine",
                            ports=[client.V1ContainerPort(container_port=8080)],
                            volume_mounts=[
                                client.V1VolumeMount(
                                    name="gateway-config",
                                    mount_path="/etc/nginx/conf.d/default.conf",
                                    sub_path="default.conf",
                                )
                            ],
                        )
                    ],
                    volumes=[
                        client.V1Volume(
                            name="gateway-config",
                            config_map=client.V1ConfigMapVolumeSource(name=f"{PREDICT_GATEWAY_NAME}-config"),
                        )
                    ],
                ),
            ),
        ),
    )


def _build_gateway_service() -> Any:
    from kubernetes import client

    return client.V1Service(
        metadata=client.V1ObjectMeta(name=PREDICT_GATEWAY_NAME, namespace=PUBLIC_INGRESS_NAMESPACE),
        spec=client.V1ServiceSpec(
            selector=_gateway_labels(),
            ports=[client.V1ServicePort(port=80, target_port=8080)],
        ),
    )


def _delete_legacy_public_ingresses(service_names: list[str]) -> None:
    from kubernetes import client
    from kubernetes.client.exceptions import ApiException

    networking_api = client.NetworkingV1Api()
    for ingress_name in [f"{service_name}-public" for service_name in service_names]:
        try:
            networking_api.delete_namespaced_ingress(ingress_name, PUBLIC_INGRESS_NAMESPACE)
        except ApiException as exc:
            if exc.status != 404:
                raise


def _ensure_predict_gateway(api: Any, namespace: str) -> None:
    from kubernetes import client

    core_api = client.CoreV1Api()
    apps_api = client.AppsV1Api()

    service_names = _list_knative_service_names(api, namespace)
    config_map_name = f"{PREDICT_GATEWAY_NAME}-config"
    nginx_config = build_predict_gateway_config(service_names, service_namespace=namespace)

    config_map_body = _build_gateway_config_map(nginx_config)
    deployment_body = _build_gateway_deployment()
    service_body = _build_gateway_service()

    _upsert_resource(
        read=lambda: core_api.read_namespaced_config_map(config_map_name, PUBLIC_INGRESS_NAMESPACE),
        create=lambda: core_api.create_namespaced_config_map(PUBLIC_INGRESS_NAMESPACE, config_map_body),
        patch=lambda: core_api.patch_namespaced_config_map(config_map_name, PUBLIC_INGRESS_NAMESPACE, config_map_body),
    )
    _upsert_resource(
        read=lambda: apps_api.read_namespaced_deployment(
            PREDICT_GATEWAY_NAME,
            PUBLIC_INGRESS_NAMESPACE,
        ),
        create=lambda: apps_api.create_namespaced_deployment(
            PUBLIC_INGRESS_NAMESPACE,
            deployment_body,
        ),
        patch=lambda: apps_api.patch_namespaced_deployment(
            PREDICT_GATEWAY_NAME,
            PUBLIC_INGRESS_NAMESPACE,
            deployment_body,
        ),
    )
    _upsert_resource(
        read=lambda: core_api.read_namespaced_service(PREDICT_GATEWAY_NAME, PUBLIC_INGRESS_NAMESPACE),
        create=lambda: core_api.create_namespaced_service(PUBLIC_INGRESS_NAMESPACE, service_body),
        patch=lambda: core_api.patch_namespaced_service(PREDICT_GATEWAY_NAME, PUBLIC_INGRESS_NAMESPACE, service_body),
    )
    _delete_legacy_public_ingresses(service_names)


def ensure_knative_service(item: pystac.Item, namespace: str = DEFAULT_NAMESPACE) -> None:
    manifest = build_knative_manifest(item, namespace=namespace)
    api = _custom_objects_api()

    _upsert_knative_service(api, manifest, namespace)

    if os.environ.get("FAIR_LABEL_DOMAIN"):
        _ensure_predict_gateway(api, namespace)


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
