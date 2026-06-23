"""Per-model KNative Service: single config object, env-overridable."""

import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from typing import Any

import pystac

KNATIVE_GROUP = "serving.knative.dev"
KNATIVE_VERSION = "v1"
KNATIVE_PLURAL = "services"


@dataclass(frozen=True)
class KnativeConfig:
    namespace: str = "predict"
    container_port: int = 8080
    s3_secret_name: str = "s3-credentials"
    min_scale: str = "0"
    max_scale: str = "5"
    scale_down_delay: str = "60s"
    # Hard per-pod request cap. 0 = unlimited (Knative default).
    container_concurrency: int = 1
    # Soft autoscaler target (in-flight reqs per pod). 0 omits the annotation
    # and lets Knative derive it from containerConcurrency * 70%.
    target_concurrency: int = 0
    cpu_request: str = "500m"
    cpu_limit: str = "2"
    memory_request: str = "1Gi"
    memory_limit: str = "3Gi"
    model_module_env: str = "MODEL_MODULE"
    onnx_intra_threads: int = 0  # 0 = ORT default (no cap)
    cors_origins: str = "*"
    cors_methods: str = "*"
    cors_headers: str = "*"

    @classmethod
    def from_env(cls) -> "KnativeConfig":
        env = os.environ.get
        return cls(
            namespace=env("FAIR_KNATIVE_NAMESPACE") or cls.namespace,
            container_port=int(env("FAIR_KNATIVE_CONTAINER_PORT") or cls.container_port),
            s3_secret_name=env("FAIR_KNATIVE_S3_SECRET") or cls.s3_secret_name,
            min_scale=env("FAIR_KNATIVE_MIN_SCALE") or cls.min_scale,
            max_scale=env("FAIR_KNATIVE_MAX_SCALE") or cls.max_scale,
            scale_down_delay=env("FAIR_KNATIVE_SCALE_DOWN_DELAY") or cls.scale_down_delay,
            container_concurrency=int(env("FAIR_KNATIVE_CONTAINER_CONCURRENCY") or cls.container_concurrency),
            target_concurrency=int(env("FAIR_KNATIVE_TARGET_CONCURRENCY") or cls.target_concurrency),
            cpu_request=env("FAIR_KNATIVE_CPU_REQUEST") or cls.cpu_request,
            cpu_limit=env("FAIR_KNATIVE_CPU_LIMIT") or cls.cpu_limit,
            memory_request=env("FAIR_KNATIVE_MEMORY_REQUEST") or cls.memory_request,
            memory_limit=env("FAIR_KNATIVE_MEMORY_LIMIT") or cls.memory_limit,
            model_module_env=env("FAIR_KNATIVE_MODEL_MODULE_ENV") or cls.model_module_env,
            onnx_intra_threads=int(env("FAIR_KNATIVE_ONNX_THREADS") or cls.onnx_intra_threads),
            cors_origins=env("FAIR_KNATIVE_CORS_ORIGINS") or cls.cors_origins,
            cors_methods=env("FAIR_KNATIVE_CORS_METHODS") or cls.cors_methods,
            cors_headers=env("FAIR_KNATIVE_CORS_HEADERS") or cls.cors_headers,
        )


KNATIVE_CONFIG: KnativeConfig = KnativeConfig.from_env()

DEFAULT_NAMESPACE = KNATIVE_CONFIG.namespace
S3_CREDENTIALS_SECRET = KNATIVE_CONFIG.s3_secret_name


def knative_service_name(name: str) -> str:
    """Convert a model identifier to a DNS-1035 label accepted by KNative."""
    return str(name).lower().replace("_", "-")


def knative_service_host(name: str, namespace: str | None = None) -> str:
    ns = namespace if namespace is not None else KNATIVE_CONFIG.namespace
    return f"{knative_service_name(name)}.{ns}.svc.cluster.local"


def public_predict_url(name: str, domain: str) -> str:
    # Must stay in lock-step with config-domain, domain-template, and the
    # wildcard Ingress; this is the only place the shape lives.
    return f"https://{knative_service_name(name)}.predict.{domain}/predict"


def _module_from_entrypoint(entrypoint: str) -> str:
    if ":" not in entrypoint:
        msg = f"Invalid mlm:entrypoint '{entrypoint}', expected 'module.path:function'"
        raise ValueError(msg)
    return entrypoint.rsplit(":", 1)[0]


def _service_name(item: pystac.Item) -> str:
    return knative_service_name(item.properties.get("mlm:name") or item.id)


def _container_env(cfg: KnativeConfig, entrypoint: str) -> list[dict[str, str]]:
    env = [
        {"name": cfg.model_module_env, "value": _module_from_entrypoint(entrypoint)},
        {"name": "FAIR_KNATIVE_ONNX_THREADS", "value": str(cfg.onnx_intra_threads)},
        {"name": "FAIR_KNATIVE_CORS_ORIGINS", "value": cfg.cors_origins},
        {"name": "FAIR_KNATIVE_CORS_METHODS", "value": cfg.cors_methods},
        {"name": "FAIR_KNATIVE_CORS_HEADERS", "value": cfg.cors_headers},
    ]
    # Match OpenMP to the ONNX cap so the numpy postprocess doesn't oversubscribe CPU.
    if cfg.onnx_intra_threads > 0:
        env.append({"name": "OMP_NUM_THREADS", "value": str(cfg.onnx_intra_threads)})
    return env


def build_knative_manifest(
    item: pystac.Item,
    namespace: str | None = None,
    config: KnativeConfig | None = None,
) -> dict[str, Any]:
    cfg = config or KNATIVE_CONFIG
    ns = namespace if namespace is not None else cfg.namespace

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

    props = item.properties
    # Per-model resources come from the STAC item so a re-register reproduces the sized pod.
    effective_cfg = replace(
        cfg,
        cpu_request=str(props.get("fair:cpu_request", cfg.cpu_request)),
        cpu_limit=str(props.get("fair:cpu_limit", cfg.cpu_limit)),
        memory_request=str(props.get("fair:memory_request", cfg.memory_request)),
        memory_limit=str(props.get("fair:memory_limit", cfg.memory_limit)),
    )

    annotations = {
        "autoscaling.knative.dev/min-scale": effective_cfg.min_scale,
        "autoscaling.knative.dev/max-scale": effective_cfg.max_scale,
        "autoscaling.knative.dev/scale-down-delay": effective_cfg.scale_down_delay,
    }
    if effective_cfg.target_concurrency > 0:
        annotations["autoscaling.knative.dev/target"] = str(effective_cfg.target_concurrency)

    pod_spec: dict[str, Any] = {
        "containerConcurrency": effective_cfg.container_concurrency,
        "containers": [
            {
                "image": inference_asset.href,
                "ports": [{"containerPort": effective_cfg.container_port}],
                "env": _container_env(effective_cfg, entrypoint),
                "envFrom": [
                    {"secretRef": {"name": effective_cfg.s3_secret_name}},
                ],
                "resources": {
                    "requests": {
                        "cpu": effective_cfg.cpu_request,
                        "memory": effective_cfg.memory_request,
                    },
                    "limits": {
                        "cpu": effective_cfg.cpu_limit,
                        "memory": effective_cfg.memory_limit,
                    },
                },
            }
        ],
    }

    # Pin to the model's declared node pool so it lands on its sized hardware.
    node_pool = props.get("fair:node_pool")
    if node_pool:
        pod_spec["nodeSelector"] = {"doks.digitalocean.com/node-pool": str(node_pool)}

    return {
        "apiVersion": f"{KNATIVE_GROUP}/{KNATIVE_VERSION}",
        "kind": "Service",
        "metadata": {
            "name": _service_name(item),
            "namespace": ns,
        },
        "spec": {
            "template": {
                "metadata": {
                    "annotations": annotations,
                },
                "spec": pod_spec,
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


def ensure_knative_service(
    item: pystac.Item,
    namespace: str | None = None,
    config: KnativeConfig | None = None,
) -> None:
    if not _knative_serving_installed():
        print(f"skip knative: {KNATIVE_GROUP}/{KNATIVE_VERSION} not registered on cluster")
        return
    cfg = config or KNATIVE_CONFIG
    ns = namespace if namespace is not None else cfg.namespace
    manifest = build_knative_manifest(item, namespace=ns, config=cfg)
    api = _custom_objects_api()

    _upsert_knative_service(api, manifest, ns)


def reconcile_knative_services(
    items: Iterable[pystac.Item],
    namespace: str | None = None,
    config: KnativeConfig | None = None,
) -> list[str]:
    applied: list[str] = []
    for item in items:
        ensure_knative_service(item, namespace=namespace, config=config)
        applied.append(item.id)
    return applied


def _knative_serving_installed() -> bool:
    from kubernetes import client, config
    from kubernetes.client.exceptions import ApiException

    try:
        try:
            config.load_incluster_config()
        except config.ConfigException:
            config.load_kube_config()
        groups = client.ApisApi().get_api_versions().groups
    except (config.ConfigException, ApiException):
        return False
    return any(g.name == KNATIVE_GROUP for g in groups)


def delete_knative_service(model_name: str, namespace: str | None = None) -> None:
    from kubernetes.client.exceptions import ApiException

    ns = namespace if namespace is not None else KNATIVE_CONFIG.namespace
    api = _custom_objects_api()
    name = knative_service_name(model_name)
    try:
        api.delete_namespaced_custom_object(
            group=KNATIVE_GROUP,
            version=KNATIVE_VERSION,
            namespace=ns,
            plural=KNATIVE_PLURAL,
            name=name,
        )
    except ApiException as exc:
        if exc.status != 404:
            raise
