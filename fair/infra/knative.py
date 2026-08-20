"""Per-model KNative Service: manifest from a YAML template, applied via the k8s API."""

import os
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import pystac
import yaml

KNATIVE_GROUP = "serving.knative.dev"
KNATIVE_VERSION = "v1"
KNATIVE_PLURAL = "services"

DEFAULT_NAMESPACE = os.environ.get("FAIR_KNATIVE_NAMESPACE") or "predict"
# The ksvc shape lives in this YAML so resources/autoscaling/env retune without a code
# change; FAIR_KNATIVE_TEMPLATE can point at a mounted ConfigMap to override it live.
_DEFAULT_TEMPLATE = Path(__file__).with_name("knative-service.yaml")

# Route served by fair.serve.base.create_app.
HEALTH_PATH = "/health"
# Generous enough for a scale-to-zero cold start, which the activator holds.
DEFAULT_HEALTH_TIMEOUT = 60.0


class KnativeError(Exception):
    """Base for KNative provisioning and readiness failures."""


class KnativeNotInstalledError(KnativeError):
    """KNative Serving is not registered on the target cluster."""


class KnativeServiceUnavailableError(KnativeError):
    """A model's KNative service did not answer 200 on its health route."""


def _template_path(override: str | None = None) -> Path:
    return Path(override or os.environ.get("FAIR_KNATIVE_TEMPLATE") or _DEFAULT_TEMPLATE)


def knative_service_name(name: str) -> str:
    """Convert a model identifier to a DNS-1035 label accepted by KNative."""
    return str(name).lower().replace("_", "-")


def knative_service_host(name: str, namespace: str | None = None) -> str:
    ns = namespace if namespace is not None else DEFAULT_NAMESPACE
    return f"{knative_service_name(name)}.{ns}.svc.cluster.local"


def public_service_url(name: str, domain: str) -> str:
    # Must stay in lock-step with config-domain, domain-template, and the
    # wildcard Ingress; this is the only place the shape lives.
    return f"https://{knative_service_name(name)}.predict.{domain}"


def public_predict_url(name: str, domain: str) -> str:
    return f"{public_service_url(name, domain)}/predict"


def health_url(endpoint_href: str) -> str:
    """Map any endpoint URL on a service to that service's health route."""
    parts = urlsplit(endpoint_href)
    if not parts.scheme or not parts.netloc:
        msg = f"Endpoint href '{endpoint_href}' is not an absolute URL"
        raise ValueError(msg)
    return urlunsplit((parts.scheme, parts.netloc, HEALTH_PATH, "", ""))


def probe_service_health(
    url: str,
    *,
    timeout: float = DEFAULT_HEALTH_TIMEOUT,
    verify: bool = True,
) -> None:
    """Return once the service answers 200, raise otherwise."""
    import httpx

    try:
        response = httpx.get(url, timeout=timeout, verify=verify)
    except httpx.HTTPError as exc:
        msg = f"{url} is unreachable: {exc}"
        raise KnativeServiceUnavailableError(msg) from exc
    if response.status_code != 200:
        msg = f"{url} returned HTTP {response.status_code}, expected 200"
        raise KnativeServiceUnavailableError(msg)


def _module_from_entrypoint(entrypoint: str) -> str:
    if ":" not in entrypoint:
        msg = f"Invalid mlm:entrypoint '{entrypoint}', expected 'module.path:function'"
        raise ValueError(msg)
    return entrypoint.rsplit(":", 1)[0]


def _service_name(item: pystac.Item) -> str:
    return knative_service_name(item.properties.get("mlm:name") or item.id)


def _entrypoint(item: pystac.Item) -> str:
    source = item.assets.get("source-code")
    if source is None:
        msg = f"Item '{item.id}' missing 'source-code' asset"
        raise KeyError(msg)
    entrypoint = source.extra_fields.get("mlm:entrypoint")
    if not entrypoint:
        msg = f"Item '{item.id}' source-code asset missing 'mlm:entrypoint'"
        raise KeyError(msg)
    return entrypoint


def build_knative_manifest(
    item: pystac.Item,
    namespace: str | None = None,
    template_path: str | None = None,
) -> dict[str, Any]:
    """Render the ksvc manifest. The YAML template supplies the static shape; the STAC
    item supplies name, image, MODEL_MODULE, and any per-model resource/node overrides.
    """
    inference = item.assets.get("mlm:inference")
    if inference is None:
        msg = f"Item '{item.id}' missing 'mlm:inference' asset"
        raise KeyError(msg)
    entrypoint = _entrypoint(item)

    manifest = yaml.safe_load(_template_path(template_path).read_text())
    props = item.properties
    service_name = _service_name(item)
    manifest["metadata"]["name"] = service_name
    manifest["metadata"]["namespace"] = namespace or DEFAULT_NAMESPACE

    labels = manifest["metadata"].setdefault("labels", {})
    labels.setdefault("app.kubernetes.io/managed-by", "fair")
    labels["app.kubernetes.io/name"] = service_name
    if version := props.get("version"):
        labels["app.kubernetes.io/version"] = str(version)

    container = manifest["spec"]["template"]["spec"]["containers"][0]
    container["image"] = inference.href
    container.setdefault("env", []).insert(
        0, {"name": "MODEL_MODULE", "value": _module_from_entrypoint(entrypoint)}
    )

    resources = container.setdefault("resources", {})
    for section, key, prop in (
        ("requests", "cpu", "fair:cpu_request"),
        ("requests", "memory", "fair:memory_request"),
        ("limits", "cpu", "fair:cpu_limit"),
        ("limits", "memory", "fair:memory_limit"),
    ):
        if prop in props:
            resources.setdefault(section, {})[key] = str(props[prop])

    node_pool = props.get("fair:node_pool")
    if node_pool:
        selector_key = os.environ.get("FAIR_KNATIVE_NODE_SELECTOR_KEY")
        if selector_key:
            manifest["spec"]["template"]["spec"]["nodeSelector"] = {selector_key: str(node_pool)}
        else:
            print(f"skip node pool: FAIR_KNATIVE_NODE_SELECTOR_KEY unset; ignoring fair:node_pool '{node_pool}'")
    return manifest


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


def _wait_until_ready(api: Any, name: str, namespace: str, timeout: int) -> None:
    """Poll the ksvc's Ready condition until True; raise on a failed or timed-out rollout."""
    deadline = time.monotonic() + timeout
    while True:
        obj = api.get_namespaced_custom_object(
            group=KNATIVE_GROUP,
            version=KNATIVE_VERSION,
            namespace=namespace,
            plural=KNATIVE_PLURAL,
            name=name,
        )
        conditions = (obj.get("status") or {}).get("conditions") or []
        ready = next((c for c in conditions if c.get("type") == "Ready"), None)
        if ready and ready.get("status") == "True":
            return
        if ready and ready.get("status") == "False":
            raise KnativeServiceUnavailableError(
                f"knative service '{name}' failed to become ready: {ready.get('message')}"
            )
        if time.monotonic() >= deadline:
            raise KnativeServiceUnavailableError(f"knative service '{name}' not ready within {timeout}s")
        time.sleep(3)


def ensure_knative_service(
    item: pystac.Item,
    namespace: str | None = None,
    template_path: str | None = None,
) -> None:
    """Apply the ksvc (create or patch, like `kubectl apply`). Raises KnativeNotInstalledError
    when Knative Serving is absent. Set FAIR_KNATIVE_VERIFY_TIMEOUT>0 to block until Ready."""
    if not _knative_serving_installed():
        msg = f"{KNATIVE_GROUP}/{KNATIVE_VERSION} is not registered on the cluster; install KNative Serving first"
        raise KnativeNotInstalledError(msg)
    ns = namespace if namespace is not None else DEFAULT_NAMESPACE
    manifest = build_knative_manifest(item, namespace=ns, template_path=template_path)
    api = _custom_objects_api()
    _upsert_knative_service(api, manifest, ns)
    timeout = int(os.environ.get("FAIR_KNATIVE_VERIFY_TIMEOUT", "0"))
    if timeout > 0:
        _wait_until_ready(api, manifest["metadata"]["name"], ns, timeout)


def reconcile_knative_services(
    items: Iterable[pystac.Item],
    namespace: str | None = None,
    template_path: str | None = None,
) -> list[str]:
    applied: list[str] = []
    for item in items:
        ensure_knative_service(item, namespace=namespace, template_path=template_path)
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


def knative_service_status(model_name: str, namespace: str | None = None) -> tuple[str, str]:
    """(Ready condition, cluster-assigned URL) for a model's KNative service."""
    ns = namespace if namespace is not None else DEFAULT_NAMESPACE
    api = _custom_objects_api()
    service = api.get_namespaced_custom_object(
        group=KNATIVE_GROUP,
        version=KNATIVE_VERSION,
        namespace=ns,
        plural=KNATIVE_PLURAL,
        name=knative_service_name(model_name),
    )
    status = service.get("status", {})
    conditions = status.get("conditions", [])
    ready = next((c.get("status", "Unknown") for c in conditions if c.get("type") == "Ready"), "Unknown")
    return ready, status.get("url", "")


def delete_knative_service(model_name: str, namespace: str | None = None) -> None:
    from kubernetes.client.exceptions import ApiException

    ns = namespace if namespace is not None else DEFAULT_NAMESPACE
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
