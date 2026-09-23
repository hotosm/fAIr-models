"""Per-model KNative Service: manifest from a YAML template, applied via the k8s API."""

import hashlib
import json
import os
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx
import pystac
import yaml

KNATIVE_GROUP = "serving.knative.dev"
KNATIVE_VERSION = "v1"
KNATIVE_PLURAL = "services"
KNATIVE_REVISIONS = "revisions"
MANAGED_BY_SELECTOR = "app.kubernetes.io/managed-by=fair"
# Each deployment has an owner label. Services are deleted after their last owner leaves.
OWNER_LABEL_PREFIX = "fair.hotosm.org/"
LIVE_OWNER = "live"
TEMPLATE_HASH_ANNOTATION = "fair.hotosm.org/template-hash"

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


def knative_tag() -> str | None:
    """Return this deployment's traffic tag, or None for the live route."""
    return os.environ.get("FAIR_KNATIVE_TAG") or None


def public_service_url(name: str, domain: str, tag: str | None = None) -> str:
    # Keep this aligned with Knative's domain and tag templates and the wildcard ingress.
    host = knative_service_name(name)
    if tag:
        host = f"{tag}-{host}"
    return f"https://{host}.predict.{domain}"


def public_predict_url(name: str, domain: str, tag: str | None = None) -> str:
    return f"{public_service_url(name, domain, tag)}/predict"


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
    container.setdefault("env", []).insert(0, {"name": "MODEL_MODULE", "value": _module_from_entrypoint(entrypoint)})

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

    if "fair:min_scale" in props:
        annotations = manifest["spec"]["template"]["metadata"].setdefault("annotations", {})
        annotations["autoscaling.knative.dev/min-scale"] = str(props["fair:min_scale"])
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


def _owner_label(tag: str | None) -> str:
    return f"{OWNER_LABEL_PREFIX}{tag or LIVE_OWNER}"


def _template_hash(template: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(template, sort_keys=True).encode()).hexdigest()[:16]


def _get_service(api: Any, name: str, namespace: str) -> dict[str, Any] | None:
    from kubernetes.client.exceptions import ApiException

    try:
        return api.get_namespaced_custom_object(
            group=KNATIVE_GROUP, version=KNATIVE_VERSION, namespace=namespace, plural=KNATIVE_PLURAL, name=name
        )
    except ApiException as exc:
        if exc.status == 404:
            return None
        raise


def _revision_with_hash(api: Any, name: str, namespace: str, digest: str) -> str | None:
    revisions = api.list_namespaced_custom_object(
        group=KNATIVE_GROUP,
        version=KNATIVE_VERSION,
        namespace=namespace,
        plural=KNATIVE_REVISIONS,
        label_selector=f"serving.knative.dev/service={name}",
    )
    for revision in revisions.get("items", []):
        metadata = revision.get("metadata") or {}
        if (metadata.get("annotations") or {}).get(TEMPLATE_HASH_ANNOTATION) == digest:
            return metadata.get("name")
    return None


def _route_traffic(service: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Return routes pinned to revisions so updates cannot move them."""
    routes: list[dict[str, Any]] = []
    for target in ((service or {}).get("status") or {}).get("traffic") or []:
        if not target.get("revisionName"):
            continue
        route: dict[str, Any] = {"revisionName": target["revisionName"], "percent": int(target.get("percent") or 0)}
        if target.get("tag"):
            route["tag"] = target["tag"]
        routes.append(route)
    return routes


def _desired_traffic(current: list[dict[str, Any]], ours: dict[str, Any], tag: str | None) -> list[dict[str, Any]]:
    """Replace only this deployment's live or tagged route."""
    if tag:
        others = [t for t in current if t.get("tag") != tag and (t.get("tag") or t["percent"])]
        if not any(t["percent"] for t in others):
            # Serve the first revision until a live route is registered.
            others = [{**ours, "percent": 100}, *({**t, "percent": 0} for t in others)]
        return [*others, {**ours, "tag": tag, "percent": 0}]
    return [{**ours, "percent": 100}, *({**t, "percent": 0} for t in current if t.get("tag"))]


def _patch_service(api: Any, name: str, namespace: str, body: dict[str, Any]) -> None:
    api.patch_namespaced_custom_object(
        group=KNATIVE_GROUP, version=KNATIVE_VERSION, namespace=namespace, plural=KNATIVE_PLURAL, name=name, body=body
    )


def ensure_knative_service(
    item: pystac.Item,
    namespace: str | None = None,
    template_path: str | None = None,
) -> None:
    """Create or update a service without changing other deployments' routes.

    Matching revisions are reused. Set FAIR_KNATIVE_VERIFY_TIMEOUT above zero to wait
    until ready. Raises KnativeNotInstalledError when Knative Serving is unavailable.
    """
    if not _knative_serving_installed():
        msg = f"{KNATIVE_GROUP}/{KNATIVE_VERSION} is not registered on the cluster; install KNative Serving first"
        raise KnativeNotInstalledError(msg)
    ns = namespace if namespace is not None else DEFAULT_NAMESPACE
    tag = knative_tag()
    manifest = build_knative_manifest(item, namespace=ns, template_path=template_path)
    name = manifest["metadata"]["name"]
    digest = _template_hash(manifest["spec"]["template"])
    manifest["spec"]["template"]["metadata"].setdefault("annotations", {})[TEMPLATE_HASH_ANNOTATION] = digest
    manifest["metadata"]["labels"][_owner_label(tag)] = "true"

    api = _custom_objects_api()
    existing = _get_service(api, name, ns)
    revision = _revision_with_hash(api, name, ns, digest) if existing else None
    if revision:
        # Keep other deployments on their existing revision.
        del manifest["spec"]["template"]
    ours: dict[str, Any] = {"revisionName": revision} if revision else {"latestRevision": True}
    manifest["spec"]["traffic"] = _desired_traffic(_route_traffic(existing), ours, tag)
    _upsert_knative_service(api, manifest, ns)

    timeout = int(os.environ.get("FAIR_KNATIVE_VERIFY_TIMEOUT", "0"))
    if timeout > 0:
        _wait_until_ready(api, name, ns, timeout)
        if not revision:
            # Pin the route so later template changes cannot move it.
            service = _get_service(api, name, ns) or {}
            latest = (service.get("status") or {}).get("latestReadyRevisionName")
            if latest:
                pinned = _desired_traffic(_route_traffic(service), {"revisionName": latest}, tag)
                _patch_service(api, name, ns, {"spec": {"traffic": pinned}})


def release_knative_service(model_name: str, namespace: str | None = None) -> bool:
    """Release this deployment's route and delete the service if no owners remain.

    Unlabelled services are treated as live. Returns True when anything is released.
    """
    ns = namespace if namespace is not None else DEFAULT_NAMESPACE
    tag = knative_tag()
    mine = _owner_label(tag)
    name = knative_service_name(model_name)
    api = _custom_objects_api()
    service = _get_service(api, name, ns)
    if service is None:
        return False
    labels = (service.get("metadata") or {}).get("labels") or {}
    owners = {k for k, v in labels.items() if k.startswith(OWNER_LABEL_PREFIX) and v == "true"}
    if mine not in owners and (tag or owners):
        return False
    if not owners - {mine}:
        delete_knative_service(name, namespace=ns)
        return True
    current = _route_traffic(service)
    if tag:
        traffic = [t for t in current if t.get("tag") != tag]
    else:
        tagged = [t for t in current if t.get("tag")]
        traffic = [{**tagged[0], "percent": 100}, *tagged[1:]] if tagged else current
    body: dict[str, Any] = {"metadata": {"labels": {mine: None}}}
    if traffic:
        body["spec"] = {"traffic": traffic}
    _patch_service(api, name, ns, body)
    return True


def reconcile_knative_services(
    items: Iterable[pystac.Item],
    namespace: str | None = None,
    template_path: str | None = None,
    prune: bool = False,
) -> dict[str, list[str]]:
    """Match Knative services to the active base models.

    When pruning, release services absent from `items`. An empty item list never prunes.
    """
    if not _knative_serving_installed():
        msg = f"{KNATIVE_GROUP}/{KNATIVE_VERSION} is not registered on the cluster; install KNative Serving first"
        raise KnativeNotInstalledError(msg)
    ns = namespace if namespace is not None else DEFAULT_NAMESPACE
    result: dict[str, list[str]] = {"applied": [], "removed": [], "failed": []}
    wanted: set[str] = set()
    for item in items:
        name = _service_name(item)
        wanted.add(name)
        try:
            ensure_knative_service(item, namespace=ns, template_path=template_path)
            result["applied"].append(name)
        except Exception as exc:  # Continue if one item fails.
            result["failed"].append(f"{name}: {exc}")
    if prune and not wanted:
        print("skip prune: no active base models found")
    elif prune:
        services = _custom_objects_api().list_namespaced_custom_object(
            group=KNATIVE_GROUP,
            version=KNATIVE_VERSION,
            namespace=ns,
            plural=KNATIVE_PLURAL,
            label_selector=MANAGED_BY_SELECTOR,
        )
        for service in services.get("items", []):
            name = service["metadata"]["name"]
            if name not in wanted and release_knative_service(name, namespace=ns):
                result["removed"].append(name)
    return result


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
