---
icon: lucide/container
---

# Kubernetes Dev Stack

Local kind cluster mirroring the EKS deployment from `hotosm/k8s-infra`.

## Quickstart

!!! info "Prerequisites"

    [kind](https://kind.sigs.k8s.io/), kubectl, helm,
    [helmfile](https://helmfile.readthedocs.io/),
    [mc](https://min.io/docs/minio/linux/reference/minio-mc.html) (minio client),
    [colima](https://github.com/abiosoft/colima) (macOS) or Docker Engine (Linux).
    `just setup` in k8s mode checks all of these are on `$PATH` before proceeding.
    For GPU support: [nvkind](https://github.com/NVIDIA/nvkind), NVIDIA driver,
    [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
    See [GPU Support](#gpu-support-optional) below.

View source code of infra files for dev [infra/dev](https://github.com/hotosm/fAIr-models/tree/master/infra/dev)

```bash title="Setup and cluster lifecycle"
just k8s              # switch to k8s mode (sticky, one-time)
just setup            # install deps + k8s extras + verify CLI tools
cd infra/dev
just up               # creates cluster if missing, deploys infra, starts port-forwards
just status            # show cluster, pods, port-forward health
just down              # stop port-forwards (cluster stays for fast restart)
just tear              # destroy everything
```

```bash title="Run pipelines"
just example           # E2E with local orchestrator against k8s infra (from repo root)
just run-example-k8s   # E2E with k8s orchestrator (from infra/dev)
```

### Verifying results

!!! success "After `just example` or `just run-example-k8s` completes, inspect outputs at"

    | What | URL |
    |------|-----|
    | ZenML dashboard (pipelines, steps, artifacts) | <http://localhost:8080> (login: `default` / empty password) |
    | STAC collections (registered & promoted models) | <http://localhost:8082/collections> |
    | MLflow experiments (training metrics, model registry) | <http://localhost:5000> |
    | MinIO browser (raw S3 objects) | <http://localhost:9001> (login: `minioadmin` / `minioadmin`) |

### ZenML Stacks

`just up` registers two stacks:

=== ":lucide-laptop: dev (active)"

    | | |
    |---|---|
    | **Orchestrator** | `default` (local process) |
    | **S3 Endpoint** | `localhost:9000` |
    | **MLflow** | `localhost:5000` |
    | **Use** | Local runs via port-forward (`just example`) |

=== ":lucide-container: k8s"

    | | |
    |---|---|
    | **Orchestrator** | `k8s_orchestrator` |
    | **S3 Endpoint** | `minio.fair.svc:9000` |
    | **MLflow** | `mlflow.fair.svc:80` |
    | **Use** | In-cluster jobs (`just run-example-k8s`) |

## Architecture

All services run in namespace `fair` on a 3-node kind cluster (1 CP + 2 workers).

```text title="Cluster topology (namespace: fair)"
postgres (PG 17 + PostGIS)           zenml (ghcr.io/hotosm/zenml-postgres:0.93.3)
  DBs: zenml, fair_models, mlflow      Official Helm chart, OCI registry
        |                               |
        +--- stac-fastapi-pgstac        +--- mlflow (community-charts/mlflow)
        |    eoapi-k8s chart                 PG backend + S3 artifacts
        |                               |
        +--- minio (s3://fair-data, s3://mlflow, s3://zenml)
```

??? note "Port-forwards (managed by `just up` / `just down`)"

    | Service  | Local           | Cluster                     |
    |----------|-----------------|-----------------------------|
    | ZenML    | localhost:8080  | zenml.fair.svc:80           |
    | STAC API | localhost:8082  | stac-stac.fair.svc:8080     |
    | MinIO API | localhost:9000  | minio.fair.svc:9000         |
    | MinIO Console | localhost:9001  | minio-console.fair.svc:9001 |
    | MLflow   | localhost:5000  | mlflow.fair.svc:80          |
    | Postgres | localhost:5432  | postgres.fair.svc:5432      |

## GPU Support (optional)

Follow the [nvkind prerequisites and setup guide](https://github.com/NVIDIA/nvkind#prerequisites) to install the NVIDIA driver, nvidia-container-toolkit, and nvkind on your host. Once `nvkind` is on `$PATH`, `just up` handles the rest.

??? info "What `just up` does"

    `kind-config.yaml` labels workers as `inference` and `train`, with the train
    node getting `extraMounts` that signal GPU presence to nvkind. The cluster
    creation step runs nvkind (installs toolkit inside the node, configures containerd).
    The infra step creates the `nvidia` RuntimeClass, labels the GPU node, and
    deploys the device plugin.

!!! warning "Caveats"

    - `PatchProcDriverNvidia` may fail on non-MIG single-GPU hosts ; non-critical, the justfile tolerates it.
    - nvkind restarts containerd on the GPU node, briefly disrupting colocated pods.
    - Device plugin uses `--set deviceDiscoveryStrategy=nvml` (default `auto` fails inside kind).

## Configuration

### Label domain

Node labels and taints use the `fair.dev` prefix (hardcoded in all dev/CI config files).
For production (dok8s), the label domain comes from the `domain` OpenTofu variable in `infra/dok8s/terraform.tfvars` (exposed as the `fair_domain` output and consumed by the `infra/dok8s/justfile` recipes).

The runtime default in `fair/zenml/config.py` can be overridden via `FAIR_LABEL_DOMAIN` env var.

??? info "Where the label domain appears"

    - **`infra/dev/kind-config.yaml`** : node labels (`fair.dev/role`) and taints (`fair.dev/workload`)
    - **`infra/ci/kind-config.yaml`** : same, single-node CI variant
    - **`infra/dev/postgres/statefulset.yaml`** : nodeSelector `fair.dev/role: infra`
    - **`stacks/k8s.yaml`** / **`stacks/ci-k8s.yaml`** : pod `node_selectors` and `tolerations`
    - **`fair/zenml/config.py`** : reads `FAIR_LABEL_DOMAIN` at runtime (default `fair.dev`) for pipeline pod scheduling

??? abstract "Decisions"

    **kind over minikube/k3s** : `hotosm/k8s-infra` runs upstream K8s (EKS). kind runs
    upstream K8s in Docker containers with guaranteed API compatibility. Lightweight, no VM. ( this can be revised in know that talos is recommended in our docs, it is mainly becuase of learning curve with talos..)

    **Single PostgreSQL, three databases** : ZenML, pgstac, and MLflow all need Postgres.
    One StatefulSet with init SQL (`CREATE DATABASE zenml; fair_models; mlflow`). Mirrors
    production where CloudNativePG hosts databases the same way.

    **MLflow over W&B** : Apache 2.0, uses Postgres (same engine as everything else),
    mature Helm chart, ZenML first-class `--flavor=mlflow` support. W&B self-hosted
    requires MySQL + Redis + commercial license.

    **eoAPI for STAC** : Dev uses the upstream eoAPI chart from
    `https://devseed.com/eoapi-k8s/` (`eoapi/eoapi`, pinned to `0.12.2` in
    `infra/dev/helmfile.yaml`) with `external-plaintext` DB.

    **ZenML Postgres patch** : OSS ZenML only supports MySQL/SQLite. The patched server
    image at [`ghcr.io/hotosm/zenml-postgres`](https://github.com/hotosm/fAIr/tree/develop/infra/zenml)
    replaces MySQL dialect (`MEDIUMTEXT`) with Postgres equivalents. The client side is
    handled automatically by `fair-py-ops`: a `.pth` startup hook
    (`fair/_patch_zenml.py`) adds the `POSTGRESQL` enum variant to
    `ServerDatabaseType` at interpreter startup, before any ZenML import. No manual
    client patching is needed.

    **StacBackend Protocol** : `StacCatalogManager` writes local JSON files.
    `PgStacBackend` writes to pgstac via pypgstac. Both conform to the `StacBackend`
    Protocol (structural subtyping). `run.py --stac-api-url` selects pgstac; omit for local.

    **PgStacBackend reads via pystac-client** : The eoAPI chart injects
    `--root-path=/stac` by default, which breaks self-links under direct port-forwarding.
    Dev values set `stac.overrideRootPath: ""` to remove it, so pystac-client works
    correctly against `http://localhost:8082`.

    **GPU scheduling from STAC metadata** : `mlm:accelerator` and `mlm:accelerator_count`
    in `stac-item.json` drive `nvidia.com/gpu` resource requests. `config.py` reads these
    and emits pod settings only when the orchestrator is Kubernetes.

## Known issues

!!! bug "eoAPI root_path (resolved)"

    The chart's
    [deployment template](https://github.com/developmentseed/eoapi-k8s/blob/main/charts/eoapi/templates/services/stac/deployment.yaml)
    injects `--root-path={{ .Values.stac.ingress.path }}` (defaults to `/stac`) into
    the uvicorn command when an ingress class is set. Dev values set
    `stac.overrideRootPath: ""` which removes the arg entirely, so pystac-client
    works via direct port-forwarding.

## References

!!! quote "Further reading"

    - [ZenML Postgres patch](https://github.com/hotosm/fAIr/tree/develop/infra/zenml)
    - [ZenML K8s Orchestrator](https://docs.zenml.io/stack-components/orchestrators/kubernetes)
    - [pypgstac](https://stac-utils.github.io/pgstac/pypgstac/)
    - [eoAPI-k8s](https://github.com/developmentseed/eoapi-k8s)

- [community-charts/mlflow](https://github.com/community-charts/helm-charts/tree/main/charts/mlflow)
- [kind](https://kind.sigs.k8s.io/)
