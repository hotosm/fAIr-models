# Kubernetes Dev Stack

Local kind cluster mirroring the production EKS deployment from `hotosm/k8s-infra`.
Validates the full pipeline lifecycle (register -> train -> promote -> predict) against
the same infrastructure surface as production before code is merged.

## Quickstart

Prerequisites: [kind](https://kind.sigs.k8s.io/), kubectl, helm, [mc](https://min.io/docs/minio/linux/reference/minio-mc.html) (minio client), [colima](https://github.com/abiosoft/colima) (macOS) or Docker Engine (Linux).

```bash
uv sync --group example --group k8s
cd infra/dev
make up            # cluster + infra + port-forwards + seed data + zenml stack
make run-example   # init -> register -> finetune -> promote -> predict
make teardown      # destroy everything
```

Individual targets: `make help`.

## Architecture

All services run in namespace `fair` on a 3-node kind cluster (1 CP + 2 workers).

```
postgres (PG 17 + PostGIS)           zenml (ghcr.io/hotosm/zenml-postgres:0.93.3)
  DBs: zenml, fair_models, mlflow      Official Helm chart, OCI registry
        |                               |
        +--- stac-fastapi-pgstac        +--- mlflow (community-charts/mlflow)
        |    eoapi-k8s chart                 PG backend + S3 artifacts
        |                               |
        +--- minio (s3://fair-data, s3://mlflow, s3://zenml)
```

Port-forwards (via `make port-forward`):

| Service  | Local           | Cluster                     |
|----------|-----------------|-----------------------------|
| ZenML    | localhost:8080  | zenml.fair.svc:80           |
| STAC API | localhost:8082  | stac-stac.fair.svc:8080     |
| MinIO    | localhost:9000  | minio.fair.svc:9000         |
| MLflow   | localhost:5000  | mlflow.fair.svc:80          |
| Postgres | localhost:5432  | postgres.fair.svc:5432      |

## Decisions

**kind over minikube/k3s** -- `hotosm/k8s-infra` runs upstream K8s (EKS). kind runs
upstream K8s in Docker containers with guaranteed API compatibility. Lightweight, no VM. ( this can be revised in know that tailos is recommended in our docs, i wanted something up and running fast)

**Single PostgreSQL, three databases** -- ZenML, pgstac, and MLflow all need Postgres.
One StatefulSet with init SQL (`CREATE DATABASE zenml; fair_models; mlflow`). Mirrors
production where CloudNativePG hosts databases the same way.

**MLflow over W&B** -- Apache 2.0, uses Postgres (same engine as everything else),
mature Helm chart, ZenML first-class `--flavor=mlflow` support. W&B self-hosted
requires MySQL + Redis + commercial license.

**eoAPI for STAC** -- Production deploys eoAPI at `stac.ai.hotosm.org`
(`k8s-infra/apps/fair/eoapi/values.yaml`). Dev uses the same chart (v0.12.0)
with `external-plaintext` DB.

**ZenML Postgres patch** -- OSS ZenML only supports MySQL/SQLite. The patched image
at [`ghcr.io/hotosm/zenml-postgres`](https://github.com/hotosm/fAIr/tree/develop/infra/zenml)
replaces MySQL dialect (`MEDIUMTEXT`) with Postgres equivalents. Both server image
and client need patching -- `make client-patch` handles the client side via the same
[`patch_zenml.py`](https://github.com/hotosm/fAIr/blob/develop/infra/zenml/patch_zenml.py)
used in the Dockerfile.

**StacBackend Protocol** -- `StacCatalogManager` writes local JSON files.
`PgStacBackend` writes to pgstac via pypgstac. Both conform to the `StacBackend`
Protocol (structural subtyping). `run.py --stac-api-url` selects pgstac; omit for local.

**PgStacBackend reads via pystac-client** -- The eoAPI chart injects
`--root-path=/stac` by default, which breaks self-links under direct port-forwarding.
Dev values set `stac.overrideRootPath: ""` to remove it, so pystac-client works
correctly against `http://localhost:8082`.

**GPU scheduling from STAC metadata** -- `mlm:accelerator` and `mlm:accelerator_count`
in `stac-item.json` drive `nvidia.com/gpu` resource requests. `config.py` reads these
and emits pod settings only when the orchestrator is Kubernetes.

## Dev -> Prod delta

| Dev (kind)            | Prod (EKS)                            |
|-----------------------|---------------------------------------|
| PG StatefulSet        | CloudNativePG cluster                 |
| MinIO                 | AWS S3                                |
| eoAPI dev values      | `k8s-infra/apps/fair/eoapi/values.yaml` |
| ZenML Helm (same OCI) | TBF               |
| MLflow dev values     | TBF           |
| kind kubeconfig       | TBF                        |

## Known issues

**eoAPI root_path (resolved)** -- The chart's
[deployment template](https://github.com/developmentseed/eoapi-k8s/blob/main/charts/eoapi/templates/services/stac/deployment.yaml)
injects `--root-path={{ .Values.stac.ingress.path }}` (defaults to `/stac`) into
the uvicorn command when an ingress class is set. Dev values set
`stac.overrideRootPath: ""` which removes the arg entirely, so pystac-client
works via direct port-forwarding.

## References

- [ZenML Postgres patch](https://github.com/hotosm/fAIr/tree/develop/infra/zenml)
- [ZenML K8s Orchestrator](https://docs.zenml.io/stack-components/orchestrators/kubernetes)
- [pypgstac](https://stac-utils.github.io/pgstac/pypgstac/)
- [eoAPI-k8s](https://github.com/developmentseed/eoapi-k8s)
- [community-charts/mlflow](https://github.com/community-charts/helm-charts/tree/main/charts/mlflow)
- [kind](https://kind.sigs.k8s.io/)
