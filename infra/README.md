# fAIr Infrastructure

One stack, two targets:

- **dev**: `kind` cluster on your laptop (default).
- **cluster**: any managed Kubernetes cluster, adding ingress, TLS, and KNative live serving.

Both run **the same in-cluster services** (Postgres+PostGIS, MinIO, STAC, MLflow, ZenML), driven by a single helmfile and a single justfile.

Provisioning the cluster itself, the kubeconfig, and the DNS records is out of scope here. The `cluster` recipes act on whatever context `kubectl` currently points at.

## Layout

Everything below lives under `infra/`. `helmfile.yaml.gotmpl` declares all
chart releases (env-aware), `justfile` holds all commands, `kind-config.yaml`
defines the local cluster, and `ports.conf` maps the dev port-forwards.
Per-env values sit in `environments/` (`dev.yaml`, `cluster.yaml.gotmpl`),
chart values in `values/` (env-templated), raw k8s manifests (postgres,
ingress, knative, and so on) in `manifests/`, and helper scripts
(`seed_data.py`, `zenml-token.sh`) in `scripts/`.

## Prerequisites

| Use case   | Tools                                                     |
| ---------- | --------------------------------------------------------- |
| dev (kind) | `kind`, `kubectl`, `helm`, `helmfile`, `uv`               |
| cluster    | + `envsubst`, `psql`, a kubeconfig for the target cluster |

## Commands

```bash
just up           # spin up local stack (kind)
just up cluster   # deploy to the cluster kubectl points at
just example      # run both example pipelines on the active stack
just predict      # smoke-test live KNative endpoints
just urls         # show service URLs
just status       # cluster + pod health
just down         # stop port-forwards / helm uninstall
just tear         # delete kind cluster / uninstall releases
```

## Cluster setup

The `cluster` recipes read their configuration from the environment:

```bash
export FAIR_DOMAIN=fair.example.com
export LETSENCRYPT_EMAIL=ops@example.com
export MLFLOW_ADMIN_USER=... MLFLOW_ADMIN_PASSWORD=...
export ZENML_ADMIN_USER=... ZENML_ADMIN_PASSWORD=... ZENML_STORE_API_KEY=...
just up cluster
```

`just up cluster` runs: apply Postgres → `helmfile apply` → KNative-serving + s3-credentials → wait for the ingress LoadBalancer IP → cluster-issuer + ingresses. Point `*.$FAIR_DOMAIN` and `*.predict.$FAIR_DOMAIN` at the printed IP; wildcards do not span dots, so both records are needed.

## Serving a model

KNative services are registered per model, separately from the STAC registration:

```bash
fair knative register models/dinov3s_buildings/stac-item.json
fair knative status dinov3s-buildings
```

`fair basemodel register` then checks `https://<model>.predict.$FAIR_PREDICT_DOMAIN/health` and refuses to publish a model whose service is not already serving.

## Running examples remotely

`just example` handles the port-forwards, the ZenML token, and the env. To submit a pipeline by hand:

```bash
eval "$(just _env)"
uv run python examples/run.py dinov3s_buildings
```

The `cluster` ZenML stack schedules pipeline pods on the ML pool via `fair/workload=ml`.
