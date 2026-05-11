# fAIr Infrastructure

One stack, two targets:

- **dev** — `kind` cluster on your laptop (default).
- **dok8s** — DigitalOcean Kubernetes (DOKS) with managed DNS, TLS, and KNative live serving.

Both run **the same in-cluster services** (Postgres+PostGIS, MinIO, STAC, MLflow, ZenML), driven by a single helmfile and a single justfile.

## Layout

```
infra/
├── helmfile.yaml.gotmpl    # all chart releases (env-aware)
├── justfile                # all commands
├── kind-config.yaml        # local cluster
├── ports.conf              # dev port-forward map
├── environments/           # per-env values (dev.yaml, dok8s.yaml.gotmpl)
├── values/                 # chart values (env-templated)
├── manifests/              # raw k8s manifests (postgres, ingress, knative, ...)
├── scripts/                # seed_data.py, zenml-token.sh
└── dok8s/                  # OpenTofu: cluster + ml node pool only
```

## Prerequisites

| Use case | Tools |
|---|---|
| dev (kind) | `kind`, `kubectl`, `helm`, `helmfile`, `uv` |
| dok8s | + `tofu`, `doctl` (authenticated), `envsubst`, `psql` |

## Commands

```bash
just up           # spin up local stack (kind)
just up dok8s     # provision + deploy on DigitalOcean
just example      # run all 3 example pipelines on the active stack
just example dok8s
just predict      # smoke-test live KNative endpoints (dok8s)
just urls         # show service URLs
just status       # cluster + pod health
just down         # stop port-forwards / helm uninstall
just tear         # delete kind cluster / tofu destroy
```

## dok8s setup

```bash
cd infra/dok8s
cp terraform.tfvars.example terraform.tfvars
# fill in: do_token, domain, letsencrypt_email, mlflow/zenml admin creds
cd ..
just up dok8s
```

`just up dok8s` runs: `tofu apply` → save kubeconfig → apply Postgres → `helmfile apply` → cluster-issuer + ingresses + KNative-serving + s3-credentials → wait for LB IP → set wildcard DNS → seed data → register ZenML stack → write `.env.dok8s`.

## Running examples remotely (after `just up dok8s`)

The justfile-driven `just example dok8s` flow handles everything. To submit a pipeline by hand:

```bash
source infra/.env.dok8s
uv run --group example python examples/segmentation/run.py
```

The `dok8s` ZenML stack is configured to schedule pipeline pods on the `ml` autoscaling pool via `fair/workload=ml`.
