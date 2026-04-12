# DigitalOcean Deployment

Deploys fAIr-models on DOKS with managed PostgreSQL and DO Spaces using OpenTofu.

## Architecture

- **DOKS cluster**: `infra` pool + autoscaling `ml` pool (0-1 nodes)
- **Managed PostgreSQL 17**: PostGIS-enabled (databases: zenml, fair_models, mlflow, mlflow_auth)
- **DO Spaces**: S3-compatible artifact store
- **nginx-ingress**: single DO Load Balancer, wildcard DNS `*.{domain}`
- **cert-manager**: automated HTTPS via Let's Encrypt

Services exposed at `stac.{domain}`, `mlflow.{domain}`, `zenml.{domain}`.

Pipelines run in-cluster via ZenML K8s orchestrator. Training and inference pods schedule on ML pool nodes labeled `{domain}/training=true` and `{domain}/inference=true`.

## Prerequisites

- `tofu` (OpenTofu), `doctl` (authenticated), `helm`, `kubectl`
- `psql`, `jq`, `envsubst`, `uv`

## Configuration

Copy the tfvars example and fill in your values:

```
cp terraform.tfvars.example terraform.tfvars
```

| Variable | Description |
|---|---|
| `do_token` | DigitalOcean API token |
| `domain` | Base domain for wildcard DNS (e.g. `fair.example.com`) |
| `spaces_bucket` | DO Spaces bucket name |
| `spaces_access_key` | Spaces access key |
| `spaces_secret_key` | Spaces secret key |
| `mlflow_admin_user` | MLflow admin username |
| `mlflow_admin_password` | MLflow admin password |
| `zenml_admin_user` | ZenML admin username |
| `zenml_admin_password` | ZenML admin password |
| `letsencrypt_email` | Email for Let's Encrypt certificates |
| `cluster_name` | Cluster name (default: `fair`) |
| `region` | DO region (default: `nyc3`) |
| `infra_node_size` | Infra node size (default: `s-2vcpu-4gb`) |
| `ml_node_size` | ML node size (default: `s-4vcpu-8gb`) |

## Usage

```bash
just init           # initialize OpenTofu
just plan           # preview changes
just up             # first-time setup: provision, deploy, seed, register stack
just deploy         # apply infra/service changes only (no seed, no stack re-register)
just status         # check cluster health
just urls           # print service URLs
just tear           # destroy all resources (keeps Spaces bucket data)
just run-example    # submit all pipelines to cluster
```

`just up` runs the full sequence: `tofu apply`, saves kubeconfig, applies cluster-issuer, installs PostGIS extensions, runs pgstac migration, seeds data, and registers the ZenML stack.

## GPU Node Pool

The ML node pool uses CPU instances by default. To add a GPU pool for accelerated training:

```bash
doctl kubernetes cluster node-pool create fair \
  --name gpu \
  --size gpu_1x_nvidia_a100 \
  --count 0 --auto-scale --min-nodes 0 --max-nodes 1 \
  --label "{domain}/training=true" \
  --taint "{domain}/training=true:NoSchedule"
```

Replace `{domain}` with the value of your `domain` variable.
