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

## Running an example manually

Yes. If you want to submit an example to the remote k8s stack outside `just run-example`, you need the pgstac database credentials exposed through `PGSTAC_DSN`. The client writes STAC records through PostgreSQL and reads through the STAC API.

If you leave `FAIR_STAC_API_URL` unset, the client falls back to the local file catalog and `FAIR_DSN` is not required. Make sure the client has access to postgresql ( or it is port forwarded ) as well as aws s3 bucket. 

| Variable | Required | Purpose | Example value |
|---|---|---|---|
| `ZENML_STORE_URL` | Yes | Remote ZenML server URL | `https://zenml.fair.example.com` |
| `ZENML_STORE_API_TOKEN` | Yes | Token used to submit authenticated runs | `paste-your-zenml-token-here` |
| `ZENML_STORE_VERIFY_SSL` | Usually | TLS verification setting for ZenML | `true` |
| `FAIR_STAC_API_URL` | Yes for remote runs | Remote STAC API endpoint | `https://stac.fair.example.com/stac` |
| `FAIR_DSN` | Yes for remote runs | pgstac PostgreSQL DSN used for writes | `postgresql://doadmin:your-password@db-host:25060/fair_models?sslmode=require` |
| `FAIR_USER_ID` | No | Logical user identifier | `krschap` |
| `FAIR_LABEL_DOMAIN` | Recommended on k8s | Label domain for pod scheduling | `fair.example.com` |
| `FAIR_UPLOAD_ARTIFACTS` | Optional | Upload registered model and dataset artifacts through the client | `true` |
| `AWS_ENDPOINT_URL` | If uploads enabled | S3-compatible object storage endpoint | `https://nyc3.digitaloceanspaces.com` |
| `AWS_ACCESS_KEY_ID` | If uploads enabled | Object storage access key | `your-access-key` |
| `AWS_SECRET_ACCESS_KEY` | If uploads enabled | Object storage secret key | `your-secret-key` |
| `FAIR_FORCE_CPU` | Optional | Force CPU execution | `1` |

Example for the segmentation pipeline:

```bash
uv sync --extra k8s --group example

export ZENML_STORE_URL="https://zenml.fair.example.com"
export ZENML_STORE_API_TOKEN="paste-your-zenml-token-here"
export ZENML_STORE_VERIFY_SSL="true"
export FAIR_DSN="postgresql://doadmin:your-password@db-host:25060/fair_models?sslmode=require"
export FAIR_STAC_API_URL="https://stac.fair.example.com/stac"
export FAIR_LABEL_DOMAIN="fair.example.com"
export FAIR_UPLOAD_ARTIFACTS="true"
export AWS_ENDPOINT_URL="https://nyc3.digitaloceanspaces.com"
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
export FAIR_FORCE_CPU=1

uv run --group example python examples/segmentation/run.py
```

To run a different example, swap the path:

```bash
uv run --group example python examples/classification/run.py
uv run --group example python examples/detection/run.py
```

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

Replace `{domain}` with the value of  `domain` variable.
