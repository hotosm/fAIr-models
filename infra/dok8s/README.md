# DigitalOcean Deployment

Deploys fAIr-models on DOKS with managed PostgreSQL and DO Spaces.

## Architecture

- **DOKS cluster**: `infra` pool + autoscaling `ml` pool (0-1 nodes)
- **Managed PostgreSQL 17**: PostGIS-enabled (databases: zenml, fair_models, mlflow, mlflow_auth)
- **DO Spaces**: S3-compatible artifact store
- **nginx-ingress**: single DO Load Balancer, wildcard DNS `*.FAIR_DOMAIN`
- **cert-manager**: automated HTTPS via Let's Encrypt

Pipelines run in-cluster via ZenML K8s orchestrator. Training and inference pods schedule on ML pool nodes labeled `FAIR_DOMAIN/training=true` and `FAIR_DOMAIN/inference=true` (labels set at pool level, not manually).

## Prerequisites

- `doctl` (authenticated), `helm`, `helmfile`, `kubectl`
- `psql`, `aws` CLI, `uv`

## Environment

All configuration lives in `.env`. Copy the example and fill in your values:

```
cp env.example .env
```

| Variable | Source |
|---|---|
| `DO_TOKEN` | DO API token |
| `CLUSTER_NAME` | Cluster name (default: `fair`) |
| `DO_REGION` | DO region (default: `nyc3`) |
| `FAIR_DOMAIN` | Your wildcard domain |
| `PG_*` | Auto-populated by `make infra` |
| `SPACES_BUCKET` | S3 bucket name |
| `SPACES_ACCESS_KEY` | Spaces access key |
| `SPACES_SECRET_KEY` | Spaces secret key |
| `MLFLOW_ADMIN_USER` | MLflow admin username |
| `MLFLOW_ADMIN_PASSWORD` | MLflow admin password |
| `ZENML_ADMIN_USER` | ZenML admin username |
| `ZENML_ADMIN_PASSWORD` | ZenML admin password |
| `LETSENCRYPT_EMAIL` | Email for Let's Encrypt certs |

## Usage

```
make up             # provision infra + deploy + DNS
make run-example    # submit UNet pipeline to cluster
make status         # check cluster health
make urls           # print service URLs
make tear           # destroy (keeps Spaces bucket)
```

Individual steps can be run directly:

```
scripts/cluster.sh create       # create DOKS cluster
scripts/database.sh create      # create managed Postgres
scripts/database.sh write-env   # populate .env with PG creds
scripts/deploy.sh helm          # deploy helm releases
scripts/deploy.sh dns           # set up wildcard DNS
```

## GPU Node Pool

To add a GPU pool for accelerated training:

```bash
doctl kubernetes cluster node-pool create fair \
  --name gpu \
  --size gpu_1x_nvidia_a100 \
  --count 0 --auto-scale --min-nodes 0 --max-nodes 1 \
  --label "FAIR_DOMAIN/training=true" \
  --taint "FAIR_DOMAIN/training=true:NoSchedule"
```

Replace `FAIR_DOMAIN` with your actual domain value. Node pool labels handle scheduling automatically.
