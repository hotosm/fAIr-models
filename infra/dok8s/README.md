# fAIr on DigitalOcean Kubernetes (dok8s)

OpenTofu for the dok8s cluster, the DigitalOcean DNS01 token, and the
in-cluster secrets pods consume at boot. Helm releases, manifests, and
Knative routing live one level up under `infra/`.

## Files

| File | What it does |
| --- | --- |
| `main.tf` | DOKS cluster + two node pools (`infra-v2`, `ml`) |
| `secrets.tf` | `fair-backend-secrets`, `digitalocean-dns`, `cert-manager` namespace |
| `variables.tf` | inputs: `do_token`, `domain`, `letsencrypt_email`, admin creds |
| `outputs.tf` | values consumed by `infra/justfile` |
| `terraform.tfvars.example` | template for a local `terraform.tfvars` |

## Cluster

```
                          <client>
                             │  HTTPS
                             ▼
                  ingress-nginx (LoadBalancer)
                             │
            ┌────────────────┼─────────────────────┐
            │                │                     │
            ▼                ▼                     ▼
       api.<domain>    stac / zenml /        *.predict.<domain>
                       mlflow / s3                 │
                                                   ▼
                                              Kourier
                                                   │
                                                   ▼
                                       Knative ksvc per model
                                       (scale 0..5, ONNX runtime)
```

Two node pools:

- **`infra-v2`** (`s-4vcpu-8gb`, 1 node, always on): hosts the long-running
  services (postgres, mlflow, zenml, minio, stac, ingress-nginx,
  cert-manager, knative control plane, fair-backend). Pinned via
  `nodeSelector: fair/role=infra`.
- **`ml`** (autoscale 0..5): ZenML pipeline pods and Knative Revision pods.
  Pinned via `nodeSelector: fair/workload=ml`. Scales to zero when idle.

TLS terminates at ingress-nginx. The `*.predict.<domain>` host uses a
wildcard cert issued by cert-manager's DigitalOcean DNS01 solver, because
Let's Encrypt rejects HTTP01 for wildcards. Every other public host uses a
per-host cert via HTTP01.

DNS has two wildcard A records: `*.<domain>` and `*.predict.<domain>`, both
pointing at the LoadBalancer IP. DNS wildcards do not span dots, so the
second record is required.

## Bring up

```bash
cd infra/dok8s
cp terraform.tfvars.example terraform.tfvars
# fill in: do_token, domain, letsencrypt_email, mlflow/zenml admin creds
cd ..
just up dok8s
```

`just up dok8s` runs the full chain: `tofu apply` (cluster + secrets) →
save kubeconfig → bootstrap postgres → `helmfile apply` (charts) →
install Knative + ingresses → wait for LB IP → write DNS records →
issue ClusterIssuers + wildcard cert.

## Tear down

```bash
cd infra
just tear dok8s
```

Runs `helmfile destroy` then `tofu destroy`. Wildcard DNS records written
by `_set-dns` are not removed automatically; delete them by hand if also
giving up the domain.
