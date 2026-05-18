---
icon: lucide/wrench
---

# Development

The default dev stack is Docker Compose. See [Getting Started](../getting-started.md):

```bash
just setup    # uv sync + docker compose up + zenml stack register
just example  # run all 3 pipelines
```

For production parity (kind cluster + helmfile, mirrors the EKS deployment), see [Kubernetes Dev Stack](k8s.md).
