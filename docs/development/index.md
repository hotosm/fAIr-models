---
icon: lucide/wrench
---

# Development

Guides for developing and testing fAIr Models locally and on Kubernetes.

!!! tip "Choose your path"

    Switch modes with `make local` or `make k8s` (sticky, persists across sessions).
    All targets adapt automatically.

    === ":lucide-laptop: Local"

        ```bash
        make setup    # local ZenML orchestrator + file-based STAC
        make example  # run full pipeline locally
        ```

        See [Getting Started](../getting-started.md) for the full guide.

    === ":lucide-container: Kubernetes"

        ```bash
        make k8s && make setup  # install k8s extras + check CLI tools
        cd infra/dev && make up
        make example
        ```

        See [Kubernetes Dev Stack](k8s.md) for the full guide.
