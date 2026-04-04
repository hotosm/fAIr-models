---
icon: lucide/wrench
---

# Development

Guides for developing and testing fAIr Models locally and on Kubernetes.

!!! tip "Choose your path"

    Switch modes with `just local` or `just k8s` (sticky, persists across sessions).
    All targets adapt automatically.

    === ":lucide-laptop: Local"

        ```bash
        just setup    # local ZenML orchestrator + file-based STAC
        just example  # run full pipeline locally
        ```

        See [Getting Started](../getting-started.md) for the full guide.

    === ":lucide-container: Kubernetes"

        ```bash
        just k8s && just setup  # install k8s extras + check CLI tools
        cd infra/dev && just up
        just example
        ```

        See [Kubernetes Dev Stack](k8s.md) for the full guide.
