---
icon: lucide/rocket
---

# Getting Started

## Prerequisites

!!! info "Required tools"

    - :simple-python: Python 3.12+
    - :simple-astral: [uv](https://docs.astral.sh/uv/) (package manager)
    - :simple-docker: Docker (for model runtime containers)

## Installation

=== ":lucide-laptop: Local development"

    ```bash title="Clone and set up the project"
    git clone https://github.com/hotosm/fAIr-models.git
    cd fAIr-models
    make setup
    ```

=== ":lucide-container: Kubernetes dev stack"

    ```bash title="Clone and set up with k8s extras"
    git clone https://github.com/hotosm/fAIr-models.git
    cd fAIr-models
    make k8s
    make setup
    ```

    `make k8s` switches to k8s mode (sticky, persists across sessions).
    `make setup` then installs k8s extras and checks that
    `kind`, `kubectl`, `helm`, `helmfile`, `mc`, and `envsubst` are on `$PATH`.
    Use `make local` to switch back. See [Kubernetes Dev Stack](development/k8s.md).

=== ":lucide-package: As a library"

    ```bash title="Add to your project"
    uv add fair-py-ops
    ```

## Running the Example Pipeline

The included UNet example demonstrates the full workflow — register a base
model, finetune it on sample data, promote the best version, and run inference.

```bash title="Run the full pipeline"
make example  # init → register → finetune → promote → predict
```

??? example "Individual steps"

    ```bash
    python examples/unet/run.py init       # Initialize ZenML + STAC catalog
    python examples/unet/run.py register   # Register base model + dataset
    python examples/unet/run.py finetune   # Train (1 epoch on sample data)
    python examples/unet/run.py promote    # Promote to production + publish STAC
    python examples/unet/run.py predict    # Run inference
    ```

### Verifying Results

!!! success "After the pipeline completes"

    | Artifact | Location |
    | --- | --- |
    | STAC catalog | `stac_catalog/` (3 collections) |
    | Trained weights | `artifacts/` |
    | Predictions | `data/sample/predict/predictions/` |
    | ZenML dashboard | <http://localhost:8080> |

## Project Structure

```text
fair/                  # Core library (pip-installable as fair-py-ops)
  stac/                # STAC catalog management, builders, validators
  utils/               # Data helpers, model validation
  zenml/               # ZenML config generation, promotion, steps
models/                # Base model contributions (one subdir per model)
examples/              # CLI runners for local development
infra/dev/             # Kind cluster Helm values for K8s dev stack
tests/                 # pytest suite
```

## Development Commands

All targets adapt to the active mode (`local` by default). Switch with `make k8s` or `make local`.

```bash title="Available make targets"
make local             # switch to local mode (default)
make k8s               # switch to k8s mode (sticky)
make setup             # install deps (k8s mode adds extras + tool checks)
make lint              # ruff check + format + ty check
make test              # pytest
make validate          # validate STAC items + model pipelines
make example           # run example pipeline (k8s mode delegates to infra/dev)
make docs              # serve documentation locally
make clean             # remove ZenML state + artifacts (k8s mode also tears down cluster)
```

## Next Steps

!!! tip

    - :lucide-blocks: Read the [Architecture](architecture.md) overview to understand the system
    - :lucide-box: [Contribute a model](contributing/model.md) to fAIr
    - :lucide-container: Set up the [Kubernetes dev stack](development/k8s.md) for production-like testing
    - :lucide-book-open: Learn [Markdown authoring with Zensical](https://zensical.org/docs/authoring/markdown/) for writing documentation
