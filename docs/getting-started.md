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

=== "For model development"

    ```bash title="Clone and set up the project"
    git clone https://github.com/hotosm/fAIr-models.git
    cd fAIr-models
    make setup
    ```

=== "As a library"

    ```bash title="Add to your project"
    uv add fair-py-ops
    ```

## Running the Example Pipeline

The included UNet example demonstrates the full workflow — register a base
model, finetune it on sample data, promote the best version, and run inference.

```bash title="Run the full pipeline"
python examples/unet/run.py all  # init → register → finetune → promote → predict
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

```bash title="Available make targets"
make setup             # install deps, pre-commit hooks, zenml init
make lint              # ruff check + format + ty check
make test              # pytest
make validate          # validate STAC items + model pipelines
make example           # run full example pipeline (clean + all)
make docs              # serve documentation locally
make clean             # remove ZenML state + build artifacts
```

## Next Steps

!!! tip

    - :lucide-blocks: Read the [Architecture](architecture.md) overview to understand the system
    - :lucide-box: [Contribute a model](contributing/model.md) to fAIr
    - :lucide-container: Set up the [Kubernetes dev stack](development/k8s.md) for production-like testing
    - :lucide-book-open: Learn [Markdown authoring with Zensical](https://zensical.org/docs/authoring/markdown/) for writing documentation
