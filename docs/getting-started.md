---
icon: lucide/rocket
---

# Getting Started

## Prerequisites

!!! info "Required tools"

    - :simple-python: Python 3.11+
    - :simple-astral: [uv](https://docs.astral.sh/uv/) (package manager)
    - :simple-docker: Docker (for model runtime containers)

## Installation

=== ":lucide-laptop: Local development"

    ```bash title="Clone and set up the project"
    git clone https://github.com/hotosm/fAIr-models.git
    cd fAIr-models
    just setup
    ```

=== ":lucide-container: Kubernetes dev stack"

    ```bash title="Clone and set up with k8s extras"
    git clone https://github.com/hotosm/fAIr-models.git
    cd fAIr-models
    just k8s
    just setup
    ```

    `just k8s` switches to k8s mode (sticky, persists across sessions).
    `just setup` then installs k8s extras and checks that
    `kind`, `kubectl`, `helm`, `helmfile`, and `mc` are on `$PATH`.
    Use `just local` to switch back. See [Kubernetes Dev Stack](development/k8s.md).

=== ":lucide-package: As a library"

    ```bash title="Add to your project"
    uv add fair-py-ops
    ```

## Running the Example Pipelines

Three example pipelines demonstrate the full workflow for each supported task
type: register a base model, finetune on sample data, promote the best version,
and run inference.

| Example | Task | Model |
|---|---|---|
| `examples/segmentation/` | Semantic segmentation | UNet (torchgeo) |
| `examples/classification/` | Binary classification | ResNet18 (torchvision) |
| `examples/detection/` | Object detection | YOLOv11n (ultralytics) |

### Running All Pipelines

```bash title="Run all three pipelines"
just example
```

??? example "Running a single example"

    ```bash
    uv run python examples/segmentation/run.py
    uv run python examples/classification/run.py
    uv run python examples/detection/run.py
    ```

??? example "Kubernetes orchestrator run"

    ```bash
    just k8s
    just setup
    cd infra/dev
    just up
    just run-example-k8s
    ```

Current example scripts execute the full workflow in one run: setup, register
base model, register dataset, finetune, promote, and predict.

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
  utils/               # Data helpers
  zenml/               # ZenML config generation, promotion, steps
models/                # Base model contributions (one subdir per model)
examples/              # CLI runners for local development
infra/ci/              # Kind cluster config for CI integration tests
infra/dev/             # Helmfile stack for local Kubernetes dev
infra/dok8s/           # OpenTofu config for the DigitalOcean deployment
tests/                 # pytest suite
```

## Development Commands

All targets adapt to the active mode (`local` by default). Switch with `just k8s` or `just local`.

```bash title="Available recipes"
just local             # switch to local mode (default)
just k8s               # switch to k8s mode (sticky)
just setup             # install deps (k8s mode adds extras + tool checks)
just lint              # ruff check + format + ty check
just test              # pytest
just validate          # validate STAC items + model pipelines
just example           # run example pipeline (k8s mode delegates to infra/dev)
just docs              # serve documentation locally
just clean             # remove ZenML state + artifacts (k8s mode also tears down cluster)
```

## Next Steps

!!! tip

    - :lucide-blocks: Read the [Architecture](architecture.md) overview to understand the system
    - :lucide-box: [Contribute a model](contributing/model.md) to fAIr
    - :lucide-container: Set up the [Kubernetes dev stack](development/k8s.md) for production-like testing
    - :lucide-book-open: Learn [Markdown authoring with Zensical](https://zensical.org/docs/authoring/markdown/) for writing documentation
