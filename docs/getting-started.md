---
icon: lucide/rocket
---

# Getting Started

## Prerequisites

!!! info "Required tools"

    - :simple-docker: Docker
    - :simple-astral: [uv](https://docs.astral.sh/uv/)
    - [just](https://just.systems/)

## Installation

=== ":lucide-laptop: Develop fAIr-models"

    ```bash title="Clone and bring up the stack"
    git clone https://github.com/hotosm/fAIr-models.git
    cd fAIr-models
    just setup
    ```

    `just setup` installs Python deps with `uv`, brings up Postgres, MinIO,
    STAC, MLflow, and ZenML via Docker Compose, and registers the `compose`
    ZenML stack as active.

=== ":lucide-package: Use as a library"

    ```bash title="Add to your project"
    uv add fair-py-ops
    ```

## Running the Example Pipelines

Three example pipelines demonstrate the full workflow for each supported task
type: register a base model, finetune on sample data, promote the best version,
and run inference.

| Example | Task | Model |
| --- | --- | --- |
| `examples/segmentation/` | Semantic segmentation | UNet (torchgeo) |
| `examples/classification/` | Binary classification | ResNet18 (torchvision) |
| `examples/detection/` | Object detection | YOLOv11n (ultralytics) |

### Run All Pipelines

```bash
just example
```

??? example "Running a single example"

    ```bash
    AWS_ENDPOINT_URL=http://localhost:9000 \
    AWS_ACCESS_KEY_ID=minioadmin \
    AWS_SECRET_ACCESS_KEY=minioadmin \
    FAIR_STAC_API_URL=http://localhost:8082 \
    FAIR_DSN=postgresql://postgres:postgres@localhost:5432/fair_models \
        uv run python examples/segmentation/run.py
    ```

### Verifying Results

!!! success "After the pipeline completes"

    | What | Where |
    | --- | --- |
    | ZenML pipelines, steps, artifacts | <http://localhost:8080> (login: `default` / empty) |
    | STAC collections | <http://localhost:8082/collections> |
    | MLflow runs | <http://localhost:5000> |
    | MinIO objects | <http://localhost:9001> (login: `minioadmin` / `minioadmin`) |
    | Trained weights | `artifacts/` |
    | Predictions | `data/sample/test/predictions/` |

## Project Structure

```text
fair/                  # Core library (pip-installable as fair-py-ops)
  stac/                # STAC catalog management, builders, validators
  utils/               # Data helpers
  zenml/               # ZenML config generation, promotion, steps
models/                # Base model contributions (one subdir per model)
examples/              # Example pipelines (segmentation, classification, detection)
infra/                 # Production stack (Kubernetes via helmfile, DigitalOcean via OpenTofu)
infra/compose/         # Local dev stack (this is what `just setup` uses)
stacks/compose.yaml    # ZenML stack definition for the compose stack
tests/                 # pytest suite
```

## Development Commands

```bash title="Available recipes"
just setup     # install deps + bring up stack + register ZenML stack
just example   # run all 3 example pipelines
just down      # stop the stack (state preserved, fast restart)
just up        # restart after `just down`
just tear      # destroy stack + volumes + local ZenML state
just lint      # ruff check + format + ty check
just test      # pytest
just validate  # validate STAC items + model pipelines
just docs      # serve documentation locally
just commit    # run pre-commit hooks + commitizen
```

## Next Steps

!!! tip

    - :lucide-blocks: Read the [Architecture](architecture.md) overview to understand the system
    - :lucide-box: [Contribute a model](contributing/model.md) to fAIr
    - :lucide-container: Stand up the [Kubernetes dev stack](development/k8s.md) for production-parity testing
