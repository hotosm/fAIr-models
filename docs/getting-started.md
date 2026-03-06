---
icon: lucide/rocket
---

# Getting Started

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (package manager)
- Docker (for model runtime containers)

## Installation

### For model development

```bash
git clone https://github.com/hotosm/fAIr-models.git
cd fAIr-models
uv sync --group local --group example
make init
```

### As a library

```bash
uv add fair-py-ops
```

## Running the Example Pipeline

The included UNet example demonstrates the full workflow — register a base
model, finetune it on sample data, promote the best version, and run inference.

```bash
# Run the full pipeline: init → register → finetune → promote → predict
python examples/unet/run.py all
```

Individual steps:

```bash
python examples/unet/run.py init       # Initialize ZenML + STAC catalog
python examples/unet/run.py register   # Register base model + dataset
python examples/unet/run.py finetune   # Train (1 epoch on sample data)
python examples/unet/run.py promote    # Promote to production + publish STAC
python examples/unet/run.py predict    # Run inference
```

### Verifying Results

After the pipeline completes:

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

```bash
make lint              # ruff check + format check
make format            # auto-fix lint issues
make typecheck         # ty check
make test              # pytest
make validate-stac     # validate all STAC items
make validate-models   # validate model pipeline exports
```

## Next Steps

- Read the [Architecture](architecture.md) overview to understand the system
- [Contribute a model](contributing/model.md) to fAIr
- Set up the [Kubernetes dev stack](development/k8s.md) for production-like testing
- Learn [Markdown authoring with Zensical](https://zensical.org/docs/authoring/markdown/) for writing documentation
