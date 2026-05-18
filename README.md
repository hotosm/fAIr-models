# fAIr-models

[![codecov](https://codecov.io/gh/hotosm/fAIr-models/branch/master/graph/badge.svg)](https://codecov.io/gh/hotosm/fAIr-models)

Model registry and ML pipeline orchestration for [fAIr](https://github.com/hotosm/fAIr).

**`fair-py-ops`** is the Python package for building [ZenML](https://zenml.io/) pipelines, validating [STAC](https://stacspec.org/) items, and testing locally. The `models/` directory is the single source of truth for base model contributions.

## Quick Start

Prerequisites: Docker, [uv](https://docs.astral.sh/uv/), [just](https://just.systems/).

```bash
git clone https://github.com/hotosm/fAIr-models.git
cd fAIr-models
just setup
just example
```

`just setup` installs Python deps, brings up the full stack via Docker Compose (Postgres + MinIO + STAC + MLflow + ZenML), and registers the ZenML stack. `just example` runs all three reference pipelines end-to-end.

| Service | URL | Credentials |
| --- | --- | --- |
| ZenML dashboard | <http://localhost:8080> | `default` / (empty) |
| MLflow | <http://localhost:5000> | none |
| STAC API | <http://localhost:8082> | none |
| MinIO console | <http://localhost:9001> | `minioadmin` / `minioadmin` |

See [Getting Started](docs/getting-started.md) for the full guide. For Kubernetes parity or production deploys, see [`infra/README.md`](infra/README.md).

## Documentation

- **[Getting Started](docs/getting-started.md)** : Installation and running the examples
- **[Architecture](docs/architecture.md)** : STAC catalog structure, flows, identity model, infrastructure
- **[Contributing a Model](docs/contributing/model.md)** : Guide for adding base models to fAIr
- **[API Reference](docs/reference/index.md)** : Python package documentation
- **[Changelog](docs/changelog.md)** : Release history

## Examples

Three reference implementations demonstrate the full workflow for each supported task:

| Example | Task | Model | Path |
| --- | --- | --- | --- |
| Segmentation | Semantic segmentation | UNet (torchgeo) | [`examples/segmentation/`](examples/segmentation/) |
| Classification | Binary classification | ResNet18 (torchvision) | [`examples/classification/`](examples/classification/) |
| Detection | Object detection | YOLOv11n (ultralytics) | [`examples/detection/`](examples/detection/) |

## Commands

Run `just` to see all recipes.

```bash
just setup     # install deps + bring up stack + register ZenML stack
just example   # run all 3 example pipelines
just down      # stop the stack (state preserved, fast restart)
just up        # restart after `just down`
just tear      # destroy stack + volumes + local ZenML state
just lint      # ruff + ty
just test      # pytest
just validate  # validate STAC items + model pipelines
just docs      # serve documentation locally
just commit    # run pre-commit hooks + commitizen
```

## Key Concepts

| Concept | Description |
| --- | --- |
| **Base model** | Reusable ML blueprint (weights, code, Docker image, STAC item) |
| **Local model** | Finetuned model produced by ZenML pipeline on user data |
| **STAC catalog** | Model/dataset registry with [MLM](https://github.com/stac-extensions/mlm) and [Version](https://github.com/stac-extensions/version) extensions |
| **ZenML pipeline** | Orchestrated training and inference workflows |
