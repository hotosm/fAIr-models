# fAIr-models

Model registry and ML pipeline orchestration for [fAIr](https://github.com/hotosm/fAIr). 

**`fair-py-ops`** is the Python package for building [ZenML](https://zenml.io/) pipelines, validating [STAC](https://stacspec.org/) items, and testing locally. The `models/` directory is the single source of truth for base model contributions.

## Quick Start

```bash
git clone https://github.com/hotosm/fAIr-models.git
cd fAIr-models
just setup
just example
```

See [Getting Started](docs/getting-started.md) for detailed setup, environment options, and running individual examples.

## Documentation

- **[Getting Started](docs/getting-started.md)** : Installation, local setup, Kubernetes, and running examples
- **[Architecture](docs/architecture.md)** : STAC catalog structure, flows, identity model, infrastructure
- **[Contributing a Model](docs/contributing/model.md)** : Guide for adding base models to fAIr
- **[API Reference](docs/reference/index.md)** : Python package documentation
- **[Changelog](docs/changelog.md)** : Release history

## Examples

Three reference implementations demonstrate the full workflow for each supported task:

| Example | Task | Model | Path |
|---------|------|-------|------|
| Segmentation | Semantic segmentation | UNet (torchgeo) | [`examples/segmentation/`](examples/segmentation/) |
| Classification | Binary classification | ResNet18 (torchvision) | [`examples/classification/`](examples/classification/) |
| Detection | Object detection | YOLOv11n (ultralytics) | [`examples/detection/`](examples/detection/) |

## Available Commands

Run `just` to see all recipes. Common commands:

```bash
just setup          # Install dependencies and set up environment
just example        # Run all three example pipelines
just lint           # Run Ruff linting and type checking (ty)
just test           # Run unit tests
just k8s            # Set up Kubernetes dev environment
just local          # Switch back to local mode
```

## Key Concepts

| Concept | Description |
|---------|-------------|
| **Base model** | Reusable ML blueprint (weights, code, Docker image, STAC item) |
| **Local model** | Finetuned model produced by ZenML pipeline on user data |
| **STAC catalog** | Model/dataset registry with [MLM](https://github.com/stac-extensions/mlm) and [Version](https://github.com/stac-extensions/version) extensions |
| **ZenML pipeline** | Orchestrated training and inference workflows |
