---
icon: lucide/home
---

# fAIr Models

Model registry and ML pipeline orchestration for [fAIr](https://github.com/hotosm/fAIr).

---

**`fair-py-ops`** is the Python package model developers install to build
[ZenML](https://zenml.io/) pipelines, validate
[STAC](https://stacspec.org/) items, and test the full workflow locally
before submitting to the fAIr platform.

## What is fAIr?

!!! info "Humanitarian AI for OpenStreetMap"

    fAIr is a humanitarian AI platform by [HOT](https://www.hotosm.org/) that
    enables feature extraction from very high resolution aerial imagery (buildings,
    roads, trees) for OpenStreetMap mapping. This repository provides the ML
    infrastructure layer.

## Key Concepts

| Concept | Description |
|---|---|
| **Base model** | A reusable ML blueprint contributed via PR (e.g. UNet, YOLOv8) |
| **Local model** | A finetuned model produced by a ZenML pipeline on user data |
| **STAC catalog** | The model/dataset registry using [MLM](https://github.com/stac-extensions/mlm) and [Version](https://github.com/stac-extensions/version) extensions |
| **ZenML pipeline** | Orchestrated training and inference workflows |

## Quick Links

!!! tip "Start here"

    - :lucide-rocket: [Getting Started](getting-started.md) : install, run example, understand the workflow
    - :lucide-blocks: [Architecture](architecture.md) : STAC catalog structure, flows, infrastructure
    - :lucide-box: [Contributing a Model](contributing/model.md) : add your model to fAIr
    - :lucide-book-open: [API Reference](reference/index.md) : Python package documentation
    - :lucide-clock: [Changelog](changelog.md) : release history
