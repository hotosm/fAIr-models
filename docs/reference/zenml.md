---
icon: lucide/workflow
---

# ZenML Integration

Config generation, model promotion lifecycle, and reusable pipeline steps.

## Workflow Runner

`FairWorkflowRunner` is the main entry point for running fAIr pipelines locally.
Each example (`examples/segmentation/run.py`, `examples/classification/run.py`,
`examples/detection/run.py`) instantiates a runner with the base model ID,
pipeline module, dataset config, and optional hyperparameter overrides.

The runner provides a CLI with commands: `init`, `register`, `finetune`,
`promote`, `predict`, `verify`, `all`, and `clean`. Config YAML files are
auto-generated from the STAC item metadata via `generate_training_config` and
`generate_inference_config`, so model developers never write pipeline configs
by hand.

::: fair.zenml.runner

## Config Generation

::: fair.zenml.config

## Promotion

::: fair.zenml.promotion

## Steps

::: fair.zenml.steps
