# Examples

End-to-end pipelines that register a base model, finetune on the sample
Banepa OAM+OSM dataset, promote, and run inference. One runner for all
models.

## Quick Start

```bash
just setup                          # bring up the stack
just build                          # build all model images
just example                        # run all examples end-to-end
```

Or one model at a time:

```bash
just example unet_segmentation
just example resnet18_classification
just example yolo11n_detection
```

## Overriding hyperparameters

The runner uses STAC `mlm:hyperparameters` defaults. Override any of
`epochs`, `batch_size`, `learning_rate`, `samples_per_epoch`, `chip_size`
by invoking python directly:

```bash
uv run python examples/run.py unet_segmentation --epochs 1 --samples-per-epoch 10
```

## How it works

1. Reads `models/<name>/stac-item.json`; derives the task type from
   `mlm:tasks[0]` and the model id from `id`.
2. Picks the matching dataset at `data/sample/buildings-banepa-<task>/stac-item.json`.
3. Calls `FairClient.setup → register_base_model → register_dataset →
   finetune → promote → predict`.
4. Predicts on `data/sample/predict/oam/`.

Each step runs inside the model's docker image (via ZenML's docker
orchestrator). Same image dok8s deploys via KNative — see
[`just test-serve`](../justfile) for the API smoke test.

## Output

| Where | What |
| --- | --- |
| <http://localhost:8080> | ZenML dashboard: pipeline runs, steps, artifacts |
| <http://localhost:5000> | MLflow runs and metrics |
| <http://localhost:8082/collections> | STAC items for registered models and datasets |
| <http://localhost:9001> | MinIO browser (login: `minioadmin` / `minioadmin`) |
| `data/sample/predict/predictions/` | Per-task predictions on disk |

## Adding a new model

Drop a new directory under `models/`:

```
models/<your_model>/
├── Dockerfile
├── pipeline.py        # exports training_pipeline, inference_pipeline, predict
└── stac-item.json     # declares mlm:tasks, mlm:hyperparameters, mlm:training/inference image hrefs
```

Then:

```bash
just build <your_model>
just example <your_model>
just test-serve <your_model>
```