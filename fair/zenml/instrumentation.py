from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any

from fair.zenml.metrics import log_fair_metrics, log_training_wall_time


@contextmanager
def mlflow_training_context(
    hyperparameters: dict[str, Any],
    model_name: str | None = None,
    base_model_id: str | None = None,
    dataset_id: str | None = None,
):
    """Context manager that instruments a training step with MLflow.

    Handles autolog, param logging, tag setting, and wall-clock timing.
    Contributors use this instead of manual MLflow calls.
    """
    import mlflow

    mlflow.autolog()  # ty: ignore[possibly-missing-attribute]
    mlflow.log_params(  # ty: ignore[possibly-missing-attribute]
        {k: v for k, v in hyperparameters.items() if not isinstance(v, (dict, list))}
    )

    tags: dict[str, str] = {}
    if model_name:
        tags["fair.model_name"] = model_name
    if base_model_id:
        tags["fair.base_model"] = base_model_id
    if dataset_id:
        tags["fair.dataset"] = dataset_id
    if tags:
        mlflow.set_tags(tags)  # ty: ignore[possibly-missing-attribute]

    wall_start = time.perf_counter()
    yield
    wall_seconds = time.perf_counter() - wall_start
    log_training_wall_time(wall_seconds)


def log_evaluation_results(metrics: dict[str, Any]) -> None:
    """Log evaluation metrics to both MLflow and ZenML fair-prefixed metadata."""
    import mlflow

    scalar_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
    mlflow.log_metrics(scalar_metrics)  # ty: ignore[possibly-missing-attribute]
    log_fair_metrics(metrics)
