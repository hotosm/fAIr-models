"""Hyperparameter prefix filtering shared by config generation and client."""

from __future__ import annotations

from typing import Any

TRAINING_PREFIX = "training."
INFERENCE_PREFIX = "inference."


def filter_params(hyperparams: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {key[len(prefix) :]: value for key, value in hyperparams.items() if key.startswith(prefix)}


def training_params(hyperparams: dict[str, Any]) -> dict[str, Any]:
    return filter_params(hyperparams, TRAINING_PREFIX)


def inference_params(hyperparams: dict[str, Any]) -> dict[str, Any]:
    return filter_params(hyperparams, INFERENCE_PREFIX)
