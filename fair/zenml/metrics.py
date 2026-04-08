from __future__ import annotations

from typing import Any

from zenml import log_metadata

_ZENML_PREFIX = "fair/"
_STAC_PREFIX = "fair:"
_WALL_TIME_KEY = f"{_ZENML_PREFIX}training_wall_seconds"
_SPLIT_KEY = f"{_ZENML_PREFIX}split"
_NON_METRIC_KEYS = frozenset({_WALL_TIME_KEY, _SPLIT_KEY})


def log_fair_metrics(metrics: dict[str, Any], *, infer_model: bool = True) -> None:
    prefixed = {f"{_ZENML_PREFIX}{k}": v for k, v in metrics.items()}
    log_metadata(metadata=prefixed, infer_model=infer_model)


def read_fair_metrics(run_metadata: dict[str, Any] | None) -> dict[str, Any] | None:
    if not run_metadata:
        return None
    converted = {
        k.replace(_ZENML_PREFIX, _STAC_PREFIX, 1): v
        for k, v in run_metadata.items()
        if k.startswith(_ZENML_PREFIX) and k not in _NON_METRIC_KEYS
    }
    return converted or None


def log_training_wall_time(seconds: float) -> None:
    log_metadata(metadata={_WALL_TIME_KEY: seconds}, infer_model=True)


def read_training_wall_time(run_metadata: dict[str, Any] | None) -> float | None:
    if not run_metadata:
        return None
    value = run_metadata.get(_WALL_TIME_KEY)
    return float(value) if value is not None else None
