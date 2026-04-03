from __future__ import annotations

from unittest.mock import patch

import pytest

from fair.zenml.metrics import (
    log_fair_metrics,
    log_training_wall_time,
    read_fair_metrics,
    read_training_wall_time,
)


@patch("fair.zenml.metrics.log_metadata")
def test_log_fair_metrics_prefixes_keys(mock_log):
    log_fair_metrics({"accuracy": 0.95, "mean_iou": 0.80})
    mock_log.assert_called_once_with(
        metadata={"fair/accuracy": 0.95, "fair/mean_iou": 0.80},
        infer_model=True,
    )


@patch("fair.zenml.metrics.log_metadata")
def test_log_fair_metrics_respects_infer_model_flag(mock_log):
    log_fair_metrics({"accuracy": 0.5}, infer_model=False)
    mock_log.assert_called_once_with(
        metadata={"fair/accuracy": 0.5},
        infer_model=False,
    )


def test_read_fair_metrics_converts_prefix():
    raw = {"fair/accuracy": 0.95, "fair/mean_iou": 0.80, "other_key": "ignored"}
    result = read_fair_metrics(raw)
    assert result == {"fair:accuracy": 0.95, "fair:mean_iou": 0.80}


def test_read_fair_metrics_excludes_wall_time():
    raw = {"fair/accuracy": 0.95, "fair/training_wall_seconds": 42.5}
    result = read_fair_metrics(raw)
    assert result == {"fair:accuracy": 0.95}


def test_read_fair_metrics_returns_none_for_empty():
    assert read_fair_metrics(None) is None
    assert read_fair_metrics({}) is None


def test_read_fair_metrics_returns_none_when_no_fair_keys():
    assert read_fair_metrics({"some_other": 1}) is None


@patch("fair.zenml.metrics.log_metadata")
def test_log_training_wall_time(mock_log):
    log_training_wall_time(123.456)
    mock_log.assert_called_once_with(
        metadata={"fair/training_wall_seconds": 123.456},
        infer_model=True,
    )


def test_read_training_wall_time():
    assert read_training_wall_time({"fair/training_wall_seconds": 42.5}) == 42.5


def test_read_training_wall_time_returns_none():
    assert read_training_wall_time(None) is None
    assert read_training_wall_time({}) is None
    assert read_training_wall_time({"fair/accuracy": 0.9}) is None


@pytest.mark.parametrize("value", [42, "42.5", 42.5])
def test_read_training_wall_time_coerces_to_float(value):
    assert read_training_wall_time({"fair/training_wall_seconds": value}) == float(value)
