from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from fair.zenml.runs import (
    LogEntry,
    RunSummary,
    fetch_run_logs,
    fetch_step_logs,
    get_run_status,
    list_runs_for_model,
)


def _stub_status(value: str) -> Any:
    return MagicMock(value=value)


def _stub_run(*, run_id: str = "run-1", status: str = "running", with_logs: bool = True) -> MagicMock:
    run = MagicMock()
    run.id = run_id
    run.name = "training-pipeline-2026-05-01"
    run.status = _stub_status(status)
    run.created = datetime(2026, 5, 1, 12, 0, tzinfo=UTC)
    run.pipeline = MagicMock(name="pipeline")
    run.pipeline.name = "training-pipeline"
    run.log_collection = [MagicMock()] if with_logs else []
    return run


def _stub_log_entry(message: str, level: str = "INFO") -> Any:
    entry = MagicMock()
    entry.message = message
    entry.level = MagicMock(value=level)
    entry.timestamp = datetime(2026, 5, 1, 12, 1, tzinfo=UTC)
    return entry


@patch("fair.zenml.runs.Client")
def test_get_run_status_normalizes_enum(mock_client_cls: MagicMock) -> None:
    mock_client_cls.return_value.get_pipeline_run.return_value = _stub_run(status="COMPLETED")
    assert get_run_status("run-1") == "completed"


@patch("fair.zenml.runs.Client")
def test_get_run_status_rejects_unknown(mock_client_cls: MagicMock) -> None:
    mock_client_cls.return_value.get_pipeline_run.return_value = _stub_run(status="MYSTERY")
    with pytest.raises(RuntimeError, match="unknown ZenML run status"):
        get_run_status("run-1")


@patch("fair.zenml.runs.Client")
def test_get_run_status_handles_provisioning(mock_client_cls: MagicMock) -> None:
    mock_client_cls.return_value.get_pipeline_run.return_value = _stub_run(status="PROVISIONING")
    assert get_run_status("run-1") == "provisioning"


@patch("fair.zenml.runs.Client")
def test_fetch_run_logs_raises_when_log_store_missing(mock_client_cls: MagicMock) -> None:
    client = mock_client_cls.return_value
    client.get_pipeline_run.return_value = _stub_run()
    client.active_stack.log_store = None
    with pytest.raises(RuntimeError, match="no log_store"):
        fetch_run_logs("run-1")


@patch("fair.zenml.runs.Client")
def test_fetch_run_logs_concatenates_collections(mock_client_cls: MagicMock) -> None:
    client = mock_client_cls.return_value
    run = _stub_run()
    run.log_collection = [MagicMock(), MagicMock()]
    client.get_pipeline_run.return_value = run

    log_store = MagicMock()
    log_store.fetch.side_effect = [
        [_stub_log_entry("first"), _stub_log_entry("second")],
        [_stub_log_entry("third")],
    ]
    client.active_stack.log_store = log_store

    entries = fetch_run_logs("run-1", tail=10)
    assert [e.message for e in entries] == ["first", "second", "third"]
    assert all(isinstance(e, LogEntry) for e in entries)


@patch("fair.zenml.runs.Client")
def test_fetch_run_logs_respects_tail(mock_client_cls: MagicMock) -> None:
    client = mock_client_cls.return_value
    run = _stub_run()
    run.log_collection = [MagicMock(), MagicMock()]
    client.get_pipeline_run.return_value = run

    log_store = MagicMock()
    log_store.fetch.side_effect = [
        [_stub_log_entry("a"), _stub_log_entry("b")],
        [_stub_log_entry("c")],
    ]
    client.active_stack.log_store = log_store

    entries = fetch_run_logs("run-1", tail=2)
    assert len(entries) == 2
    assert log_store.fetch.call_count == 1


@patch("fair.zenml.runs.Client")
def test_fetch_step_logs_unknown_step_raises_keyerror(mock_client_cls: MagicMock) -> None:
    client = mock_client_cls.return_value
    run = _stub_run()
    run.steps = {"train_model": MagicMock()}
    client.get_pipeline_run.return_value = run

    with pytest.raises(KeyError, match="not found"):
        fetch_step_logs("run-1", "missing_step")


@patch("fair.zenml.runs.Client")
def test_fetch_step_logs_returns_entries(mock_client_cls: MagicMock) -> None:
    client = mock_client_cls.return_value
    step = MagicMock()
    step.log_collection = [MagicMock()]
    run = _stub_run()
    run.steps = {"train_model": step}
    client.get_pipeline_run.return_value = run

    log_store = MagicMock()
    log_store.fetch.return_value = [_stub_log_entry("training started")]
    client.active_stack.log_store = log_store

    entries = fetch_step_logs("run-1", "train_model")
    assert len(entries) == 1
    assert entries[0].message == "training started"


@patch("fair.zenml.runs.Client")
def test_list_runs_for_model_flattens_versions(mock_client_cls: MagicMock) -> None:
    client = mock_client_cls.return_value

    version_1 = MagicMock()
    version_1.id = "v1-id"
    version_1.number = 1
    version_2 = MagicMock()
    version_2.id = "v2-id"
    version_2.number = 2
    client.list_model_versions.return_value = MagicMock(items=[version_2, version_1])

    def list_links(*, model_version_id: str):
        run = _stub_run(run_id=f"run-for-{model_version_id}", status="completed")
        return MagicMock(items=[MagicMock(pipeline_run=run)])

    client.list_model_version_pipeline_run_links.side_effect = list_links

    summaries = list_runs_for_model("my-model", limit=10)
    assert [s.id for s in summaries] == ["run-for-v2-id", "run-for-v1-id"]
    assert all(isinstance(s, RunSummary) for s in summaries)
    assert summaries[0].model_version == 2
