from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

from zenml.client import Client

log = logging.getLogger(__name__)

RunStatus = Literal[
    "initializing",
    "provisioning",
    "running",
    "completed",
    "failed",
    "cached",
    "retrying",
    "retried",
    "stopped",
    "stopping",
]


@dataclass(frozen=True, slots=True)
class LogEntry:
    level: str
    message: str
    timestamp: str | None = None


@dataclass(frozen=True, slots=True)
class RunSummary:
    id: str
    name: str
    status: RunStatus
    created_at: str | None
    pipeline_name: str
    model_name: str | None
    model_version: int | None


def get_run_status(run_id: str) -> RunStatus:
    """Return the lifecycle status of a ZenML pipeline run."""
    run = Client().get_pipeline_run(run_id)
    return _normalize_status(run.status)


def fetch_run_logs(run_id: str, *, tail: int = 1000) -> list[LogEntry]:
    """Fetch run-level logs.

    Raises if the active stack has no log store. We never silently return [] when
    the user asked for logs and the system can't supply them.
    """
    client = Client()
    run = client.get_pipeline_run(run_id)
    log_store = client.active_stack.log_store
    if log_store is None:
        raise RuntimeError(
            "active ZenML stack has no log_store; cannot fetch run logs. "
            "Configure a log store (e.g. artifact-store-backed) on your stack."
        )
    collection = run.log_collection or []
    return _collect_entries(log_store, collection, tail)


def fetch_step_logs(run_id: str, step_name: str, *, tail: int = 1000) -> list[LogEntry]:
    """Fetch logs for a single step within a run."""
    client = Client()
    run = client.get_pipeline_run(run_id)
    step = run.steps.get(step_name)
    if step is None:
        available = ", ".join(sorted(run.steps.keys())) or "<none>"
        raise KeyError(f"step '{step_name}' not found in run '{run_id}'. Available: {available}")
    log_store = client.active_stack.log_store
    if log_store is None:
        raise RuntimeError("active ZenML stack has no log_store; cannot fetch step logs.")
    collection = step.log_collection or []
    return _collect_entries(log_store, collection, tail)


def list_runs_for_model(model_name: str, *, limit: int = 50) -> list[RunSummary]:
    """List pipeline runs that produced a version of the given ZenML model.

    Sorted newest first. Used by the backend to render "all training runs for this
    model" without storing a duplicate Training table.
    """
    client = Client()
    versions = client.list_model_versions(model=model_name, sort_by="desc:created", size=limit)
    summaries: list[RunSummary] = []
    for version in versions.items:
        links = client.list_model_version_pipeline_run_links(model_version_id=version.id)
        for link in links.items:
            run = link.pipeline_run
            summaries.append(
                RunSummary(
                    id=str(run.id),
                    name=run.name,
                    status=_normalize_status(run.status),
                    created_at=run.created.isoformat() if run.created else None,
                    pipeline_name=run.pipeline.name if run.pipeline else "",
                    model_name=model_name,
                    model_version=version.number,
                )
            )
    return summaries[:limit]


def _collect_entries(log_store: object, collection: list, tail: int) -> list[LogEntry]:
    out: list[LogEntry] = []
    for logs_model in collection:
        remaining = tail - len(out)
        if remaining <= 0:
            break
        entries = log_store.fetch(logs_model=logs_model, limit=remaining)  # type: ignore[attr-defined]
        out.extend(_to_entry(e) for e in entries)
    return out


_KNOWN_STATUSES: frozenset[str] = frozenset(
    {
        "initializing",
        "provisioning",
        "running",
        "completed",
        "failed",
        "cached",
        "retrying",
        "retried",
        "stopped",
        "stopping",
    }
)
_TERMINAL_STATUSES: frozenset[str] = frozenset({"completed", "failed", "cached", "retried", "stopped"})


def _normalize_status(status: object) -> RunStatus:
    raw = getattr(status, "value", status)
    if not isinstance(raw, str):
        raise RuntimeError(f"unexpected status type from ZenML: {type(status)!r}")
    lowered = raw.lower()
    if lowered in _KNOWN_STATUSES:
        return lowered  # type: ignore[return-value]
    raise RuntimeError(f"unknown ZenML run status: {raw!r}")


def is_terminal(status: RunStatus) -> bool:
    return status in _TERMINAL_STATUSES


def _to_entry(raw: object) -> LogEntry:
    level = getattr(raw, "level", None)
    level_str = getattr(level, "value", level) if level is not None else "INFO"
    return LogEntry(
        level=str(level_str),
        message=str(getattr(raw, "message", "")),
        timestamp=_to_iso(getattr(raw, "timestamp", None)),
    )


def _to_iso(value: object) -> str | None:
    if value is None:
        return None
    iso = getattr(value, "isoformat", None)
    if callable(iso):
        return str(iso())
    return str(value)
