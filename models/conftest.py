"""Shared pytest fixtures for model pipeline tests.

Reads model metadata from stac-item.json. Per-model conftest provides a
generate_toy_dataset fixture that creates chips, labels, and a dataset
STAC item in a tmp directory. Mocks ZenML/MLflow instrumentation so step
tests run without a live server.
"""

from __future__ import annotations

import contextlib
import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--model-dir", type=str, help="Path to model directory under models/")


@pytest.fixture(scope="session")
def model_dir(request: pytest.FixtureRequest) -> Path:
    raw = request.config.getoption("--model-dir", default=None)
    if raw:
        return Path(raw).resolve()
    for arg in request.config.args:
        candidate = Path(arg).resolve()
        for parent in [candidate, *candidate.parents]:
            if (parent / "stac-item.json").exists() and (parent / "pipeline.py").exists():
                return parent
    msg = "Cannot determine model directory. Pass --model-dir or run from models/<name>/tests/"
    raise pytest.UsageError(msg)


@pytest.fixture(scope="session")
def model_stac_item(model_dir: Path) -> dict[str, Any]:
    return json.loads((model_dir / "stac-item.json").read_text())


@pytest.fixture(scope="session")
def task_type(model_stac_item: dict[str, Any]) -> str:
    tasks = model_stac_item["properties"].get("mlm:tasks", [])
    return tasks[0] if tasks else "unknown"


@pytest.fixture(scope="session")
def chip_size(model_stac_item: dict[str, Any]) -> int:
    shape = model_stac_item["properties"]["mlm:input"][0]["input"]["shape"]
    return shape[-1]


@pytest.fixture(scope="session")
def class_names(model_stac_item: dict[str, Any]) -> list[str]:
    classes = model_stac_item["properties"]["mlm:output"][0]["classification:classes"]
    return [cls["name"] for cls in classes]


@pytest.fixture(scope="session")
def num_classes(class_names: list[str]) -> int:
    return len(class_names)


@pytest.fixture(scope="session")
def toy_chips(generate_toy_dataset: dict[str, Path]) -> Path:
    return generate_toy_dataset["chips"]


@pytest.fixture(scope="session")
def toy_labels(generate_toy_dataset: dict[str, Path]) -> Path:
    return generate_toy_dataset["labels"]


@pytest.fixture(scope="session")
def dataset_stac_item(generate_toy_dataset: dict[str, Path]) -> Path:
    return generate_toy_dataset["dataset_stac_item"]


@pytest.fixture()
def base_hyperparameters() -> dict[str, Any]:
    return {"epochs": 1}


@contextmanager
def _noop_context(*_args: Any, **_kwargs: Any):
    yield


def _discover_pipeline_modules() -> list[str]:
    models_root = Path(__file__).parent
    return sorted(f"models.{p.parent.name}.pipeline" for p in models_root.glob("*/pipeline.py"))


@pytest.fixture(autouse=True)
def mock_instrumentation():
    import importlib

    patches = [
        patch("fair.zenml.instrumentation.mlflow_training_context", _noop_context),
        patch("fair.zenml.instrumentation.log_evaluation_results"),
        patch("fair.zenml.metrics.log_metadata"),
        patch("fair.zenml.metrics.log_loss_history"),
        patch("zenml.log_metadata"),
    ]
    for module_name in _discover_pipeline_modules():
        try:
            importlib.import_module(module_name)
        except ImportError:
            continue
        patches.append(patch(f"{module_name}.log_metadata"))

    with contextlib.ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        yield


@pytest.fixture(scope="session", autouse=True)
def _force_cpu():
    os.environ["FAIR_FORCE_CPU"] = "1"
