"""Fixtures for the Chesapeake model tests and shared integration test."""

from pathlib import Path
from typing import Any

import pytest

from models.chesapeake_rsc_v2.example_data import create_toy_data as _create_toy_data


def create_toy_data(root: Path) -> dict[str, Any]:
    # The shared test forces the mask threshold to zero after one training epoch.
    return {**_create_toy_data(root), "predict_params": {"trace_method": "mask"}}


@pytest.fixture(scope="session")
def generate_toy_dataset(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    return _create_toy_data(tmp_path_factory.mktemp("chesapeake-roads"))
