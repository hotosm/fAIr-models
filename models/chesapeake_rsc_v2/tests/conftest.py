"""Synthetic road fixtures for model and integration tests."""

from pathlib import Path

import pytest

from models.chesapeake_rsc_v2.example_data import create_toy_data


@pytest.fixture(scope="session")
def generate_toy_dataset(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    return create_toy_data(tmp_path_factory.mktemp("chesapeake-roads"))
