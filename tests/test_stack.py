from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from fair.cli import app
from fair.infra import stack

runner = CliRunner()


def test_assets_resolve_to_existing_paths() -> None:
    for path in (stack.compose_file(), stack.stack_config_file()):
        assert path.exists()


def test_missing_asset_raises() -> None:
    with pytest.raises(stack.FairStackError):
        stack._asset(("nope", "missing.yml"), ("nope", "missing.yml"))


def test_up_brings_stack_up_then_registers(monkeypatch) -> None:
    calls = MagicMock()
    monkeypatch.setattr(stack, "_compose", calls.compose)
    monkeypatch.setattr(stack, "register_zenml_stack", calls.register)

    result = runner.invoke(app, ["stack", "up"])

    assert result.exit_code == 0
    calls.compose.assert_called_once_with("up", "-d", "--wait")
    calls.register.assert_called_once_with()


def test_down_without_volumes(monkeypatch) -> None:
    compose = MagicMock()
    monkeypatch.setattr(stack, "_compose", compose)

    result = runner.invoke(app, ["stack", "down"])

    assert result.exit_code == 0
    compose.assert_called_once_with("down")


def test_down_with_volumes(monkeypatch) -> None:
    compose = MagicMock()
    monkeypatch.setattr(stack, "_compose", compose)

    result = runner.invoke(app, ["stack", "down", "--volumes"])

    assert result.exit_code == 0
    compose.assert_called_once_with("down", "-v")


def test_require_missing_binary_raises(monkeypatch) -> None:
    monkeypatch.setattr(stack.shutil, "which", lambda _: None)
    with pytest.raises(stack.FairStackError):
        stack._require("docker")


def test_fetch_token_returns_access_token(monkeypatch) -> None:
    response = MagicMock()
    response.json.return_value = {"access_token": "tok-123"}
    monkeypatch.setattr(stack.httpx, "post", lambda *a, **k: response)

    assert stack._fetch_token() == "tok-123"
