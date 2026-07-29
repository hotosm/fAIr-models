"""Every pipeline submission must run in the model image without source download."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from zenml.config import DockerSettings

import fair.client as client_module
from fair.client import FairClient

_DOWNLOAD_FLAGS = ("allow_download_from_code_repository", "allow_download_from_artifact_store")


class RecordingPipeline:
    def __init__(self, run: Any) -> None:
        self._run = run
        self.settings: dict[str, Any] = {}

    def with_options(self, *, config_path: str, enable_cache: bool, settings: dict[str, Any]) -> Any:
        self.settings = settings
        return lambda: self._run


def _training_run() -> SimpleNamespace:
    return SimpleNamespace(id="run-1", status="completed")


def _inference_run() -> SimpleNamespace:
    predictions = SimpleNamespace(load=lambda: {"predictions": []})
    step = SimpleNamespace(outputs={"predictions": [predictions]})
    return SimpleNamespace(id="run-2", status="completed", steps={"run_inference": step})


@pytest.fixture
def client(tmp_path: Path) -> FairClient:
    return FairClient(config_dir=str(tmp_path))


def _stub_training(monkeypatch: pytest.MonkeyPatch, client: FairClient, tmp_path: Path) -> RecordingPipeline:
    pipeline = RecordingPipeline(_training_run())
    monkeypatch.setattr(
        client,
        "_prepare_training_pipeline",
        lambda **_: (SimpleNamespace(training_pipeline=pipeline), tmp_path / "train.yaml"),
    )
    return pipeline


def _stub_inference(monkeypatch: pytest.MonkeyPatch, client: FairClient, tmp_path: Path) -> RecordingPipeline:
    pipeline = RecordingPipeline(_inference_run())
    monkeypatch.setattr(
        client,
        "_prepare_inference_pipeline",
        lambda **_: (SimpleNamespace(inference_pipeline=pipeline), tmp_path / "inference.yaml"),
    )
    return pipeline


def test_docker_settings_disable_every_source_download_path() -> None:
    settings = client_module._MODEL_IMAGE_DOCKER_SETTINGS
    for flag in _DOWNLOAD_FLAGS:
        # DockerSettings ignores unknown fields, so a rename upstream would
        # silently drop the flag instead of raising.
        assert flag in DockerSettings.model_fields, f"ZenML no longer defines DockerSettings.{flag}"
        assert getattr(settings, flag) is False


def test_finetune_runs_in_the_model_image(monkeypatch: pytest.MonkeyPatch, client: FairClient, tmp_path) -> None:
    pipeline = _stub_training(monkeypatch, client, tmp_path)

    client.finetune(base_model_id="b-1", dataset_id="d-1", model_name="m-1")

    assert pipeline.settings["docker"] is client_module._MODEL_IMAGE_DOCKER_SETTINGS


def test_submit_finetune_runs_in_the_model_image(monkeypatch: pytest.MonkeyPatch, client: FairClient, tmp_path) -> None:
    pipeline = _stub_training(monkeypatch, client, tmp_path)

    assert client.submit_finetune(base_model_id="b-1", dataset_id="d-1", model_name="m-1") == "run-1"

    assert pipeline.settings["docker"] is client_module._MODEL_IMAGE_DOCKER_SETTINGS
    assert pipeline.settings["orchestrator"] == {"synchronous": False}


def test_predict_runs_in_the_model_image(monkeypatch: pytest.MonkeyPatch, client: FairClient, tmp_path) -> None:
    pipeline = _stub_inference(monkeypatch, client, tmp_path)

    assert client.predict("lm-1", str(tmp_path)) == {"predictions": []}

    assert pipeline.settings["docker"] is client_module._MODEL_IMAGE_DOCKER_SETTINGS


def test_submit_predict_runs_in_the_model_image(monkeypatch: pytest.MonkeyPatch, client: FairClient, tmp_path) -> None:
    pipeline = _stub_inference(monkeypatch, client, tmp_path)

    assert client.submit_predict(local_model_id="lm-1", image_path=str(tmp_path)) == "run-2"

    assert pipeline.settings["docker"] is client_module._MODEL_IMAGE_DOCKER_SETTINGS
    assert pipeline.settings["orchestrator"] == {"synchronous": False}
