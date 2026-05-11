"""Step tests for RAMP building segmentation.

Each test runs real @step entrypoints against the toy OAM chips/labels fixture.
No pipeline-internal mocks; telemetry sinks are already no-ops via
models/conftest.py::mock_instrumentation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

_PRETRAINED_URL = "https://huggingface.co/hotosm/ramp/resolve/74daea54694f2e4924f1222520c614c7f5c029fe/v1-baseline.zip"


@pytest.fixture(scope="session")
def pretrained_weights(tmp_path_factory: pytest.TempPathFactory) -> str:
    from upath import UPath

    cache = tmp_path_factory.mktemp("ramp_weights") / "baseline.zip"
    cache.write_bytes(UPath(_PRETRAINED_URL).read_bytes())
    return str(cache)


def test_split_dataset(toy_chips: Path, toy_labels: Path, base_hyperparameters: dict[str, Any]) -> None:
    from models.ramp.pipeline import split_dataset

    hyperparameters = dict(base_hyperparameters)
    hyperparameters.update(
        {
            "epochs": 1,
            "batch_size": 1,
            "training.val_ratio": 0.25,
            "training.split_seed": 42,
            "training.boundary_width": 1,
            "training.contact_spacing": 2,
        }
    )

    result = split_dataset.entrypoint(
        dataset_chips=str(toy_chips),
        dataset_labels=str(toy_labels),
        hyperparameters=hyperparameters,
    )

    assert result["strategy"] == "random"
    assert result["train_count"] > 0
    assert result["val_count"] > 0
    assert "_ramp_train_dir" in result
    assert "_preprocessed_dir" in result
    ramp_dir = Path(result["_ramp_train_dir"])
    assert ramp_dir.exists()
    assert (ramp_dir / "chips").is_dir()
    assert (ramp_dir / "val-chips").is_dir()


def test_train_model(
    toy_chips: Path,
    toy_labels: Path,
    base_hyperparameters: dict[str, Any],
    pretrained_weights: str,
) -> None:
    from models.ramp.pipeline import split_dataset, train_model

    hyperparameters = dict(base_hyperparameters)
    hyperparameters.update(
        {
            "epochs": 1,
            "batch_size": 1,
            "training.val_ratio": 0.25,
            "training.split_seed": 42,
            "training.boundary_width": 1,
            "training.contact_spacing": 2,
        }
    )

    split_info = split_dataset.entrypoint(
        dataset_chips=str(toy_chips),
        dataset_labels=str(toy_labels),
        hyperparameters=hyperparameters,
    )
    model_bytes = train_model.entrypoint(
        dataset_chips=str(toy_chips),
        dataset_labels=str(toy_labels),
        base_model_weights=pretrained_weights,
        hyperparameters=hyperparameters,
        split_info=split_info,
        num_classes=4,
    )

    assert isinstance(model_bytes, bytes)
    assert len(model_bytes) > 0


def test_evaluate_model(
    toy_chips: Path,
    toy_labels: Path,
    base_hyperparameters: dict[str, Any],
    pretrained_weights: str,
) -> None:
    from models.ramp.pipeline import evaluate_model, split_dataset, train_model

    hyperparameters = dict(base_hyperparameters)
    hyperparameters.update(
        {
            "epochs": 1,
            "batch_size": 1,
            "training.val_ratio": 0.25,
            "training.split_seed": 42,
            "training.boundary_width": 1,
            "training.contact_spacing": 2,
        }
    )

    split_info = split_dataset.entrypoint(
        dataset_chips=str(toy_chips),
        dataset_labels=str(toy_labels),
        hyperparameters=hyperparameters,
    )
    model_bytes = train_model.entrypoint(
        dataset_chips=str(toy_chips),
        dataset_labels=str(toy_labels),
        base_model_weights=pretrained_weights,
        hyperparameters=hyperparameters,
        split_info=split_info,
        num_classes=4,
    )
    metrics = evaluate_model.entrypoint(
        trained_model=model_bytes,
        dataset_chips=str(toy_chips),
        dataset_labels=str(toy_labels),
        hyperparameters=hyperparameters,
        split_info=split_info,
    )

    expected = {"fair:accuracy", "fair:mean_iou", "fair:precision", "fair:recall"}
    assert set(metrics.keys()) == expected
    for value in metrics.values():
        assert isinstance(value, float)
        assert 0.0 <= value <= 1.0


def test_export_onnx(
    toy_chips: Path,
    toy_labels: Path,
    base_hyperparameters: dict[str, Any],
    pretrained_weights: str,
) -> None:
    import onnx

    from models.ramp.pipeline import export_onnx, split_dataset, train_model

    hyperparameters = dict(base_hyperparameters)
    hyperparameters.update(
        {
            "epochs": 1,
            "batch_size": 1,
            "training.val_ratio": 0.25,
            "training.split_seed": 42,
            "training.boundary_width": 1,
            "training.contact_spacing": 2,
        }
    )

    split_info = split_dataset.entrypoint(
        dataset_chips=str(toy_chips),
        dataset_labels=str(toy_labels),
        hyperparameters=hyperparameters,
    )
    model_bytes = train_model.entrypoint(
        dataset_chips=str(toy_chips),
        dataset_labels=str(toy_labels),
        base_model_weights=pretrained_weights,
        hyperparameters=hyperparameters,
        split_info=split_info,
        num_classes=4,
    )
    exported = export_onnx.entrypoint(trained_model=model_bytes)

    assert isinstance(exported, bytes)
    loaded = onnx.load_from_string(exported)
    onnx.checker.check_model(loaded)
