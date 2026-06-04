"""Step tests for dinov3s-buildings.

Each test runs the real @step entrypoint against toy OAM chips + GeoJSON labels.
Telemetry sinks (zenml/mlflow) are no-ops via models/conftest.py.
"""

from pathlib import Path
from typing import Any

import pytest


@pytest.fixture(scope="session")
def pretrained_weights(tmp_path_factory: pytest.TempPathFactory) -> str:
    """v5s Lightning checkpoint published alongside dinov3-buildings on HF."""
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id="kshitijrajsharma/dinov3-hot-buildings",
        filename="dinov3s_upernet_hot.ckpt",
    )


def test_split_dataset(toy_chips: Path, toy_labels: Path, base_hyperparameters: dict[str, Any]) -> None:
    from models.dinov3s_buildings.pipeline import split_dataset

    info = split_dataset.entrypoint(
        dataset_chips=str(toy_chips),
        dataset_labels=str(toy_labels),
        hyperparameters=base_hyperparameters,
    )
    assert info["strategy"] == "spatial"
    assert info["val_ratio"] > 0
    assert info["train_count"] > 0
    assert info["val_count"] > 0
    assert len(info["train_chip_names"]) == info["train_count"]
    assert len(info["val_chip_names"]) == info["val_count"]
    assert set(info["train_chip_names"]).isdisjoint(info["val_chip_names"])


def test_train_model(
    toy_chips: Path,
    toy_labels: Path,
    base_hyperparameters: dict[str, Any],
    pretrained_weights: str,
) -> None:
    from models.dinov3s_buildings.pipeline import split_dataset, train_model

    info = split_dataset.entrypoint(
        dataset_chips=str(toy_chips),
        dataset_labels=str(toy_labels),
        hyperparameters=base_hyperparameters,
    )
    hp = {**base_hyperparameters, "epochs": 1}
    model = train_model.entrypoint(
        dataset_chips=str(toy_chips),
        dataset_labels=str(toy_labels),
        base_model_weights=pretrained_weights,
        hyperparameters=hp,
        split_info=info,
        num_classes=2,
    )
    assert model is not None
    assert hasattr(model, "parameters")
    assert next(model.parameters()).device.type == "cpu"


def test_evaluate_model(
    toy_chips: Path,
    toy_labels: Path,
    base_hyperparameters: dict[str, Any],
    pretrained_weights: str,
) -> None:
    from models.dinov3s_buildings.pipeline import evaluate_model, split_dataset, train_model

    info = split_dataset.entrypoint(
        dataset_chips=str(toy_chips),
        dataset_labels=str(toy_labels),
        hyperparameters=base_hyperparameters,
    )
    hp = {**base_hyperparameters, "epochs": 1}
    model = train_model.entrypoint(
        dataset_chips=str(toy_chips),
        dataset_labels=str(toy_labels),
        base_model_weights=pretrained_weights,
        hyperparameters=hp,
        split_info=info,
        num_classes=2,
    )
    metrics = evaluate_model.entrypoint(
        trained_model=model,
        dataset_chips=str(toy_chips),
        dataset_labels=str(toy_labels),
        hyperparameters=hp,
        split_info=info,
        num_classes=2,
    )
    assert set(metrics) >= {
        "pixel_iou",
        "instance_precision",
        "instance_recall",
        "instance_f1",
        "pred_avg_vertices",
        "pred_orthogonality",
    }
    assert 0.0 <= metrics["pixel_iou"] <= 1.0
    assert 0.0 <= metrics["instance_f1"] <= 1.0
    assert 0.0 <= metrics["pred_orthogonality"] <= 1.0
    assert metrics["pred_avg_vertices"] >= 0.0


def test_tune_postprocess_skips_on_small_val(
    toy_chips: Path,
    toy_labels: Path,
    base_hyperparameters: dict[str, Any],
) -> None:
    """Toy fixture is 6 chips; the under-8 guard returns v5s defaults without touching the model."""
    from models.dinov3s_buildings.pipeline import (
        DEFAULT_INFERENCE_PARAMS,
        split_dataset,
        tune_postprocess,
    )

    info = split_dataset.entrypoint(
        dataset_chips=str(toy_chips),
        dataset_labels=str(toy_labels),
        hyperparameters=base_hyperparameters,
    )
    recommended = tune_postprocess.entrypoint(
        trained_model=None,
        dataset_chips=str(toy_chips),
        dataset_labels=str(toy_labels),
        hyperparameters=base_hyperparameters,
        split_info=info,
    )
    assert recommended == DEFAULT_INFERENCE_PARAMS


def test_export_onnx(base_hyperparameters: dict[str, Any]) -> None:
    import onnx
    from dinov3_hot.config import load_config
    from dinov3_hot.model import build_model

    from models.dinov3s_buildings.pipeline import (
        AUX_IN_INDEX,
        BACKBONE_KEY,
        ENCODER_FILENAME,
        SEG_OUT_INDICES,
        export_onnx,
    )

    cfg = load_config(None)
    cfg.backbone = BACKBONE_KEY
    cfg.hf_ckpt_file = ENCODER_FILENAME
    cfg.seg_out_indices = list(SEG_OUT_INDICES)
    cfg.aux_in_index = AUX_IN_INDEX
    model = build_model(cfg).cpu().net
    onnx_bytes = export_onnx.entrypoint(
        trained_model=model,
        hyperparameters=base_hyperparameters,
        num_classes=2,
    )
    assert isinstance(onnx_bytes, bytes)
    loaded = onnx.load_from_string(onnx_bytes)
    assert len(loaded.graph.input) == 1
    assert len(loaded.graph.output) == 1
