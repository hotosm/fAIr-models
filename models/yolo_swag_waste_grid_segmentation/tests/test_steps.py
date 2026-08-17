"""High-level tests for the four waste-grid pipeline stages.

The toy data is deliberately easy: bright grid cells are waste and dark cells
are background.  Together, these tests describe the normal pipeline flow:
split data, train a classifier, evaluate it, then export and use it.
"""

from pathlib import Path
from typing import Any

import numpy as np


def test_split_dataset(
    toy_chips: Path,
    toy_labels: Path,
    base_hyperparameters: dict[str, Any],
) -> None:
    """Split the labelled grid cells into train, validation, and test folders."""
    from models.yolo_swag_waste_grid_segmentation.pipeline import CLASS_NAMES, split_dataset

    info = split_dataset.entrypoint(
        dataset_chips=str(toy_chips),
        dataset_labels=str(toy_labels),
        hyperparameters=base_hyperparameters,
    )

    assert info["strategy"] == "grid_5m_stratified_grouped"
    assert info["train_count"] > 0
    assert info["val_count"] > 0
    assert info["test_count"] > 0
    yolo_dir = Path(info["_yolo_dir"])
    for split in ("train", "val", "test"):
        for class_name in CLASS_NAMES:
            assert list((yolo_dir / split / class_name).glob("cell_*.png"))
    train_cells = {image.stem for image in (yolo_dir / "train").rglob("cell_*.png")}
    val_cells = {image.stem for image in (yolo_dir / "val").rglob("cell_*.png")}
    test_cells = {image.stem for image in (yolo_dir / "test").rglob("cell_*.png")}
    assert train_cells.isdisjoint(val_cells)
    assert train_cells.isdisjoint(test_cells)
    assert val_cells.isdisjoint(test_cells)


def test_train_model(
    toy_chips: Path,
    toy_labels: Path,
    base_hyperparameters: dict[str, Any],
    pretrained_weights: str,
) -> None:
    """Train from the public checkpoint and return a reloadable classifier."""
    from models.yolo_swag_waste_grid_segmentation.pipeline import (
        CLASS_NAMES,
        _restore_checkpoint,
        split_dataset,
        train_model,
    )

    split_info = split_dataset.entrypoint(
        dataset_chips=str(toy_chips),
        dataset_labels=str(toy_labels),
        hyperparameters=base_hyperparameters,
    )
    model_bytes = train_model.entrypoint(
        dataset_chips=str(toy_chips),
        dataset_labels=str(toy_labels),
        base_model_weights=pretrained_weights,
        hyperparameters=base_hyperparameters,
        split_info=split_info,
        num_classes=len(CLASS_NAMES),
    )

    assert model_bytes
    model = _restore_checkpoint(model_bytes)
    assert model.task == "classify"
    assert [model.names[index] for index in range(len(CLASS_NAMES))] == list(CLASS_NAMES)


def test_evaluate_model(
    toy_chips: Path,
    toy_labels: Path,
    base_hyperparameters: dict[str, Any],
    pretrained_weights: str,
) -> None:
    """Evaluate a valid classifier and return its accuracy metric."""
    from models.yolo_swag_waste_grid_segmentation.pipeline import evaluate_model, split_dataset

    split_info = split_dataset.entrypoint(
        dataset_chips=str(toy_chips),
        dataset_labels=str(toy_labels),
        hyperparameters=base_hyperparameters,
    )
    metrics = evaluate_model.entrypoint(
        trained_model=Path(pretrained_weights).read_bytes(),
        dataset_chips=str(toy_chips),
        dataset_labels=str(toy_labels),
        hyperparameters=base_hyperparameters,
        split_info=split_info,
    )

    assert set(metrics) == {"accuracy"}
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_export_onnx(
    pretrained_weights: str,
) -> None:
    """Export the public classifier and use its ONNX session for grid predictions."""
    import onnx
    import onnxruntime as ort

    from models.yolo_swag_waste_grid_segmentation.pipeline import export_onnx

    onnx_bytes = export_onnx.entrypoint(trained_model=Path(pretrained_weights).read_bytes())

    assert onnx_bytes
    model = onnx.load_from_string(onnx_bytes)
    assert len(model.graph.input) == 1
    assert len(model.graph.output) == 1

    session = ort.InferenceSession(onnx_bytes, providers=["CPUExecutionProvider"])
    output = session.run(
        None,
        {session.get_inputs()[0].name: np.zeros((1, 3, 128, 128), dtype=np.float32)},
    )[0]
    assert isinstance(output, np.ndarray)
    assert output.shape == (1, 2)
    assert np.isfinite(output).all()
