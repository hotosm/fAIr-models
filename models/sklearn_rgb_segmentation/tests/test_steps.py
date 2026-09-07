"""End-to-end tests for the four pipeline stages on toy data.

The toy chips are bright on their left half, which the labels mark as building, so
a pixel classifier can separate the two classes.
"""

import pickle
from pathlib import Path
from typing import Any

HYPERPARAMETERS: dict[str, Any] = {"val_ratio": 0.5, "split_seed": 42, "max_iter": 200}


def _split(dataset: dict[str, Path]) -> dict[str, Any]:
    from models.sklearn_rgb_segmentation.pipeline import split_dataset

    return split_dataset.entrypoint(
        dataset_chips=str(dataset["chips"]),
        dataset_labels=str(dataset["labels"]),
        hyperparameters=HYPERPARAMETERS,
    )


def _train(dataset: dict[str, Path], split_info: dict[str, Any]) -> bytes:
    from models.sklearn_rgb_segmentation.pipeline import train_model

    return train_model.entrypoint(
        dataset_chips=str(dataset["chips"]),
        dataset_labels=str(dataset["labels"]),
        base_model_weights="",
        hyperparameters=HYPERPARAMETERS,
        split_info=split_info,
        num_classes=2,
    )


def test_split_dataset(generate_toy_dataset: dict[str, Path]) -> None:
    info = _split(generate_toy_dataset)
    assert info["train_count"] > 0
    assert info["val_count"] > 0
    assert set(info["train_chip_names"]).isdisjoint(info["val_chip_names"])


def test_train_model(generate_toy_dataset: dict[str, Path]) -> None:
    model = pickle.loads(_train(generate_toy_dataset, _split(generate_toy_dataset)))
    assert hasattr(model, "predict")


def test_evaluate_model(generate_toy_dataset: dict[str, Path]) -> None:
    from models.sklearn_rgb_segmentation.pipeline import evaluate_model

    split_info = _split(generate_toy_dataset)
    metrics = evaluate_model.entrypoint(
        trained_model=_train(generate_toy_dataset, split_info),
        dataset_chips=str(generate_toy_dataset["chips"]),
        dataset_labels=str(generate_toy_dataset["labels"]),
        hyperparameters=HYPERPARAMETERS,
        split_info=split_info,
    )
    # The toy chips are colour-separable, so a working model must score a high IoU.
    assert metrics["building_iou"] > 0.9


def test_export_onnx(generate_toy_dataset: dict[str, Path]) -> None:
    import numpy as np
    from onnxruntime import InferenceSession

    from models.sklearn_rgb_segmentation.pipeline import export_onnx

    onnx_bytes = export_onnx.entrypoint(_train(generate_toy_dataset, _split(generate_toy_dataset)))
    session = InferenceSession(onnx_bytes, providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    outputs = session.run(None, {name: np.array([[0.9, 0.9, 0.9], [0.1, 0.1, 0.1]], dtype=np.float32)})
    probabilities = np.asarray(outputs[-1])
    assert probabilities.shape == (2, 2)
