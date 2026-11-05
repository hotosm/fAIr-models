"""End-to-end tests for the four pipeline stages on toy data.

The toy chips are dark with a bright east-west band that the polygon labels mark as a road.
A small untrained-encoder U-Net keeps the tests fast and offline on CPU.
"""

from pathlib import Path
from typing import Any

HYPERPARAMETERS: dict[str, Any] = {
    "val_ratio": 0.25,
    "split_seed": 42,
    "block_size": 1,
    "epochs": 2,
    "batch_size": 2,
    "learning_rate": 1e-3,
    "encoder_name": "resnet18",
    "encoder_weights": None,
    "road_width_m": 6.0,
}


def _split(dataset: dict[str, Any]) -> dict[str, Any]:
    from models.unet_roads.pipeline import split_dataset

    return split_dataset.entrypoint(
        dataset_chips={"image": str(dataset["chips"])},
        dataset_labels=str(dataset["labels"]),
        hyperparameters=HYPERPARAMETERS,
    )


def _train(dataset: dict[str, Any], split_info: dict[str, Any]) -> bytes:
    from models.unet_roads.pipeline import train_model

    return train_model.entrypoint(
        dataset_chips={"image": str(dataset["chips"])},
        dataset_labels=str(dataset["labels"]),
        base_model_weights="",
        hyperparameters=HYPERPARAMETERS,
        split_info=split_info,
        num_classes=2,
    )


def test_split_dataset(generate_toy_dataset: dict[str, Path]) -> None:
    info = _split(generate_toy_dataset)
    assert info["strategy"] == "spatial"
    assert info["train_count"] > 0
    assert info["val_count"] > 0
    assert set(info["train_chip_names"]).isdisjoint(info["val_chip_names"])
    for key in ("val_ratio", "seed", "description"):
        assert key in info


def test_train_model(generate_toy_dataset: dict[str, Path]) -> None:
    from models.unet_roads.pipeline import _load_model

    model, encoder_name = _load_model(_train(generate_toy_dataset, _split(generate_toy_dataset)))
    assert encoder_name == "resnet18"
    assert not model.training


def test_train_model_starts_from_checkpoint_file(generate_toy_dataset: dict[str, Path], tmp_path: Path) -> None:
    from models.unet_roads.pipeline import _load_model, train_model

    split_info = _split(generate_toy_dataset)
    checkpoint = tmp_path / "base.pt"
    checkpoint.write_bytes(_train(generate_toy_dataset, split_info))

    # fAIr hands the base weights to the step as a file reference, not as bytes.
    finetuned = train_model.entrypoint(
        dataset_chips={"image": str(generate_toy_dataset["chips"])},
        dataset_labels=str(generate_toy_dataset["labels"]),
        base_model_weights=str(checkpoint),
        # The encoder comes from the checkpoint, whatever the hyperparameters ask for.
        hyperparameters={**HYPERPARAMETERS, "encoder_name": "resnet34"},
        split_info=split_info,
        num_classes=2,
    )
    _, encoder_name = _load_model(finetuned)
    assert encoder_name == "resnet18"


def test_evaluate_model(generate_toy_dataset: dict[str, Path]) -> None:
    from models.unet_roads.pipeline import evaluate_model

    split_info = _split(generate_toy_dataset)
    metrics = evaluate_model.entrypoint(
        trained_model=_train(generate_toy_dataset, split_info),
        dataset_chips={"image": str(generate_toy_dataset["chips"])},
        dataset_labels=str(generate_toy_dataset["labels"]),
        hyperparameters=HYPERPARAMETERS,
        split_info=split_info,
    )
    assert set(metrics) == {"iou_road", "relaxed_precision", "relaxed_recall", "relaxed_f1"}
    assert all(0.0 <= value <= 1.0 for value in metrics.values())


def test_export_onnx(generate_toy_dataset: dict[str, Path]) -> None:
    import numpy as np
    from onnxruntime import InferenceSession

    from models.unet_roads.pipeline import MODEL_INPUT_SIZE, export_onnx

    onnx_bytes = export_onnx.entrypoint(_train(generate_toy_dataset, _split(generate_toy_dataset)))
    session = InferenceSession(onnx_bytes, providers=["CPUExecutionProvider"])
    image = np.zeros((1, 3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), dtype=np.float32)
    (logits,) = session.run(None, {session.get_inputs()[0].name: image})
    assert np.asarray(logits).shape == (1, 1, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)
