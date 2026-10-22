"""Step tests for deepforest-tree.

Each test runs the real @step entrypoint against toy OAM chips + GeoJSON labels. Weights are a random-init
RetinaNet state dict, so the tests are offline; accuracy is not under test. Telemetry sinks (zenml/mlflow)
are no-ops via models/conftest.py.
"""

import io
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from td_hot.export import build_retinanet

HYPERPARAMETERS: dict[str, Any] = {"epochs": 1, "batch_size": 2}


@pytest.fixture(scope="session")
def random_checkpoint() -> bytes:
    torch.manual_seed(0)
    buffer = io.BytesIO()
    torch.save(build_retinanet().state_dict(), buffer)
    return buffer.getvalue()


@pytest.fixture(scope="session")
def base_weights(tmp_path_factory: pytest.TempPathFactory, random_checkpoint: bytes) -> str:
    path = tmp_path_factory.mktemp("weights") / "random_init.pt"
    path.write_bytes(random_checkpoint)
    return str(path)


@pytest.fixture(scope="session")
def split_info(toy_chips: Path, toy_labels: Path) -> dict[str, Any]:
    from models.deepforest_tree.pipeline import split_dataset

    return split_dataset.entrypoint(
        dataset_chips=str(toy_chips), dataset_labels=str(toy_labels), hyperparameters=HYPERPARAMETERS
    )


@pytest.fixture(scope="session")
def trained_model(toy_chips: Path, toy_labels: Path, base_weights: str, split_info: dict[str, Any]) -> bytes:
    from models.deepforest_tree.pipeline import train_model

    return train_model.entrypoint(
        dataset_chips=str(toy_chips),
        dataset_labels=str(toy_labels),
        base_model_weights=base_weights,
        hyperparameters=HYPERPARAMETERS,
        split_info=split_info,
        num_classes=1,
    )


def test_split_dataset(toy_chips: Path, toy_labels: Path) -> None:
    from models.deepforest_tree.pipeline import split_dataset

    kwargs = {
        "dataset_chips": str(toy_chips),
        "dataset_labels": str(toy_labels),
        "hyperparameters": HYPERPARAMETERS,
    }
    info = split_dataset.entrypoint(**kwargs)

    assert info["strategy"] == "spatial"
    assert 0 < info["val_ratio"] < 1
    assert info["train_count"] > 0
    assert info["val_count"] > 0
    assert info["train_count"] + info["val_count"] == len(list(toy_chips.glob("*.tif")))
    assert len(info["train_chip_names"]) == info["train_count"]
    assert len(info["val_chip_names"]) == info["val_count"]
    assert set(info["train_chip_names"]).isdisjoint(info["val_chip_names"])
    assert {"seed", "description"} <= set(info)
    assert split_dataset.entrypoint(**kwargs) == info


def test_train_model(trained_model: bytes, random_checkpoint: bytes) -> None:
    assert isinstance(trained_model, bytes)
    trained = torch.load(io.BytesIO(trained_model), map_location="cpu", weights_only=True)
    base = torch.load(io.BytesIO(random_checkpoint), map_location="cpu", weights_only=True)
    assert trained.keys() == base.keys()
    # The encoder is frozen by default; the head must have moved.
    assert torch.equal(trained["backbone.body.layer4.2.conv3.weight"], base["backbone.body.layer4.2.conv3.weight"])
    assert not torch.equal(
        trained["head.classification_head.cls_logits.weight"], base["head.classification_head.cls_logits.weight"]
    )


def test_train_model_requires_single_class(
    toy_chips: Path, toy_labels: Path, base_weights: str, split_info: dict[str, Any]
) -> None:
    from models.deepforest_tree.pipeline import train_model

    with pytest.raises(ValueError, match="single-class"):
        train_model.entrypoint(
            dataset_chips=str(toy_chips),
            dataset_labels=str(toy_labels),
            base_model_weights=base_weights,
            hyperparameters=HYPERPARAMETERS,
            split_info=split_info,
            num_classes=2,
        )


def test_evaluate_model(toy_chips: Path, toy_labels: Path, trained_model: bytes, split_info: dict[str, Any]) -> None:
    from models.deepforest_tree.pipeline import evaluate_model

    metrics = evaluate_model.entrypoint(
        trained_model=trained_model,
        dataset_chips=str(toy_chips),
        dataset_labels=str(toy_labels),
        hyperparameters=HYPERPARAMETERS,
        split_info=split_info,
    )

    assert set(metrics) == {"precision", "recall", "f1"}
    assert all(0.0 <= value <= 1.0 for value in metrics.values())


def test_export_onnx(random_checkpoint: bytes) -> None:
    import onnx
    import onnxruntime as ort

    from models.deepforest_tree.pipeline import export_onnx

    onnx_bytes = export_onnx.entrypoint(trained_model=random_checkpoint, hyperparameters=HYPERPARAMETERS)

    assert isinstance(onnx_bytes, bytes)
    proto = onnx.load_from_string(onnx_bytes)
    onnx.checker.check_model(proto)
    assert [i.name for i in proto.graph.input] == ["image"]
    assert [o.name for o in proto.graph.output] == ["boxes", "scores", "labels"]

    session = ort.InferenceSession(onnx_bytes, providers=["CPUExecutionProvider"])
    for height, width in ((96, 128), (64, 160)):  # dynamic height and width
        boxes, scores, labels = session.run(None, {"image": np.zeros((3, height, width), dtype=np.float32)})
        assert boxes.ndim == 2 and boxes.shape[1] == 4
        assert scores.shape == labels.shape == (len(boxes),)
        assert np.all(np.diff(scores) <= 0)  # best score first
