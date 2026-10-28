"""Step tests for deepforest-unet-tree.

Each test runs the real @step entrypoint on toy OAM chips and GeoJSON labels. Weights are random (a U-Net state dict, or
the RetinaNet state dict of the NEON_OAM encoder), so the tests are offline; accuracy is not under test. Telemetry sinks
are no-ops via models/conftest.py.
"""

import io
from pathlib import Path
from typing import Any

import pytest
import torch


def _to_bytes(state: dict) -> bytes:
    buffer = io.BytesIO()
    torch.save(state, buffer)
    return buffer.getvalue()


@pytest.fixture(scope="session")
def unet_checkpoint() -> bytes:
    from ts_hot.export import build_unet

    torch.manual_seed(0)
    return _to_bytes(build_unet().state_dict())


@pytest.fixture(scope="session")
def encoder_checkpoint() -> bytes:
    """A RetinaNet state dict with the keys of NEON_OAM (`backbone.body.*`, one class), randomly initialised."""
    from torchvision.models.detection import retinanet_resnet50_fpn

    torch.manual_seed(0)
    retina = retinanet_resnet50_fpn(weights=None, weights_backbone=None, num_classes=1)
    return _to_bytes(retina.state_dict())


@pytest.fixture(scope="session")
def base_weights(tmp_path_factory: pytest.TempPathFactory, unet_checkpoint: bytes) -> str:
    path = tmp_path_factory.mktemp("weights") / "random_unet.pt"
    path.write_bytes(unet_checkpoint)
    return str(path)


def _split(toy_chips: Path, toy_labels: Path, hyperparameters: dict[str, Any]) -> dict[str, Any]:
    from models.deepforest_unet_tree.pipeline import split_dataset

    return split_dataset.entrypoint(
        dataset_chips={"image": str(toy_chips)},
        dataset_labels=str(toy_labels),
        hyperparameters=hyperparameters,
    )


def _train(
    toy_chips: Path,
    toy_labels: Path,
    weights_path: str,
    hyperparameters: dict[str, Any],
    num_classes: int = 1,
) -> bytes:
    from models.deepforest_unet_tree.pipeline import train_model

    return train_model.entrypoint(
        dataset_chips={"image": str(toy_chips)},
        dataset_labels=str(toy_labels),
        base_model_weights=weights_path,
        hyperparameters=hyperparameters,
        split_info=_split(toy_chips, toy_labels, hyperparameters),
        num_classes=num_classes,
    )


def test_split_dataset(toy_chips: Path, toy_labels: Path, base_hyperparameters: dict[str, Any]) -> None:
    info = _split(toy_chips, toy_labels, base_hyperparameters)

    assert info["strategy"] == "spatial"
    assert 0 < info["val_ratio"] < 1
    assert info["train_count"] > 0
    assert info["val_count"] > 0
    assert info["train_count"] + info["val_count"] == len(list(toy_chips.glob("*.tif")))
    assert set(info["train_chip_names"]).isdisjoint(info["val_chip_names"])
    assert _split(toy_chips, toy_labels, base_hyperparameters) == info


def test_train_model(
    toy_chips: Path,
    toy_labels: Path,
    base_weights: str,
    unet_checkpoint: bytes,
    base_hyperparameters: dict[str, Any],
) -> None:
    model_bytes = _train(toy_chips, toy_labels, base_weights, base_hyperparameters)

    assert isinstance(model_bytes, bytes)
    trained = torch.load(io.BytesIO(model_bytes), map_location="cpu", weights_only=True)
    base = torch.load(io.BytesIO(unet_checkpoint), map_location="cpu", weights_only=True)
    assert trained.keys() == base.keys()
    # The encoder is frozen by default; the decoder head must have moved.
    assert torch.equal(trained["encoder.layer4.2.conv3.weight"], base["encoder.layer4.2.conv3.weight"])
    assert not torch.equal(trained["head.weight"], base["head.weight"])


def test_train_model_starts_the_encoder_from_neon_oam(
    toy_chips: Path,
    toy_labels: Path,
    encoder_checkpoint: bytes,
    tmp_path: Path,
    base_hyperparameters: dict[str, Any],
) -> None:
    weights = tmp_path / "neon_oam.pt"
    weights.write_bytes(encoder_checkpoint)
    model_bytes = _train(toy_chips, toy_labels, str(weights), base_hyperparameters)

    trained = torch.load(io.BytesIO(model_bytes), map_location="cpu", weights_only=True)
    retina = torch.load(io.BytesIO(encoder_checkpoint), map_location="cpu", weights_only=True)
    # A RetinaNet checkpoint is recognised by its missing `up_blocks`, and its ResNet-50 body is the frozen encoder.
    assert torch.equal(trained["encoder.conv1.weight"], retina["backbone.body.conv1.weight"])
    assert torch.equal(trained["encoder.layer1.0.conv1.weight"], retina["backbone.body.layer1.0.conv1.weight"])


def test_train_model_requires_single_class(
    toy_chips: Path,
    toy_labels: Path,
    base_weights: str,
    base_hyperparameters: dict[str, Any],
) -> None:
    with pytest.raises(ValueError, match="single-class"):
        _train(toy_chips, toy_labels, base_weights, base_hyperparameters, num_classes=2)


def test_evaluate_model(
    toy_chips: Path,
    toy_labels: Path,
    base_weights: str,
    base_hyperparameters: dict[str, Any],
) -> None:
    from models.deepforest_unet_tree.pipeline import evaluate_model

    model_bytes = _train(toy_chips, toy_labels, base_weights, base_hyperparameters)
    metrics = evaluate_model.entrypoint(
        trained_model=model_bytes,
        dataset_chips={"image": str(toy_chips)},
        dataset_labels=str(toy_labels),
        hyperparameters=base_hyperparameters,
        split_info=_split(toy_chips, toy_labels, base_hyperparameters),
    )

    assert set(metrics) == {"precision", "recall", "f1", "iou"}
    assert all(0.0 <= value <= 1.0 for value in metrics.values())


def test_export_onnx(unet_checkpoint: bytes, base_hyperparameters: dict[str, Any]) -> None:
    import numpy as np
    import onnx
    import onnxruntime as ort

    from models.deepforest_unet_tree.pipeline import export_onnx

    onnx_bytes = export_onnx.entrypoint(trained_model=unet_checkpoint, hyperparameters=base_hyperparameters)

    assert isinstance(onnx_bytes, bytes)
    proto = onnx.load_from_string(onnx_bytes)
    onnx.checker.check_model(proto)
    assert [i.name for i in proto.graph.input] == ["image"]
    assert [o.name for o in proto.graph.output] == ["probability"]

    session = ort.InferenceSession(onnx_bytes, providers=["CPUExecutionProvider"])
    for height, width in ((96, 128), (64, 160)):  # dynamic height and width
        (probability,) = session.run(None, {"image": np.zeros((3, height, width), dtype=np.float32)})
        assert probability.shape == (height, width)


def test_predict_declares_every_feature_variable(
    toy_chips: Path,
    unet_checkpoint: bytes,
) -> None:
    import onnxruntime as ort

    from models.deepforest_unet_tree.pipeline import export_onnx, predict

    onnx_bytes = export_onnx.entrypoint(trained_model=unet_checkpoint, hyperparameters={})
    session = ort.InferenceSession(onnx_bytes, providers=["CPUExecutionProvider"])
    # A zero threshold makes every pixel tree, so each chip gives at least one polygon whatever the random weights.
    collection = predict(session, {"image": str(toy_chips)}, {"threshold": 0.0, "model_gsd_m": 0.3})

    assert collection["type"] == "FeatureCollection"
    assert collection["features"]
    for feature in collection["features"]:
        assert feature["geometry"]["type"] == "Polygon"
        assert set(feature["properties"]) == {"class", "label", "confidence", "source"}
        assert feature["properties"]["class"] == 1 and feature["properties"]["label"] == "tree"
