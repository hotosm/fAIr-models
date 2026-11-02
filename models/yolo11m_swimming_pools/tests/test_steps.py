"""Step tests for yolo11m-swimming-pools.

Each test runs the real @step entrypoint on toy OAM chips and GeoJSON labels. The weights are a randomly initialised
YOLO11m with one class, built offline, so nothing is downloaded and accuracy is not under test. Telemetry sinks are
no-ops via models/conftest.py. A trained checkpoint and its ONNX are built once and reused, inside the first test that
needs them, so the instrumentation patches of that test apply.
"""

import io
import json
import tempfile
from pathlib import Path
from typing import Any

import pytest

_CACHE: dict[str, Any] = {}
METRIC_KEYS = {"precision", "recall", "f1", "map50", "map50_95"}


@pytest.fixture(scope="session")
def base_weights(tmp_path_factory: pytest.TempPathFactory) -> str:
    """A randomly initialised YOLO11m checkpoint with one class, in the file layout Ultralytics loads."""
    import torch
    from ultralytics.nn.tasks import DetectionModel

    torch.manual_seed(0)
    model = DetectionModel("yolo11m.yaml", nc=1, verbose=False)
    model.names = {0: "swimming_pool"}
    path = tmp_path_factory.mktemp("weights") / "random_yolo11m.pt"
    torch.save({"model": model, "train_args": {"imgsz": 512, "task": "detect"}}, path)
    return str(path)


def _inputs(toy_chips: Path) -> dict[str, str]:
    return {"image": str(toy_chips)}


def _split(toy_chips: Path, toy_labels: Path, hyperparameters: dict[str, Any]) -> dict[str, Any]:
    from models.yolo11m_swimming_pools.pipeline import split_dataset

    return split_dataset.entrypoint(
        dataset_chips=_inputs(toy_chips),
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
    from models.yolo11m_swimming_pools.pipeline import train_model

    return train_model.entrypoint(
        dataset_chips=_inputs(toy_chips),
        dataset_labels=str(toy_labels),
        base_model_weights=weights_path,
        hyperparameters=hyperparameters,
        split_info=_split(toy_chips, toy_labels, hyperparameters),
        num_classes=num_classes,
    )


def _trained(toy_chips: Path, toy_labels: Path, weights_path: str, hyperparameters: dict[str, Any]) -> bytes:
    if "trained" not in _CACHE:
        _CACHE["trained"] = _train(toy_chips, toy_labels, weights_path, hyperparameters)
    return _CACHE["trained"]


def _onnx(toy_chips: Path, toy_labels: Path, weights_path: str, hyperparameters: dict[str, Any]) -> bytes:
    from models.yolo11m_swimming_pools.pipeline import export_onnx

    if "onnx" not in _CACHE:
        trained = _trained(toy_chips, toy_labels, weights_path, hyperparameters)
        _CACHE["onnx"] = export_onnx.entrypoint(trained_model=trained, hyperparameters=hyperparameters)
    return _CACHE["onnx"]


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
    base_hyperparameters: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import torch
    from ultralytics import YOLO

    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    model_bytes = _trained(toy_chips, toy_labels, base_weights, base_hyperparameters)

    assert isinstance(model_bytes, bytes)
    # The step makes the dataset folder (chip links and labels), the base weights and the runs in a temporary folder of
    # its own and removes it: none of it is left behind, and nothing was passed to another step. (PyTorch keeps its own
    # compile cache in the temp folder, which is not ours.)
    ours = {"yolo", "runs", "base.pt", "best.pt", "data.yaml"}
    assert [path.name for path in scratch.rglob("*") if path.name in ours or path.suffix in {".tif", ".txt"}] == []
    trained_path = tmp_path / "trained.pt"
    trained_path.write_bytes(model_bytes)
    trained = YOLO(str(trained_path))
    assert trained.model.model[-1].nc == 1
    # The first 10 layers are frozen by default, so the first convolution is the base one (stored in half precision).
    base = torch.load(base_weights, map_location="cpu", weights_only=False)["model"].state_dict()
    key = "model.0.conv.weight"
    assert torch.allclose(trained.model.state_dict()[key].float(), base[key].half().float(), atol=1e-3)


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
    from models.yolo11m_swimming_pools.pipeline import evaluate_model

    trained = _trained(toy_chips, toy_labels, base_weights, base_hyperparameters)
    metrics = evaluate_model.entrypoint(
        trained_model=trained,
        dataset_chips=_inputs(toy_chips),
        dataset_labels=str(toy_labels),
        hyperparameters=base_hyperparameters,
        split_info=_split(toy_chips, toy_labels, base_hyperparameters),
    )

    assert set(metrics) == METRIC_KEYS
    assert all(0.0 <= value <= 1.0 for value in metrics.values())


def test_export_onnx(
    toy_chips: Path,
    toy_labels: Path,
    base_weights: str,
    base_hyperparameters: dict[str, Any],
) -> None:
    import onnx

    onnx_bytes = _onnx(toy_chips, toy_labels, base_weights, base_hyperparameters)

    assert isinstance(onnx_bytes, bytes)
    graph = onnx.load_from_string(onnx_bytes).graph
    onnx.checker.check_model(onnx.load_from_string(onnx_bytes))

    def dims(tensor: Any) -> list[int]:
        return [dim.dim_value for dim in tensor.type.tensor_type.shape.dim]

    assert graph.input[0].name == "images"
    assert dims(graph.input[0]) == [1, 3, 512, 512]
    assert dims(graph.output[0]) == [1, 5, 5376]


def test_predict_returns_pool_boxes_in_the_fair_schema(
    toy_chips: Path,
    toy_labels: Path,
    base_weights: str,
    base_hyperparameters: dict[str, Any],
) -> None:
    import onnxruntime as ort

    from models.yolo11m_swimming_pools.pipeline import predict

    onnx_bytes = _onnx(toy_chips, toy_labels, base_weights, base_hyperparameters)
    session = ort.InferenceSession(onnx_bytes, providers=["CPUExecutionProvider"])
    # Random weights score every anchor low, so keep every box to have features to inspect.
    collection = predict(session, _inputs(toy_chips), {"confidence_threshold": 0.0})

    assert collection["type"] == "FeatureCollection"
    assert collection["features"]
    chip_names = {path.name for path in toy_chips.glob("*.tif")}
    for feature in collection["features"]:
        assert feature["geometry"]["type"] == "Polygon"
        assert set(feature["properties"]) == {"class", "label", "confidence", "source"}
        assert feature["properties"]["class"] == 0
        assert feature["properties"]["label"] == "swimming_pool"
        assert 0.0 <= feature["properties"]["confidence"] <= 1.0
        assert feature["properties"]["source"] in chip_names


def test_run_inference_step(
    toy_chips: Path,
    toy_labels: Path,
    base_weights: str,
    base_hyperparameters: dict[str, Any],
    tmp_path: Path,
) -> None:
    from models.yolo11m_swimming_pools.pipeline import run_inference

    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(_onnx(toy_chips, toy_labels, base_weights, base_hyperparameters))
    collection = run_inference.entrypoint(
        model_uri=str(model_path),
        input_images=_inputs(toy_chips),
        inference_params={"confidence_threshold": 0.999},
    )

    assert collection["type"] == "FeatureCollection"
    assert isinstance(collection["features"], list)


def test_predict_rejects_a_bad_threshold(
    toy_chips: Path,
    toy_labels: Path,
    base_weights: str,
    base_hyperparameters: dict[str, Any],
) -> None:
    import onnxruntime as ort

    from models.yolo11m_swimming_pools.pipeline import predict

    session = ort.InferenceSession(
        _onnx(toy_chips, toy_labels, base_weights, base_hyperparameters), providers=["CPUExecutionProvider"]
    )
    with pytest.raises(ValueError, match="confidence_threshold"):
        predict(session, _inputs(toy_chips), {"confidence_threshold": 2.0})


def test_the_checkpoint_bytes_are_a_loadable_torch_file(
    toy_chips: Path, toy_labels: Path, base_weights: str, base_hyperparameters: dict[str, Any]
) -> None:
    import torch

    trained = _trained(toy_chips, toy_labels, base_weights, base_hyperparameters)
    checkpoint = torch.load(io.BytesIO(trained), map_location="cpu", weights_only=False)
    assert "model" in checkpoint or "ema" in checkpoint


def test_the_class_constants_match_the_stac_item() -> None:
    """`predict` cannot read the STAC at serving time, so its class value and label must equal the STAC's."""
    from models.yolo11m_swimming_pools import pipeline

    stac = json.loads((Path(pipeline.__file__).parent / "stac-item.json").read_text(encoding="utf-8"))
    for output in stac["properties"]["mlm:output"]:
        assert output["classification:classes"] == [
            {"name": pipeline.CLASS_LABEL, "value": pipeline.CLASS_VALUE, "description": "Swimming pool"}
        ]
