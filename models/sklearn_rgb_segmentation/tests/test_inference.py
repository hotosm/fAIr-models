"""Serve-path test: a trained model predicts building polygons on a toy chip."""

from pathlib import Path

from .test_steps import _split, _train


def test_predict_returns_building_polygons(generate_toy_dataset: dict[str, Path]) -> None:
    from onnxruntime import InferenceSession

    from models.sklearn_rgb_segmentation.pipeline import export_onnx, predict

    onnx_bytes = export_onnx.entrypoint(_train(generate_toy_dataset, _split(generate_toy_dataset)))
    session = InferenceSession(onnx_bytes, providers=["CPUExecutionProvider"])

    result = predict(session, str(generate_toy_dataset["chips"]), {"confidence_threshold": 0.5})

    assert result["type"] == "FeatureCollection"
    assert result["features"]
    assert all(f["properties"]["label"] == "building" for f in result["features"])
