"""Check the fAIr step contracts, dynamic ONNX export, and road geometry."""

import io
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnxruntime
import pytest
import rasterio
import segmentation_models_pytorch as smp
import torch
from rasterio.crs import CRS
from rasterio.transform import from_origin

from models.chesapeake_rsc_v2 import inference, pipeline
from models.chesapeake_rsc_v2.centerlines import probability_ridges, trace
from models.chesapeake_rsc_v2.training import spatial_split


@pytest.fixture(scope="module")
def split(toy_chips, toy_labels):
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(pipeline, "log_metadata", lambda **kwargs: None)
        return pipeline.split_dataset.entrypoint({"image": str(toy_chips)}, str(toy_labels), {})


@pytest.fixture(scope="module")
def trained(tmp_path_factory, toy_chips, toy_labels, split):
    import contextlib

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(pipeline, "mlflow_training_context", lambda *args: contextlib.nullcontext())
        import mlflow

        patch.setattr(mlflow, "log_metric", lambda *args, **kwargs: None)
        weights = os.environ.get("CHESAPEAKE_RGB_CHECKPOINT")
        if not weights:
            path = tmp_path_factory.mktemp("weights") / "checkpoint.pt"
            model = smp.Unet(encoder_name="resnet18", encoder_weights=None, in_channels=3, classes=2)
            torch.save(model.state_dict(), path)
            weights = str(path)
        return pipeline.train_model.entrypoint(
            {"image": str(toy_chips)},
            str(toy_labels),
            weights,
            {"epochs": 1, "batch_size": 2},
            split,
        )


def test_split_dataset(split):
    assert split["train_count"] == split["val_count"] == 2
    assert set(split["train_chip_names"]).isdisjoint(split["val_chip_names"])
    with pytest.raises(ValueError, match="at least two blocks"):
        spatial_split([Path("OAM-0-0-18.tif")], 0.2, 42, 4)


def test_train_model(trained):
    weights = torch.load(io.BytesIO(trained), weights_only=True)
    assert weights["encoder.conv1.weight"].shape[1] == 3


def test_evaluate_model(trained, toy_chips, toy_labels, split):
    metrics = pipeline.evaluate_model.entrypoint(trained, {"image": str(toy_chips)}, str(toy_labels), {}, split)
    assert set(metrics) == {"road_f1", "road_iou"}
    assert all(0 <= value <= 1 for value in metrics.values())


def test_export_onnx(trained):
    options = onnxruntime.SessionOptions()  # ty: ignore[possibly-missing-attribute]
    options.intra_op_num_threads = 1
    session = onnxruntime.InferenceSession(pipeline.export_onnx.entrypoint(trained), sess_options=options)
    for height, width in ((64, 96), (256, 256)):
        logits = session.run(None, {"image": np.zeros((1, 3, height, width), np.float32)})[0]
        assert isinstance(logits, np.ndarray)
        assert logits.shape == (1, 2, height, width)
        assert np.isfinite(logits).all()


def test_run_inference(toy_chips, tmp_path):
    for chip in sorted(toy_chips.glob("*.tif"))[:2]:
        (tmp_path / chip.name).symlink_to(chip)

    class Session:
        def get_inputs(self):
            return [SimpleNamespace(name="image")]

        def run(self, _, inputs):
            road = 10 * (inputs["image"][:, 0] - 0.5)
            return [np.stack((-road, road), axis=1)]

    result = pipeline.predict(Session(), {"image": str(tmp_path)}, {})
    assert len(result["features"]) == 1
    feature = result["features"][0]
    assert feature["properties"] == {"class": 1, "label": "road"}
    assert feature["geometry"]["type"] == "LineString"
    with rasterio.open(sorted(tmp_path.glob("*.tif"))[0]) as src:
        seam = src.bounds.right
    x = [lon for lon, _ in feature["geometry"]["coordinates"]]
    assert min(x) < seam < max(x)
    params = {"confidence_threshold": 1}
    assert pipeline.predict(Session(), {"image": str(tmp_path)}, params)["features"] == []


def test_centerline_junction():
    mask = np.zeros((32, 32), dtype=bool)
    mask[16, 2:30] = True
    mask[4:17, 16] = True
    mask[17:19, 16] = True
    features = trace(mask, from_origin(85, 27, 0.00001, 0.00001), CRS.from_epsg(4326))
    assert len(features) == 3
    ends = [
        tuple(point) for f in features for point in (f["geometry"]["coordinates"][0], f["geometry"]["coordinates"][-1])
    ]
    assert max(ends.count(point) for point in ends) == 3


def test_mosaic_resamples_at_training_gsd(tmp_path):
    path = tmp_path / "coarse.tif"
    transform = from_origin(500_000, 3_100_000, 1.2, 1.2)
    with rasterio.open(
        path, "w", driver="GTiff", width=4, height=4, count=3, dtype="uint8", crs="EPSG:32645", transform=transform
    ) as dst:
        dst.write(np.full((3, 4, 4), 80, dtype=np.uint8))
    rgb, valid, grid, _ = inference._mosaic([path])
    assert rgb.shape == (3, 8, 8)
    assert np.all(rgb == 80) and valid.all()
    assert grid.a == pytest.approx(0.6)
    assert grid.e == pytest.approx(-0.6)


def test_hann_blends_probabilities_before_threshold(monkeypatch):
    monkeypatch.setattr(inference, "PATCH_SIZE", 8)
    monkeypatch.setattr(inference, "STRIDE", 4)

    class Session:
        calls = 0

        def get_inputs(self):
            return [SimpleNamespace(name="image")]

        def run(self, _, inputs):
            self.calls += 1
            probability = 0.1 if self.calls == 1 else 0.7
            shape = inputs["image"].shape[-2:]
            road_logit = np.full(shape, np.log(probability / (1 - probability)), dtype=np.float32)
            return [np.stack((np.zeros_like(road_logit), road_logit))[None]]

    session = Session()
    valid = np.ones((8, 12), dtype=bool)
    valid[0, 0] = False
    scores = inference._blend_probabilities(session, np.zeros((3, 8, 12), dtype=np.uint8), valid)
    assert session.calls == 2
    assert scores[0, 0] == 0
    np.testing.assert_allclose(scores[1:, :4], 0.1, atol=1e-6)
    np.testing.assert_allclose(scores[:, 8:], 0.7, atol=1e-6)
    weights = np.maximum(np.outer(np.hanning(8), np.hanning(8)), 1e-3)
    expected = (0.1 * weights[3, 5] + 0.7 * weights[3, 1]) / (weights[3, 5] + weights[3, 1])
    assert scores[3, 5] == pytest.approx(expected)


def test_ridge_spans_processing_stripes():
    cols = np.arange(32, dtype=np.float32)
    profile = 0.18 * np.exp(-0.5 * ((cols - 16) / 2) ** 2)
    probability = np.broadcast_to(profile, (540, 32))
    valid = np.ones(probability.shape, dtype=bool)
    ridges = probability_ridges(probability, valid, 0.08)
    assert not (probability >= 0.25).any()
    assert ridges[1:-1, 16].all()
    assert ridges.sum() == 538


def test_centerlines_reproject_to_wgs84():
    mask = np.zeros((8, 8), dtype=bool)
    mask[4, 1:7] = True
    grid = from_origin(111319.49079327357, 111325.1428663851, 1, 1)
    features = trace(mask, grid, CRS.from_epsg(3857))
    coords = features[0]["geometry"]["coordinates"]
    assert all(1 < lon < 1.001 and 0.999 < lat < 1 for lon, lat in coords)
