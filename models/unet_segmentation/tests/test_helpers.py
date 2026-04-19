from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from rasterio.transform import from_origin


@pytest.mark.parametrize(
    ("mps_available", "cuda_available", "expected"),
    [
        (True, False, "mps"),
        (False, True, "cuda"),
        (False, False, "cpu"),
    ],
)
def test_get_device_prefers_available_accelerator(
    monkeypatch: pytest.MonkeyPatch,
    mps_available: bool,
    cuda_available: bool,
    expected: str,
) -> None:
    import torch

    from models.unet_segmentation.pipeline import _get_device

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps_available)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)

    assert _get_device() == expected


def test_segmentation_helpers_cover_preprocess_and_postprocess() -> None:
    import torch

    from models.unet_segmentation.pipeline import _build_feature_collection, _softmax, postprocess, preprocess

    batch = {
        "image": torch.full((1, 3, 2, 2), 255.0),
        "mask": torch.ones((1, 1, 2, 2), dtype=torch.int64),
    }
    images, masks = preprocess(batch)
    assert images.max().item() == 1.0
    assert masks.shape == (1, 2, 2)

    logits = torch.tensor([[[[0.0, 3.0], [5.0, 1.0]], [[2.0, 1.0], [1.0, 4.0]]]])
    prediction = postprocess(logits)
    assert prediction.dtype == np.uint8
    assert prediction.shape == (1, 2, 2)

    probs = _softmax(np.array([[1.0, 2.0], [3.0, 4.0]]), axis=0)
    assert np.allclose(probs.sum(axis=0), np.array([1.0, 1.0]))

    feature_collection = _build_feature_collection([{"type": "Feature"}])
    assert feature_collection["type"] == "FeatureCollection"
    assert len(feature_collection["features"]) == 1


def test_vectorize_mask_download_checkpoint_and_predict_errors(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from models.unet_segmentation.pipeline import _download_checkpoint, _vectorize_segmentation_mask, predict

    mask = np.array([[0, 1], [1, 1]], dtype=np.uint8)
    transform = from_origin(0, 2, 1, 1)
    features = _vectorize_segmentation_mask(mask, transform, None)
    assert features
    assert features[0]["properties"]["class"] == 1

    class DummyUPath:
        def __init__(self, path: str) -> None:
            self.path = path

        @property
        def name(self) -> str:
            return "weights.bin"

        def read_bytes(self) -> bytes:
            return b"weights"

    import upath

    monkeypatch.setattr(upath, "UPath", DummyUPath)
    checkpoint = _download_checkpoint("s3://bucket/weights.bin")
    assert checkpoint.exists()
    assert checkpoint.read_bytes() == b"weights"

    class FakeSession:
        def get_inputs(self) -> list[Any]:
            return [type("Input", (), {"name": "images"})()]

    with pytest.raises(ValueError):
        predict(FakeSession(), str(tmp_path), {})

    with pytest.raises(FileNotFoundError):
        predict(FakeSession(), str(tmp_path), {"confidence_threshold": 0.5})


def test_build_dataset_resize_and_run_inference_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import rasterio

    from models.unet_segmentation.pipeline import _build_dataset, _preprocess_onnx_image, run_inference

    captured: dict[str, Any] = {}

    class FakeRasterDataset:
        def __init__(self, *, paths: str) -> None:
            self.paths = paths
            self.res = (1.0, 1.0)

        def __and__(self, other: Any) -> str:
            return "merged-dataset"

    class FakeVectorDataset:
        def __init__(self, **kwargs: Any) -> None:
            captured["vector_kwargs"] = kwargs

    class FakeGridGeoSampler:
        def __init__(self, dataset: Any, *, size: int, stride: int, units: Any) -> None:
            self.kind = "grid"
            self.dataset = dataset
            self.size = size
            self.stride = stride
            self.units = units

    class FakeRandomGeoSampler:
        def __init__(self, dataset: Any, *, size: int, length: int, units: Any, generator: Any) -> None:
            self.kind = "random"
            self.dataset = dataset
            self.size = size
            self.length = length
            self.units = units
            self.generator = generator

    class FakeUnits:
        PIXELS = "pixels"

    def fake_dataloader(dataset: Any, *, sampler: Any, batch_size: int, collate_fn: Any) -> dict[str, Any]:
        return {
            "dataset": dataset,
            "sampler": sampler,
            "batch_size": batch_size,
            "collate_fn": collate_fn,
        }

    import torch.utils.data as data_utils
    import torchgeo.datasets as torchgeo_datasets
    import torchgeo.samplers as torchgeo_samplers

    import fair.utils.data as data_module

    monkeypatch.setattr(data_module, "resolve_directory", lambda path, pattern=None: Path(path))
    monkeypatch.setattr(data_utils, "DataLoader", fake_dataloader)
    monkeypatch.setattr(torchgeo_datasets, "RasterDataset", FakeRasterDataset)
    monkeypatch.setattr(torchgeo_datasets, "VectorDataset", FakeVectorDataset)
    monkeypatch.setattr(torchgeo_datasets, "stack_samples", "stacked")
    monkeypatch.setattr(torchgeo_samplers, "GridGeoSampler", FakeGridGeoSampler)
    monkeypatch.setattr(torchgeo_samplers, "RandomGeoSampler", FakeRandomGeoSampler)
    monkeypatch.setattr(torchgeo_samplers, "Units", FakeUnits)

    chips_dir = tmp_path / "chips"
    labels_dir = tmp_path / "labels"
    chips_dir.mkdir()
    labels_dir.mkdir()

    val_loader = _build_dataset(str(chips_dir), str(labels_dir), chip_size=32, length=5, split="val")
    train_loader = _build_dataset(str(chips_dir), str(labels_dir), chip_size=32, length=7, split="train", seed=9)

    assert val_loader["sampler"].kind == "grid"
    assert train_loader["sampler"].kind == "random"
    assert train_loader["sampler"].length == 7
    assert captured["vector_kwargs"]["label_name"] == "label"

    tif_path = tmp_path / "chip.tif"
    with rasterio.open(
        tif_path,
        "w",
        driver="GTiff",
        height=16,
        width=16,
        count=3,
        dtype="uint8",
        crs="EPSG:3857",
        transform=from_origin(0, 16, 1, 1),
    ) as dataset:
        dataset.write(np.ones((3, 16, 16), dtype=np.uint8))

    batch, transform, crs = _preprocess_onnx_image(tif_path)
    assert batch.shape == (1, 3, 256, 256)
    assert transform is not None
    assert crs is not None

    expected = {"type": "FeatureCollection", "features": []}
    monkeypatch.setattr("fair.serve.base.load_session", lambda model_uri: object())
    monkeypatch.setattr("models.unet_segmentation.pipeline.predict", lambda session, input_images, params: expected)

    result = run_inference.entrypoint(
        model_uri="https://example.com/model.onnx",
        input_images=str(tmp_path),
        inference_params={"confidence_threshold": 0.5},
    )
    assert result == expected
