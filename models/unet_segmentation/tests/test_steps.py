"""Step tests for UNet building segmentation pipeline.

split_dataset is metadata-only (no geo deps needed).
train_model, evaluate_model, export_onnx require torchgeo GeoDatasets with real
GeoTIFF data, so they build a minimal model directly rather than going through
the full torchgeo data loading path.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import patch


def _make_unet_model(num_classes: int = 2) -> Any:
    """Build a minimal UNet model without loading geo data."""
    from torchgeo.models import unet

    return unet(weights=None, classes=num_classes).cpu()


def test_split_dataset(toy_chips: Path, toy_labels: Path, base_hyperparameters: dict[str, Any]) -> None:
    from models.unet_segmentation.pipeline import split_dataset

    with patch("models.unet_segmentation.pipeline.log_metadata"):
        result = split_dataset.entrypoint(
            dataset_chips=str(toy_chips),
            dataset_labels=str(toy_labels),
            hyperparameters=base_hyperparameters,
        )

    assert result["strategy"] == "spatial"
    assert result["train_count"] > 0
    assert result["val_count"] > 0


def test_train_model(toy_chips: Path, toy_labels: Path, base_hyperparameters: dict[str, Any]) -> None:
    """Verify train_model produces a model.

    Builds a mock _build_dataset to avoid requiring real GeoTIFF data.
    The actual training loop is still exercised with mock tensors.
    """
    import torch

    from models.unet_segmentation.pipeline import split_dataset, train_model

    with patch("models.unet_segmentation.pipeline.log_metadata"):
        si = split_dataset.entrypoint(
            dataset_chips=str(toy_chips),
            dataset_labels=str(toy_labels),
            hyperparameters=base_hyperparameters,
        )

    chip_sz = base_hyperparameters.get("chip_size", 32)
    dummy_images = torch.randn(4, 3, chip_sz, chip_sz)
    dummy_masks = torch.randint(0, 2, (4, 1, chip_sz, chip_sz))

    def mock_build_dataset(*_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        batch_images = torch.stack([dummy_images[i] for i in range(4)])
        batch_masks = torch.stack([dummy_masks[i] for i in range(4)])
        return [{"image": batch_images, "mask": batch_masks}]

    def mock_download_s3(_uri: str) -> Path:
        from torchgeo.models import unet as build_unet

        tmp = Path(tempfile.mkdtemp()) / "weights.pth"
        torch.save(build_unet(weights=None, classes=2).state_dict(), tmp)
        return tmp

    with (
        patch("models.unet_segmentation.pipeline._build_dataset", mock_build_dataset),
        patch("models.unet_segmentation.pipeline._download_s3", mock_download_s3),
        patch("models.unet_segmentation.pipeline.mlflow_training_context") as mock_ctx,
        patch("models.unet_segmentation.pipeline.log_metadata"),
        patch("mlflow.log_metric"),
    ):
        mock_ctx.return_value.__enter__ = lambda s: None
        mock_ctx.return_value.__exit__ = lambda s, *a: None

        model = train_model.entrypoint(
            dataset_chips=str(toy_chips),
            dataset_labels=str(toy_labels),
            base_model_weights="none",
            hyperparameters=base_hyperparameters,
            split_info=si,
            num_classes=2,
        )

    assert model is not None
    assert hasattr(model, "parameters")


def test_evaluate_model(toy_chips: Path, toy_labels: Path, base_hyperparameters: dict[str, Any]) -> None:
    """Verify evaluate_model returns expected metrics.

    Mocks _build_dataset to avoid GeoTIFF requirement.
    """
    import torch

    from models.unet_segmentation.pipeline import evaluate_model, split_dataset

    with patch("models.unet_segmentation.pipeline.log_metadata"):
        si = split_dataset.entrypoint(
            dataset_chips=str(toy_chips),
            dataset_labels=str(toy_labels),
            hyperparameters=base_hyperparameters,
        )

    chip_sz = base_hyperparameters.get("chip_size", 32)
    dummy_images = torch.randn(4, 3, chip_sz, chip_sz)
    dummy_masks = torch.randint(0, 2, (4, 1, chip_sz, chip_sz))

    def mock_build_dataset(*_args: Any, **_kwargs: Any):
        return [{"image": torch.stack(list(dummy_images)), "mask": torch.stack(list(dummy_masks))}]

    model = _make_unet_model(num_classes=2)

    with (
        patch("models.unet_segmentation.pipeline._build_dataset", mock_build_dataset),
        patch("models.unet_segmentation.pipeline.log_evaluation_results"),
    ):
        metrics = evaluate_model.entrypoint(
            trained_model=model,
            dataset_chips=str(toy_chips),
            dataset_labels=str(toy_labels),
            hyperparameters=base_hyperparameters,
            split_info=si,
            num_classes=2,
        )

    assert "accuracy" in metrics
    assert "mean_iou" in metrics
    assert "per_class_iou" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["mean_iou"] <= 1.0


def test_export_onnx(base_hyperparameters: dict[str, Any]) -> None:
    import onnx

    from models.unet_segmentation.pipeline import export_onnx

    model = _make_unet_model(num_classes=2)

    onnx_path = export_onnx.entrypoint(
        trained_model=model,
        hyperparameters=base_hyperparameters,
        num_classes=2,
    )

    assert Path(onnx_path).exists()
    loaded = onnx.load(onnx_path)
    assert len(loaded.graph.input) == 1
    assert len(loaded.graph.output) == 1
