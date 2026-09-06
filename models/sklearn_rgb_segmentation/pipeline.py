"""sklearn-rgb-segmentation: a minimal per-pixel RGB logistic-regression segmenter.

The smallest fAIr reference model. It trains in seconds on CPU with no deep-learning
dependencies, so contributors can watch the full train, export, serve and predict flow
on a small machine.
"""

import pickle
from pathlib import Path
from typing import Annotated, Any

from zenml import log_metadata, pipeline, step

from fair.utils.data import resolve_directory
from fair.zenml.instrumentation import log_evaluation_results, mlflow_training_context
from fair.zenml.materializers import CheckpointBytesMaterializer, ONNXMaterializer

MODEL_NAME = "sklearn-rgb-segmentation"
CLASS_NAMES = ("background", "building")


def _chip_paths(dataset_chips: str) -> list[Path]:
    return sorted(resolve_directory(dataset_chips, "*.tif*").rglob("*.tif"))


def _label_geoms(dataset_labels: str) -> list[Any]:
    """Building geometries from the dataset GeoJSON, assumed EPSG:4326."""
    import json

    from shapely.geometry import shape

    labels = resolve_directory(dataset_labels)
    label_file = labels if labels.is_file() else sorted(labels.rglob("*.geojson"))[0]
    data = json.loads(label_file.read_text())
    return [shape(f["geometry"]) for f in data.get("features", []) if f.get("geometry")]


def _pixels_and_labels(chip_paths: list[Path], geoms: list[Any]) -> tuple[Any, Any]:
    """Stack every chip's RGB pixels (X) with rasterised 0/1 building labels (y)."""
    import numpy as np
    import rasterio
    from pyproj import Transformer
    from rasterio.features import rasterize
    from shapely.ops import transform as shapely_transform

    features, labels = [], []
    for chip_path in chip_paths:
        with rasterio.open(chip_path) as src:
            rgb = src.read([1, 2, 3]).astype(np.float32) / 255.0
            to_chip = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
            shapes = [shapely_transform(lambda x, y, _z=None, t=to_chip: t.transform(x, y), g) for g in geoms]
            mask = rasterize(
                [(g, 1) for g in shapes], out_shape=(src.height, src.width), transform=src.transform, dtype="uint8"
            )
        features.append(rgb.reshape(3, -1).T)
        labels.append(mask.reshape(-1))
    return np.concatenate(features), np.concatenate(labels)


def preprocess(image_path: Any) -> Any:
    """Read an RGB chip as normalised (pixels, 3) float32 features."""
    import numpy as np
    import rasterio

    with rasterio.open(image_path) as src:
        return (src.read([1, 2, 3]).astype(np.float32) / 255.0).reshape(3, -1).T


def postprocess(probabilities: Any, height: int, width: int, threshold: float) -> Any:
    """Reshape per-pixel building probabilities into a height x width uint8 mask."""
    import numpy as np

    return (np.asarray(probabilities).reshape(height, width) >= threshold).astype(np.uint8)


def predict(session: Any, input_images: str, params: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    import rasterio
    from pyproj import Transformer
    from rasterio.features import shapes
    from shapely.geometry import mapping, shape
    from shapely.ops import transform as shapely_transform

    threshold = float(params.get("confidence_threshold", 0.5))
    input_name = session.get_inputs()[0].name
    building_idx = CLASS_NAMES.index("building")

    input_path = resolve_directory(input_images)
    chip_paths = [input_path] if input_path.is_file() else _chip_paths(input_images)
    if not chip_paths:
        msg = f"No georeferenced (.tif) chips found in {input_path}"
        raise FileNotFoundError(msg)

    features: list[dict[str, Any]] = []
    for chip_path in chip_paths:
        with rasterio.open(chip_path) as src:
            rgb = src.read([1, 2, 3]).astype(np.float32) / 255.0
            transform = src.transform
            to_wgs84 = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
        height, width = rgb.shape[1], rgb.shape[2]
        probs = np.asarray(session.run(None, {input_name: rgb.reshape(3, -1).T})[-1])[:, building_idx]
        mask = postprocess(probs, height, width, threshold)
        for geom, _ in shapes(mask, mask=mask.astype(bool), transform=transform):
            polygon = shapely_transform(lambda x, y, _z=None, t=to_wgs84: t.transform(x, y), shape(geom))
            features.append({"type": "Feature", "properties": {"label": "building"}, "geometry": mapping(polygon)})
    return {"type": "FeatureCollection", "features": features}


@step
def split_dataset(
    dataset_chips: str,
    dataset_labels: str,
    hyperparameters: dict[str, Any],
) -> Annotated[dict[str, Any], "split_info"]:
    """Hold out whole chips for validation via sklearn's train_test_split."""
    from sklearn.model_selection import train_test_split

    names = [p.name for p in _chip_paths(dataset_chips)]
    train_names, val_names = train_test_split(
        names, test_size=hyperparameters.get("val_ratio", 0.2), random_state=hyperparameters.get("split_seed", 42)
    )
    info = {
        "strategy": "chip_holdout",
        "train_chip_names": train_names,
        "val_chip_names": val_names,
        "train_count": len(train_names),
        "val_count": len(val_names),
    }
    log_metadata(metadata={"fair/split": info})
    return info


@step(output_materializers={"trained_model_artifact": CheckpointBytesMaterializer})
def train_model(
    dataset_chips: str,
    dataset_labels: str,
    base_model_weights: str,
    hyperparameters: dict[str, Any],
    split_info: dict[str, Any],
    num_classes: int = 2,
    model_name: str | None = None,
    base_model_id: str | None = None,
    dataset_id: str | None = None,
) -> Annotated[bytes, "trained_model_artifact"]:
    """Fit a standardised logistic-regression pixel classifier on the training chips."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    train = set(split_info["train_chip_names"])
    chips = [p for p in _chip_paths(dataset_chips) if p.name in train]
    x, y = _pixels_and_labels(chips, _label_geoms(dataset_labels))

    with mlflow_training_context(hyperparameters, model_name, base_model_id, dataset_id):
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=hyperparameters.get("max_iter", 200))).fit(
            x, y
        )
        log_metadata(metadata={"train_accuracy": float(model.score(x, y)), "train_pixel_count": len(y)})
    return pickle.dumps(model)


@step
def evaluate_model(
    trained_model: bytes,
    dataset_chips: str,
    dataset_labels: str,
    hyperparameters: dict[str, Any],
    split_info: dict[str, Any],
    class_names: list[str] | None = None,
) -> Annotated[dict[str, Any], "metrics"]:
    """Report building-class IoU on the held-out chips via sklearn's jaccard_score."""
    from sklearn.metrics import jaccard_score

    model = pickle.loads(trained_model)
    val = set(split_info["val_chip_names"])
    chips = [p for p in _chip_paths(dataset_chips) if p.name in val]
    x, y = _pixels_and_labels(chips, _label_geoms(dataset_labels))

    metrics = {"building_iou": float(jaccard_score(y, model.predict(x), pos_label=1, zero_division=0))}
    log_evaluation_results(metrics)
    return metrics


@step(output_materializers={"onnx_model": ONNXMaterializer})
def export_onnx(trained_model: bytes) -> Annotated[bytes, "onnx_model"]:
    """Convert the fitted pipeline to ONNX with a dynamic pixel-batch axis."""
    from skl2onnx import to_onnx
    from skl2onnx.common.data_types import FloatTensorType

    model = pickle.loads(trained_model)
    # zipmap off keeps the probability output a plain (pixels, classes) array for the serve path.
    onnx_model = to_onnx(
        model,
        initial_types=[("input", FloatTensorType([None, 3]))],
        options={"zipmap": False},
    )
    return onnx_model.SerializeToString()


@step
def run_inference(
    model_uri: str,
    input_images: str,
    inference_params: dict[str, Any],
) -> Annotated[dict[str, Any], "predictions"]:
    from fair.serve.base import load_session

    return predict(load_session(model_uri), input_images, inference_params)


@pipeline
def training_pipeline(
    base_model_weights: str,
    dataset_chips: str,
    dataset_labels: str,
    num_classes: int,
    hyperparameters: dict[str, Any],
) -> None:
    split_info = split_dataset(
        dataset_chips=dataset_chips,
        dataset_labels=dataset_labels,
        hyperparameters=hyperparameters,
    )
    trained_model = train_model(
        dataset_chips=dataset_chips,
        dataset_labels=dataset_labels,
        base_model_weights=base_model_weights,
        hyperparameters=hyperparameters,
        split_info=split_info,
        num_classes=num_classes,
    )
    evaluate_model(
        trained_model=trained_model,
        dataset_chips=dataset_chips,
        dataset_labels=dataset_labels,
        hyperparameters=hyperparameters,
        split_info=split_info,
    )
    export_onnx(trained_model=trained_model)


@pipeline
def inference_pipeline(
    model_uri: str,
    input_images: str,
    inference_params: dict[str, Any],
) -> None:
    run_inference(model_uri=model_uri, input_images=input_images, inference_params=inference_params)
