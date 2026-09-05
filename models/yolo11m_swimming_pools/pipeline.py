"""ZenML pipeline for YOLO11m swimming pool detection.

Thin fAIr adapter over `spd-hot`:
- fAIr contract + ZenML orchestration live here
- model logic lives in `spd_hot.*`
"""

import tempfile
from pathlib import Path
from typing import Annotated, Any

from zenml import log_metadata, pipeline, step

from fair.utils.data import resolve_directory
from fair.zenml.instrumentation import log_evaluation_results, mlflow_training_context
from fair.zenml.materializers import CheckpointBytesMaterializer, ONNXMaterializer


def _download_checkpoint(url: str) -> Path:
    from upath import UPath

    local_path = Path(tempfile.mkdtemp()) / UPath(url).name
    local_path.write_bytes(UPath(url).read_bytes())
    return local_path


def _restore_checkpoint(trained_model: Any):
    from ultralytics import YOLO

    if isinstance(trained_model, YOLO):
        return trained_model
    if isinstance(trained_model, bytes):
        checkpoint = Path(tempfile.mkdtemp()) / "best.pt"
        checkpoint.write_bytes(trained_model)
        return YOLO(str(checkpoint))
    return YOLO(trained_model)


def _ensure_yolo_dataset(
    dataset_chips: str,
    dataset_labels: str,
    split_cfg,
    *,
    out_dir: str | Path | None,
) -> tuple[Path, dict[str, Any]]:
    from spd_hot.dataset import prepare_yolo_detection_dataset_from_geojson

    chips_dir = resolve_directory(dataset_chips, "*.tif*")
    labels_dir = resolve_directory(dataset_labels, "*.geojson")
    return prepare_yolo_detection_dataset_from_geojson(
        chips_dir,
        labels_dir,
        class_name="swimming_pool",
        val_ratio=split_cfg.val_ratio,
        seed=split_cfg.split_seed,
        block_size=split_cfg.block_size,
        out_dir=out_dir,
    )


def preprocess(image_path: Any) -> Any:
    """Contract hook for STAC `pre_processing_function`."""
    from spd_hot.preprocess import preprocess_chip_for_onnx

    batch, _meta = preprocess_chip_for_onnx(image_path)
    return batch


def postprocess(raw_output: Any, params: dict[str, Any] | None = None) -> Any:
    """Decode ONNX output to detections (confidence filter + NMS)."""
    from spd_hot.postprocess import postprocess_onnx_output

    return postprocess_onnx_output(raw_output, params)


def predict(session: Any, input_images: str, params: dict[str, Any]) -> dict[str, Any]:
    """Module-level serving entrypoint used by `fair.serve.base`."""
    from spd_hot.params import parse_postprocess_params, parse_preprocess_params
    from spd_hot.postprocess import detections_to_feature_collection
    from spd_hot.preprocess import preprocess_chip_for_onnx
    from spd_hot.serve import iter_image_paths

    chips_dir = resolve_directory(input_images)
    preprocess_cfg = parse_preprocess_params(params)
    post_cfg = parse_postprocess_params(params)
    input_name = session.get_inputs()[0].name

    all_features: list[dict[str, Any]] = []
    for img_path in iter_image_paths(chips_dir):
        batch, meta = preprocess_chip_for_onnx(img_path, preprocess_cfg)
        output = session.run(None, {input_name: batch})[0]
        detections = postprocess(output, params)
        fc = detections_to_feature_collection(
            detections,
            meta,
            post_cfg,
            source_name=img_path.name,
            class_name="swimming_pool",
        )
        all_features.extend(fc["features"])

    return {"type": "FeatureCollection", "features": all_features}


@step
def split_dataset(
    dataset_chips: str,
    dataset_labels: str,
    hyperparameters: dict[str, Any],
) -> Annotated[dict[str, Any], "split_info_artifact"]:
    from spd_hot.params import parse_split_params

    split_cfg = parse_split_params(hyperparameters)
    yolo_dir, split_info = _ensure_yolo_dataset(dataset_chips, dataset_labels, split_cfg, out_dir=None)
    split_info["description"] = "Spatial split by OAM tile blocks (fallback shuffle for non-OAM names)."
    split_info["_yolo_dir"] = str(yolo_dir)
    log_metadata(metadata={"fair/split": {k: v for k, v in split_info.items() if not k.startswith("_")}})
    return split_info


@step(output_materializers={"trained_model_artifact": CheckpointBytesMaterializer})
def train_model(
    dataset_chips: str,
    dataset_labels: str,
    base_model_weights: str,
    hyperparameters: dict[str, Any],
    split_info: dict[str, Any],
    num_classes: int = 1,
    model_name: str | None = None,
    base_model_id: str | None = None,
    dataset_id: str | None = None,
) -> Annotated[Any, "trained_model_artifact"]:
    from spd_hot.params import parse_split_params, parse_train_params
    from spd_hot.train import train_yolo

    del num_classes
    split_cfg = parse_split_params(hyperparameters)
    train_cfg = parse_train_params(hyperparameters)

    yolo_dir = Path(split_info["_yolo_dir"])
    data_yaml = yolo_dir / "data.yaml"
    if not data_yaml.exists():
        yolo_dir, _ = _ensure_yolo_dataset(dataset_chips, dataset_labels, split_cfg, out_dir=yolo_dir)
        data_yaml = yolo_dir / "data.yaml"

    local_weights = _download_checkpoint(base_model_weights)
    project_dir = Path(tempfile.mkdtemp()) / "train_runs"
    project_dir.mkdir(parents=True, exist_ok=True)

    with mlflow_training_context(hyperparameters, model_name, base_model_id, dataset_id):
        model = train_yolo(
            weights_path=str(local_weights),
            data_yaml_path=data_yaml,
            cfg=train_cfg,
            project_dir=project_dir,
            run_name="yolo11m_swimming_pools",
        )

    best_path = getattr(getattr(model, "trainer", None), "best", None)
    if best_path and Path(best_path).exists():
        return Path(best_path).read_bytes()

    saved_path = Path(tempfile.mkdtemp()) / "best.pt"
    model.save(str(saved_path))
    return saved_path.read_bytes()


@step
def evaluate_model(
    trained_model: Any,
    dataset_chips: str,
    dataset_labels: str,
    hyperparameters: dict[str, Any],
    split_info: dict[str, Any],
    class_names: list[str] | None = None,
) -> Annotated[dict[str, Any], "metrics"]:
    from spd_hot.evaluate import evaluate_yolo
    from spd_hot.params import parse_split_params, parse_train_params

    del class_names
    split_cfg = parse_split_params(hyperparameters)
    train_cfg = parse_train_params(hyperparameters)

    yolo_dir = Path(split_info["_yolo_dir"])
    data_yaml = yolo_dir / "data.yaml"
    if not data_yaml.exists():
        yolo_dir, _ = _ensure_yolo_dataset(dataset_chips, dataset_labels, split_cfg, out_dir=yolo_dir)
        data_yaml = yolo_dir / "data.yaml"

    model = _restore_checkpoint(trained_model)
    metrics = evaluate_yolo(model, str(data_yaml), imgsz=train_cfg.imgsz)
    log_evaluation_results(metrics)
    return metrics


@step(output_materializers={"onnx_model": ONNXMaterializer})
def export_onnx(trained_model: Any) -> Annotated[bytes, "onnx_model"]:
    from spd_hot.export import export_onnx_bytes

    model = _restore_checkpoint(trained_model)
    return export_onnx_bytes(model)


@step
def run_inference(
    model_uri: str,
    input_images: str,
    inference_params: dict[str, Any] | None = None,
) -> Annotated[dict[str, Any], "predictions"]:
    from spd_hot.params import parse_inference_params

    from fair.serve.base import load_session

    session = load_session(model_uri)
    params = inference_params or {}
    parse_inference_params(params)  # validate user-facing params early
    return predict(session, input_images, params)


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
    inference_params: dict[str, Any] | None = None,
) -> None:
    run_inference(
        model_uri=model_uri,
        input_images=input_images,
        inference_params=inference_params or {},
    )
