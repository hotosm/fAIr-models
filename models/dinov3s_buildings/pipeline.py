"""dinov3s-buildings: frozen DINOv3-ViT-S/16 + UperNet decoder for binary buildings."""

import json
import tempfile
from pathlib import Path
from typing import Annotated, Any

from zenml import log_metadata, pipeline, step

from fair.utils.data import resolve_directory
from fair.zenml.instrumentation import log_evaluation_results, mlflow_training_context
from fair.zenml.materializers import ONNXMaterializer
from fair.zenml.metrics import log_loss_history

MODEL_NAME = "dinov3s-buildings"
BACKBONE_KEY = "terratorch_dinov3_vits16"
ENCODER_REPO = "kshitijrajsharma/dinov3"
ENCODER_FILENAME = "dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
SEG_OUT_INDICES = (2, 5, 8, 11)
AUX_IN_INDEX = 2

# Catalog defaults from `dinov3_hot.tune`; per-area fine-tune re-tunes via `tune_postprocess`.
DEFAULT_INFERENCE_PARAMS: dict[str, Any] = {
    "confidence_threshold": 0.4371,
    "seed_min_distance": 6,
    "simplify_m": 0.9626,
    "regularize_area_threshold": 0.4949,
    "regularize_overlap_tol_m2": 3.9251,
    "min_area_m2": 2.6465,
}


def predict(session: Any, input_images: str, params: dict[str, Any]) -> dict[str, Any]:
    """Module-level predict for `fair.serve.base`. Delegates to `predict_session`."""
    from dinov3_hot.serve import predict_session

    return predict_session(session, Path(resolve_directory(input_images)), params)


@step
def split_dataset(
    dataset_chips: str,
    dataset_labels: str,
    hyperparameters: dict[str, Any],
) -> Annotated[dict[str, Any], "split_info"]:
    """Spatially-stratified val split + per-channel mean/std for normalisation."""
    from dinov3_hot.dataset import compute_dataset_stats, spatial_split
    from dinov3_hot.serve import HOT_MEAN, HOT_STD

    chips_dir = resolve_directory(dataset_chips, "*.tif")
    chip_names = sorted(p.name for p in chips_dir.glob("*.tif"))

    val_ratio = hyperparameters.get("val_ratio", 0.2)
    seed = hyperparameters.get("split_seed", 42)
    block_size = hyperparameters.get("block_size", 4)
    train_chip_names, val_chip_names = spatial_split(chip_names, val_ratio, seed, block_size=block_size)

    norm_mode = hyperparameters.get("norm_stats", "hot_global")
    if norm_mode == "hot_global":
        mean, std = HOT_MEAN, HOT_STD
    elif norm_mode == "dataset":
        mean, std = compute_dataset_stats(chips_dir)
    else:
        raise ValueError(f"hyperparameters['norm_stats'] must be 'hot_global' or 'dataset', got {norm_mode!r}")

    info = {
        "strategy": "spatial",
        "val_ratio": val_ratio,
        "seed": seed,
        "block_size": block_size,
        "train_count": len(train_chip_names),
        "val_count": len(val_chip_names),
        "train_chip_names": train_chip_names,
        "val_chip_names": val_chip_names,
        "norm_source": norm_mode,
        "norm_mean": list(mean),
        "norm_std": list(std),
    }
    log_metadata(metadata={"fair/split": info})
    return info


@step
def train_model(
    dataset_chips: str,
    dataset_labels: str,
    base_model_weights: str,
    hyperparameters: dict[str, Any],
    split_info: dict[str, Any],
    num_classes: int,
    model_name: str | None = None,
    base_model_id: str | None = None,
    dataset_id: str | None = None,
) -> Annotated[Any, "trained_model_artifact"]:
    """Fine-tune the decoder via `dinov3_hot.finetune.finetune` on the train split."""
    from dinov3_hot.config import load_config
    from dinov3_hot.dataset import read_loss_history
    from dinov3_hot.finetune import finetune
    from dinov3_hot.model import DinoV3HotLit
    from dinov3_hot.paths import download_checkpoint, resolve_labels_geojson
    from huggingface_hub import hf_hub_download

    cfg = load_config(None)
    cfg.backbone = BACKBONE_KEY
    cfg.hf_ckpt_file = ENCODER_FILENAME
    cfg.seg_out_indices = list(SEG_OUT_INDICES)
    cfg.aux_in_index = AUX_IN_INDEX
    cfg.seed = split_info["seed"]
    cfg.data_root = str(Path(tempfile.mkdtemp()))
    sample_fraction = float(hyperparameters.get("sample_fraction", 1.0))
    if 0.0 < sample_fraction < 1.0:
        cfg.data_pct = sample_fraction
    (Path(cfg.data_root) / "norm_stats.json").write_text(
        json.dumps({"mean": split_info["norm_mean"], "std": split_info["norm_std"]})
    )

    chips_dir = resolve_directory(dataset_chips, "*.tif")
    labels_geojson = resolve_labels_geojson(Path(resolve_directory(dataset_labels, "*.geojson")))
    pretrained = download_checkpoint(base_model_weights)
    encoder_ckpt = hf_hub_download(repo_id=cfg.hf_ckpt_repo, filename=cfg.hf_ckpt_file)
    out_dir = Path(tempfile.mkdtemp()) / "ft"

    train_chips_dir = Path(tempfile.mkdtemp()) / "train_chips"
    train_chips_dir.mkdir()
    for name in split_info["train_chip_names"]:
        src_path = (chips_dir / name).resolve()
        (train_chips_dir / name).symlink_to(src_path)
        # OAM chips store geotransform in a `.tif.aux.xml` sidecar; rasterio reads it
        # from the same dir as the chip, so it must be symlinked too.
        for ext in (".aux.xml", ".tfw", ".prj"):
            sidecar = src_path.parent / (src_path.name + ext)
            if sidecar.exists():
                (train_chips_dir / sidecar.name).symlink_to(sidecar)

    with mlflow_training_context(hyperparameters, model_name, base_model_id, dataset_id):
        summary = finetune(
            cfg=cfg,
            pretrained_ckpt=str(pretrained),
            chips_dir=str(train_chips_dir),
            labels_geojson=str(labels_geojson),
            out_dir=str(out_dir),
            val_frac=hyperparameters.get("inner_val_frac", 0.1),
            ft_lr=hyperparameters.get("learning_rate", 5e-5),
            ft_epochs=hyperparameters.get("epochs", 15),
            ft_patience=hyperparameters.get("early_stop_patience", 5),
        )
        log_metadata(metadata={"fair/finetune_summary": summary})

    train_losses, val_losses = read_loss_history(out_dir)
    if train_losses or val_losses:
        log_loss_history(train_losses, val_losses)

    # Encoder weights are frozen and absent from the Lightning ckpt; re-inject via ckpt_path.
    return DinoV3HotLit.load_from_checkpoint(summary["best_ckpt"], map_location="cpu", ckpt_path=str(encoder_ckpt)).net


@step
def evaluate_model(
    trained_model: Any,
    dataset_chips: str,
    dataset_labels: str,
    hyperparameters: dict[str, Any],
    split_info: dict[str, Any],
    num_classes: int = 2,
    class_names: list[str] | None = None,
) -> Annotated[dict[str, Any], "metrics"]:
    """Burn GT polygons, then delegate to `dinov3_hot.evaluation.evaluate` on the spatial val set."""
    from dinov3_hot.evaluation import evaluate as eval_run
    from dinov3_hot.paths import resolve_labels_geojson
    from geomltoolkits.raster.burn import burn_labels

    chips_dir = resolve_directory(dataset_chips, "*.tif")
    labels_geojson = resolve_labels_geojson(Path(resolve_directory(dataset_labels, "*.geojson")))
    masks_dir = Path(tempfile.mkdtemp()) / "masks"
    burn_labels(
        labels_path=str(labels_geojson),
        chips_dir=str(chips_dir),
        output_dir=str(masks_dir),
        burn_value=255,
    )

    metrics = eval_run(
        trained_model,
        chips_dir,
        masks_dir,
        split_info["val_chip_names"],
        mean=split_info["norm_mean"],
        std=split_info["norm_std"],
    )
    log_evaluation_results(metrics)
    return metrics


@step
def tune_postprocess(
    trained_model: Any,
    dataset_chips: str,
    dataset_labels: str,
    hyperparameters: dict[str, Any],
    split_info: dict[str, Any],
) -> Annotated[dict[str, Any], "recommended_inference_params"]:
    """Optuna over post-process params via `dinov3_hot.tune.tune_postprocess_run`."""
    from dinov3_hot.paths import resolve_labels_geojson
    from dinov3_hot.tune import tune_postprocess_run

    n_trials = int(hyperparameters.get("tune_postprocess_trials", 30))
    chips_dir = resolve_directory(dataset_chips, "*.tif")
    labels_geojson = resolve_labels_geojson(Path(resolve_directory(dataset_labels, "*.geojson")))

    result = tune_postprocess_run(
        trained_model,
        chips_dir,
        labels_geojson,
        split_info["val_chip_names"],
        mean=split_info["norm_mean"],
        std=split_info["norm_std"],
        n_trials=n_trials,
        seed=int(split_info["seed"]),
        default_params=DEFAULT_INFERENCE_PARAMS,
    )
    return result["best_params"]


@step
def run_inference(
    model_uri: str,
    input_images: str,
    inference_params: dict[str, Any],
) -> Annotated[dict[str, Any], "predictions"]:
    """ONNX inference via `dinov3_hot.serve.predict_session`."""
    from dinov3_hot.serve import predict_session

    from fair.serve.base import load_session

    return predict_session(
        load_session(model_uri),
        Path(resolve_directory(input_images)),
        inference_params,
    )


@step(output_materializers={"onnx_model": ONNXMaterializer})
def export_onnx(
    trained_model: Any,
    hyperparameters: dict[str, Any],
    num_classes: int = 2,
) -> Annotated[bytes, "onnx_model"]:
    """Export the trained decoder + frozen encoder to single-file ONNX bytes."""
    import onnx
    import torch
    from dinov3_hot.serve import MODEL_INPUT_SIZE
    from torch import nn

    class _MainOnly(nn.Module):
        def __init__(self, wrapped: nn.Module) -> None:
            super().__init__()
            self.wrapped = wrapped

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.wrapped(x)[0]

    model = _MainOnly(trained_model.cpu().eval()).eval()
    dummy = torch.randn(1, 3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = str(Path(tmpdir) / "model.onnx")
        torch.onnx.export(
            model,
            (dummy,),
            path,
            input_names=["image"],
            output_names=["logits"],
            dynamic_shapes={"x": {0: torch.export.Dim("batch")}},
            dynamo=True,
        )
        # Dynamo writes weights to a sidecar `.onnx.data`; inline so the returned bytes are self-contained.
        onnx.save(onnx.load(path), path, save_as_external_data=False)
        onnx.checker.check_model(path)
        return Path(path).read_bytes()


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
    trained = train_model(
        dataset_chips=dataset_chips,
        dataset_labels=dataset_labels,
        base_model_weights=base_model_weights,
        hyperparameters=hyperparameters,
        split_info=split_info,
        num_classes=num_classes,
    )
    evaluate_model(
        trained_model=trained,
        dataset_chips=dataset_chips,
        dataset_labels=dataset_labels,
        hyperparameters=hyperparameters,
        split_info=split_info,
        num_classes=num_classes,
    )
    tune_postprocess(
        trained_model=trained,
        dataset_chips=dataset_chips,
        dataset_labels=dataset_labels,
        hyperparameters=hyperparameters,
        split_info=split_info,
    )
    export_onnx(trained_model=trained, hyperparameters=hyperparameters, num_classes=num_classes)


@pipeline
def inference_pipeline(
    model_uri: str,
    input_images: str,
    inference_params: dict[str, Any],
) -> None:
    run_inference(
        model_uri=model_uri,
        input_images=input_images,
        inference_params=inference_params,
    )
