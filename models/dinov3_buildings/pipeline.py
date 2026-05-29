"""ZenML pipeline for DINOv3-L + UperNet building segmentation."""

import json
import re
import tempfile
from pathlib import Path
from typing import Annotated, Any

from zenml import log_metadata, pipeline, step

from fair.zenml.instrumentation import log_evaluation_results, mlflow_training_context
from fair.zenml.materializers import ONNXMaterializer
from fair.zenml.metrics import log_loss_history

MODEL_INPUT_SIZE = 256
_SLIDING_STRIDE = 128
_SEED_MIN_DISTANCE = 4
_TUNE_MIN_VAL_CHIPS = 8
_OAM_TILE_RE = re.compile(r"^OAM-(\d+)-(\d+)-\d+\.tiff?$")
# Inlined so the distroless inference image stays torch-free (dinov3_hot.data pulls torch).
_HOT_MEAN = [0.4296737853453577, 0.4001659668453235, 0.34333372802741474]
_HOT_STD = [0.2056069389373208, 0.16738555558380538, 0.1598986422586595]
_DEFAULT_INFERENCE_PARAMS: dict[str, Any] = {
    "confidence_threshold": 0.5,
    "seed_min_distance": _SEED_MIN_DISTANCE,
    "simplify_m": 1.0,
    "regularize_area_threshold": 0.55,
    "regularize_overlap_tol_m2": 1.0,
    "min_area_m2": 1.0,
}


def preprocess(batch: dict[str, Any]) -> tuple[Any, Any]:
    """Scale image to [0,1], apply dataset mean/std normalisation."""
    import torch

    images = batch["image"].float() / 255.0
    images = (images - torch.tensor(_HOT_MEAN).view(1, 3, 1, 1)) / torch.tensor(_HOT_STD).view(1, 3, 1, 1)
    masks = batch["mask"].long().squeeze(1)
    return images, masks


def postprocess(logits: Any) -> Any:
    """Sigmoid on the mask channel, threshold at 0.5."""
    import numpy as np

    return (1.0 / (1.0 + np.exp(-logits[:, 0])) > 0.5).astype(np.uint8)


def _gaussian_kernel(size: int, sigma_frac: float = 0.125) -> Any:
    """Separable Gaussian for sliding-window stitching; mirrors upstream `_gaussian_kernel`."""
    import numpy as np
    from scipy.signal.windows import gaussian as gauss1d

    w = gauss1d(size, std=sigma_frac * size)
    k = np.outer(w, w)
    return k / k.max()


def _normalize_chw(image_hwc_uint8: Any, mean: list[float], std: list[float]) -> Any:
    """HWC uint8 to CHW float32 in [0,1], normalised. Torch-free."""
    import numpy as np

    arr = image_hwc_uint8.astype(np.float32).transpose(2, 0, 1) / 255.0
    m = np.asarray(mean, dtype=np.float32).reshape(3, 1, 1)
    s = np.asarray(std, dtype=np.float32).reshape(3, 1, 1)
    return (arr - m) / s


def _sliding_window_onnx(
    session: Any,
    image_hwc: Any,
    mean: list[float],
    std: list[float],
    window: int = MODEL_INPUT_SIZE,
    stride: int = _SLIDING_STRIDE,
) -> tuple[Any, Any, Any]:
    """Gaussian-stitched ONNX sliding window; returns (mask_prob, boundary_prob, distance)."""
    import numpy as np

    h, w, _ = image_hwc.shape
    mask_acc = np.zeros((h, w), dtype=np.float32)
    boundary_acc = np.zeros((h, w), dtype=np.float32)
    distance_acc = np.zeros((h, w), dtype=np.float32)
    weight_acc = np.zeros((h, w), dtype=np.float32)
    kernel = _gaussian_kernel(window)

    rows = list(range(0, max(1, h - window + 1), stride))
    cols = list(range(0, max(1, w - window + 1), stride))
    if rows[-1] + window < h:
        rows.append(h - window)
    if cols[-1] + window < w:
        cols.append(w - window)

    input_name = session.get_inputs()[0].name
    for r in rows:
        for c in cols:
            tile = image_hwc[r : r + window, c : c + window, :]
            if tile.shape[0] != window or tile.shape[1] != window:
                pad = np.zeros((window, window, 3), dtype=image_hwc.dtype)
                pad[: tile.shape[0], : tile.shape[1]] = tile
                tile = pad
            x = _normalize_chw(tile, mean, std)[np.newaxis, ...]
            logits = session.run(None, {input_name: x})[0]
            mask_prob = 1.0 / (1.0 + np.exp(-logits[0, 0]))
            boundary_prob = 1.0 / (1.0 + np.exp(-logits[0, 1]))
            distance = np.tanh(logits[0, 2])
            mask_acc[r : r + window, c : c + window] += mask_prob * kernel
            boundary_acc[r : r + window, c : c + window] += boundary_prob * kernel
            distance_acc[r : r + window, c : c + window] += distance * kernel
            weight_acc[r : r + window, c : c + window] += kernel

    weight_acc = np.maximum(weight_acc, 1e-6)
    return mask_acc / weight_acc, boundary_acc / weight_acc, distance_acc / weight_acc


def _add_scores(gdf: Any, labels: Any, mask_prob: Any, transform: Any, crs: Any) -> Any:
    """Attach a per-polygon `score`: mean mask_prob over the watershed instance the polygon came from."""
    import numpy as np
    from rasterio.transform import rowcol

    if not len(gdf):
        return gdf
    if gdf.crs != crs:
        gdf = gdf.to_crs(crs)
    labels_flat = labels.ravel().astype(np.int64)
    sums = np.bincount(labels_flat, weights=mask_prob.ravel())
    counts = np.bincount(labels_flat)
    label_means = sums / np.maximum(counts, 1)
    h, w = labels.shape
    scores = []
    for geom in gdf.geometry:
        pt = geom.representative_point()
        row, col = rowcol(transform, pt.x, pt.y)
        if 0 <= row < h and 0 <= col < w:
            lv = int(labels[row, col])
            scores.append(float(label_means[lv]) if 0 <= lv < len(label_means) else 0.0)
        else:
            scores.append(0.0)
    gdf = gdf.copy()
    gdf["score"] = scores
    return gdf


def _merge_chips_to_array(input_dir: Path) -> tuple[Any, Any, Any]:
    """Merge *.tif/*.tiff/*.png chips into one HWC uint8 array via rasterio.merge."""
    import numpy as np
    import rasterio
    from rasterio.merge import merge

    patterns = ("*.tif", "*.tiff", "*.png")
    paths = sorted(p for pat in patterns for p in input_dir.glob(pat))
    if not paths:
        raise FileNotFoundError(f"No input images found in {input_dir}")

    sources = [rasterio.open(p) for p in paths]
    try:
        mosaic, transform = merge(sources, indexes=[1, 2, 3])
        crs = sources[0].crs
    finally:
        for s in sources:
            s.close()
    return mosaic.transpose(1, 2, 0).astype(np.uint8), transform, crs


def predict(session: Any, input_images: str, params: dict[str, Any]) -> dict[str, Any]:
    """Merge chips, sliding-window ONNX, watershed instance separation, vectorise to EPSG:4326."""
    import shapely.geometry as sgeom
    from dinov3_hot.infer import instance_separate, vectorize

    from fair.utils.data import resolve_directory

    if "confidence_threshold" not in params:
        raise ValueError("params['confidence_threshold'] is required")
    threshold = float(params["confidence_threshold"])
    simplify_m = float(params.get("simplify_m", 1.0))
    area_threshold = float(params.get("regularize_area_threshold", 0.55))
    overlap_tol_m2 = float(params.get("regularize_overlap_tol_m2", 1.0))
    min_area_m2 = float(params.get("min_area_m2", 1.0))
    seed_min_distance = int(params.get("seed_min_distance", _SEED_MIN_DISTANCE))

    input_dir = resolve_directory(input_images)
    image_hwc, transform, crs = _merge_chips_to_array(input_dir)
    mask_prob, _boundary, distance = _sliding_window_onnx(session, image_hwc, _HOT_MEAN, _HOT_STD)
    labels = instance_separate(mask_prob, distance, mask_threshold=threshold, seed_min_distance=seed_min_distance)
    gdf = vectorize(
        labels,
        transform,
        crs,
        min_area_m2=min_area_m2,
        simplify_m=simplify_m,
        regularize_area_threshold=area_threshold,
        regularize_overlap_tol_m2=overlap_tol_m2,
    )
    if not len(gdf):
        return {"type": "FeatureCollection", "features": []}
    gdf = _add_scores(gdf, labels, mask_prob, transform, crs)
    out_geoms = gdf.to_crs(epsg=4326).geometry
    features = [
        {"type": "Feature", "properties": {"class": 1, "score": float(s)}, "geometry": sgeom.mapping(g)}
        for s, g in zip(gdf["score"], out_geoms, strict=True)
        if not g.is_empty
    ]
    return {"type": "FeatureCollection", "features": features}


def _download_checkpoint(url: str) -> Path:
    """Download a Lightning checkpoint via upath (http(s)/s3/local all supported)."""
    from upath import UPath

    local = Path(tempfile.mkdtemp()) / UPath(url).name
    local.write_bytes(UPath(url).read_bytes())
    return local


def _compute_dataset_stats(chips_dir: Path) -> tuple[list[float], list[float]]:
    """Per-channel mean/std over all chips, scaled to [0, 1]."""
    import numpy as np
    import rasterio

    sums = np.zeros(3, dtype=np.float64)
    sums_sq = np.zeros(3, dtype=np.float64)
    px = 0
    for chip in sorted(chips_dir.glob("*.tif")):
        with rasterio.open(chip) as src:
            arr = src.read([1, 2, 3]).astype(np.float64) / 255.0
        sums += arr.sum(axis=(1, 2))
        sums_sq += (arr * arr).sum(axis=(1, 2))
        px += arr.shape[1] * arr.shape[2]
    mean = sums / px
    return mean.tolist(), np.sqrt(sums_sq / px - mean * mean).tolist()


def _read_loss_history(out_dir: Path) -> tuple[list[float], list[float]]:
    """Per-epoch (train_loss, val_loss) from upstream finetune's Lightning CSVLogger."""
    import csv

    versions = sorted((out_dir / "lightning").glob("version_*"))
    if not versions:
        return [], []
    metrics_csv = versions[-1] / "metrics.csv"
    if not metrics_csv.exists():
        return [], []
    train_by_epoch: dict[int, float] = {}
    val_by_epoch: dict[int, float] = {}
    with metrics_csv.open() as f:
        for row in csv.DictReader(f):
            if not row.get("epoch"):
                continue
            epoch = int(row["epoch"])
            tl = row.get("train/loss_epoch") or ""
            vl = row.get("val/loss") or ""
            if tl:
                train_by_epoch[epoch] = float(tl)
            if vl:
                val_by_epoch[epoch] = float(vl)
    return (
        [train_by_epoch[e] for e in sorted(train_by_epoch)],
        [val_by_epoch[e] for e in sorted(val_by_epoch)],
    )


def _spatial_split(
    chip_names: list[str], val_ratio: float, seed: int, block_size: int = 4
) -> tuple[list[str], list[str]]:
    """Spatial block split on OAM-{x}-{y}-{z}.tif filenames; whole (x//K, y//K) blocks go to one side.
    Filenames that don't match the OAM pattern fall back to a seeded random split."""
    import numpy as np

    matched: dict[tuple[int, int], list[str]] = {}
    unmatched: list[str] = []
    for name in chip_names:
        m = _OAM_TILE_RE.match(name)
        if m is None:
            unmatched.append(name)
            continue
        x, y = int(m.group(1)), int(m.group(2))
        matched.setdefault((x // block_size, y // block_size), []).append(name)

    rng = np.random.default_rng(seed)
    blocks = sorted(matched.keys())
    rng.shuffle(blocks)

    n_total = sum(len(v) for v in matched.values()) + len(unmatched)
    n_val_target = max(1, int(n_total * val_ratio))

    val: list[str] = []
    train: list[str] = []
    for block in blocks:
        bucket = val if len(val) < n_val_target else train
        bucket.extend(matched[block])

    if unmatched:
        leftover = sorted(unmatched)
        rng.shuffle(leftover)
        remaining = max(0, n_val_target - len(val))
        val.extend(leftover[:remaining])
        train.extend(leftover[remaining:])

    return sorted(train), sorted(val)


def _resolve_labels_geojson(dataset_labels: str) -> Path:
    """Accept either a directory containing one .geojson or a direct .geojson path."""
    from fair.utils.data import resolve_directory, resolve_path

    path = Path(dataset_labels)
    if path.suffix == ".geojson" and not str(dataset_labels).startswith(("s3://", "http")):
        return resolve_path(dataset_labels)
    local = resolve_directory(dataset_labels, "*.geojson")
    return next(local.glob("*.geojson"))


@step
def split_dataset(
    dataset_chips: str,
    dataset_labels: str,
    hyperparameters: dict[str, Any],
) -> Annotated[dict[str, Any], "split_info"]:
    """Resolve mean/std and pick a spatially-stratified val set; emit split_info for train/eval."""
    from fair.utils.data import resolve_directory

    chips_dir = resolve_directory(dataset_chips, "*.tif")
    chip_names = sorted(p.name for p in chips_dir.glob("*.tif"))

    val_ratio = hyperparameters.get("val_ratio", 0.2)
    seed = hyperparameters.get("split_seed", 42)
    block_size = hyperparameters.get("block_size", 4)
    train_chip_names, val_chip_names = _spatial_split(chip_names, val_ratio, seed, block_size)

    norm_mode = hyperparameters.get("norm_stats", "hot_global")
    if norm_mode == "hot_global":
        mean, std = _HOT_MEAN, _HOT_STD
    elif norm_mode == "dataset":
        mean, std = _compute_dataset_stats(chips_dir)
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
        "description": (
            f"Spatial block split (K={block_size}) on OAM-x-y-z tile coords; "
            f"non-matching names fall back to seeded random. Normalised with {norm_mode} mean/std."
        ),
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
    """Fine-tune the decoder via `dinov3_hot.finetune.finetune`."""
    from dinov3_hot.config import load_config
    from dinov3_hot.finetune import finetune
    from dinov3_hot.model import DinoV3HotLit
    from huggingface_hub import hf_hub_download

    from fair.utils.data import resolve_directory

    cfg = load_config(None)
    cfg.seed = split_info["seed"]
    cfg.data_root = str(Path(tempfile.mkdtemp()))
    # Pre-write so dinov3_hot.data.load_norm_stats uses our stats instead of fetching from HF.
    (Path(cfg.data_root) / "norm_stats.json").write_text(
        json.dumps({"mean": split_info["norm_mean"], "std": split_info["norm_std"]})
    )
    chips_dir = resolve_directory(dataset_chips, "*.tif")
    labels_geojson = _resolve_labels_geojson(dataset_labels)
    pretrained = _download_checkpoint(base_model_weights)
    encoder_ckpt = hf_hub_download(repo_id=cfg.hf_ckpt_repo, filename=cfg.hf_ckpt_file)
    out_dir = Path(tempfile.mkdtemp()) / "ft"

    # Stage only our spatial-train chips so upstream finetune's internal random split
    # is contained within them and never touches the held-out spatial val.
    train_chips_dir = Path(tempfile.mkdtemp()) / "train_chips"
    train_chips_dir.mkdir()
    for name in split_info["train_chip_names"]:
        (train_chips_dir / name).symlink_to((chips_dir / name).resolve())

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

    train_losses, val_losses = _read_loss_history(out_dir)
    if train_losses or val_losses:
        log_loss_history(train_losses, val_losses)

    # ckpt_path is excluded from save_hyperparameters; pass it back for backbone reload.
    return DinoV3HotLit.load_from_checkpoint(summary["best_ckpt"], map_location="cpu", ckpt_path=str(encoder_ckpt)).net


def evaluate(
    net: Any,
    chips_dir: Path,
    masks_dir: Path,
    val_chip_names: list[str],
    mean: list[float],
    std: list[float],
    device: str = "cpu",
) -> dict[str, Any]:
    """Pixel IoU, instance P/R/F1 @ IoU>0.5, polygon shape stats over a val split.

    `masks_dir` holds burned-label rasters matching chip filenames.
    """
    import numpy as np
    import rasterio
    import torch
    from dinov3_hot.metrics import instance_prf, polygon_orthogonality, polygon_vertex_count
    from dinov3_hot.postprocess import vectorize_binary_mask
    from scipy import ndimage

    mean_arr = np.asarray(mean, dtype=np.float32).reshape(3, 1, 1)
    std_arr = np.asarray(std, dtype=np.float32).reshape(3, 1, 1)
    model = net.to(device).eval()

    pix_inter = pix_union = 0
    tp_total = fp_total = fn_total = 0
    vertex_sum = orth_sum = 0.0
    poly_count = 0

    with torch.no_grad():
        for name in val_chip_names:
            with rasterio.open(chips_dir / name) as src:
                img = src.read([1, 2, 3]).astype(np.float32) / 255.0
                transform = src.transform
                crs = src.crs
            with rasterio.open(masks_dir / name) as src:
                gt = (src.read(1) > 0).astype(np.uint8)
            t = torch.from_numpy((img - mean_arr) / std_arr).unsqueeze(0).to(device)
            main_logits, _ = model(t)
            pred = (torch.sigmoid(main_logits[0, 0]) > 0.5).cpu().numpy().astype(np.uint8)

            pix_inter += int((pred & gt).sum())
            pix_union += int((pred | gt).sum())

            pred_lbl = ndimage.label(pred)[0].astype(np.int32)
            gt_lbl = ndimage.label(gt)[0].astype(np.int32)
            prf = instance_prf(pred_lbl, gt_lbl)
            tp_total += prf["tp"]
            fp_total += prf["fp"]
            fn_total += prf["fn"]

            gdf = vectorize_binary_mask(pred, transform, crs)
            if len(gdf):
                metric_geoms = list(gdf.to_crs(gdf.estimate_utm_crs()).geometry)
                n = len(metric_geoms)
                vertex_sum += polygon_vertex_count(metric_geoms) * n
                orth_sum += polygon_orthogonality(metric_geoms) * n
                poly_count += n

    pixel_iou = pix_inter / pix_union if pix_union else 0.0
    precision = tp_total / (tp_total + fp_total) if tp_total + fp_total else 0.0
    recall = tp_total / (tp_total + fn_total) if tp_total + fn_total else 0.0
    f1 = (2 * tp_total) / (2 * tp_total + fp_total + fn_total) if (2 * tp_total + fp_total + fn_total) else 0.0
    avg_vertices = vertex_sum / poly_count if poly_count else 0.0
    orthogonality = orth_sum / poly_count if poly_count else 0.0

    return {
        "pixel_iou": pixel_iou,
        "instance_precision": precision,
        "instance_recall": recall,
        "instance_f1": f1,
        "pred_avg_vertices": avg_vertices,
        "pred_orthogonality": orthogonality,
    }


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
    """Burn labels, take the held-out spatial val set from split_info, delegate to `evaluate(...)`."""
    import torch
    from geomltoolkits.raster.burn import burn_labels

    from fair.utils.data import resolve_directory

    chips_dir = resolve_directory(dataset_chips, "*.tif")
    labels_geojson = _resolve_labels_geojson(dataset_labels)
    masks_dir = Path(tempfile.mkdtemp()) / "masks"
    burn_labels(labels_path=str(labels_geojson), chips_dir=str(chips_dir), output_dir=str(masks_dir), burn_value=255)

    metrics = evaluate(
        net=trained_model,
        chips_dir=chips_dir,
        masks_dir=masks_dir,
        val_chip_names=split_info["val_chip_names"],
        mean=split_info["norm_mean"],
        std=split_info["norm_std"],
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    log_evaluation_results(metrics)
    return metrics


def _cache_val_forwards(
    net: Any,
    chips_dir: Path,
    val_chip_names: list[str],
    mean: list[float],
    std: list[float],
    device: str,
) -> list[dict[str, Any]]:
    """One model forward per val chip; caches the three head outputs + georeferencing for tuning."""
    import numpy as np
    import rasterio
    import torch

    mean_arr = np.asarray(mean, dtype=np.float32).reshape(3, 1, 1)
    std_arr = np.asarray(std, dtype=np.float32).reshape(3, 1, 1)
    model = net.to(device).eval()
    cache: list[dict[str, Any]] = []
    with torch.no_grad():
        for name in val_chip_names:
            with rasterio.open(chips_dir / name) as src:
                img = src.read([1, 2, 3]).astype(np.float32) / 255.0
                transform = src.transform
                crs = src.crs
            t = torch.from_numpy((img - mean_arr) / std_arr).unsqueeze(0).to(device)
            main_logits, _ = model(t)
            logits = main_logits[0].cpu().numpy()
            cache.append(
                {
                    "name": name,
                    "mask_prob": 1.0 / (1.0 + np.exp(-logits[0])),
                    "boundary_prob": 1.0 / (1.0 + np.exp(-logits[1])),
                    "distance": np.tanh(logits[2]),
                    "transform": transform,
                    "crs": crs,
                }
            )
    return cache


def _gt_clipped_to_val(labels_geojson: Path, cache: list[dict[str, Any]]) -> Any:
    """Read ground-truth polygons and clip to the union of val chip extents in the val CRS."""
    import geopandas as gpd
    import rasterio
    import shapely.geometry as sgeom
    from shapely.ops import unary_union

    boxes = [
        sgeom.box(*rasterio.transform.array_bounds(*entry["mask_prob"].shape, entry["transform"])) for entry in cache
    ]
    val_union = unary_union(boxes)
    gt = gpd.read_file(labels_geojson)
    crs = cache[0]["crs"]
    if gt.crs != crs:
        gt = gt.to_crs(crs)
    return gt[gt.intersects(val_union)].copy()


def _trial_predictions(cache: list[dict[str, Any]], params: dict[str, Any]) -> Any:
    """Run instance_separate + vectorize per cached chip with these params; return one merged gdf."""
    import geopandas as gpd
    import pandas as pd
    from dinov3_hot.infer import instance_separate, vectorize

    per_chip = []
    for entry in cache:
        labels = instance_separate(
            entry["mask_prob"],
            entry["distance"],
            mask_threshold=params["confidence_threshold"],
            seed_min_distance=params["seed_min_distance"],
        )
        gdf = vectorize(
            labels,
            entry["transform"],
            entry["crs"],
            min_area_m2=params["min_area_m2"],
            simplify_m=params["simplify_m"],
            regularize_area_threshold=params["regularize_area_threshold"],
            regularize_overlap_tol_m2=params["regularize_overlap_tol_m2"],
        )
        if len(gdf):
            per_chip.append(gdf)
    if not per_chip:
        return gpd.GeoDataFrame(geometry=[], crs=cache[0]["crs"])
    return gpd.GeoDataFrame(pd.concat(per_chip, ignore_index=True), crs=per_chip[0].crs)


@step
def tune_postprocess(
    trained_model: Any,
    dataset_chips: str,
    dataset_labels: str,
    hyperparameters: dict[str, Any],
    split_info: dict[str, Any],
) -> Annotated[dict[str, Any], "recommended_inference_params"]:
    """Optuna search over the six inference post-processing params, scored against the val set.
    Returns current defaults if the search is disabled or the val set is below the chip-count guard."""
    n_trials = int(hyperparameters.get("tune_postprocess_trials", 30))
    val_chip_names = split_info["val_chip_names"]
    if n_trials <= 0 or len(val_chip_names) < _TUNE_MIN_VAL_CHIPS:
        skip_reason = "disabled" if n_trials <= 0 else "val_too_small"
        log_metadata(metadata={"fair/tune_postprocess_skipped": skip_reason})
        return dict(_DEFAULT_INFERENCE_PARAMS)

    import optuna
    import polymetrics
    import torch

    from fair.utils.data import resolve_directory

    chips_dir = resolve_directory(dataset_chips, "*.tif")
    labels_geojson = _resolve_labels_geojson(dataset_labels)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cache = _cache_val_forwards(
        trained_model,
        chips_dir,
        val_chip_names,
        split_info["norm_mean"],
        split_info["norm_std"],
        device,
    )
    gt = _gt_clipped_to_val(labels_geojson, cache)

    def objective(trial: Any) -> float:
        params = {
            "confidence_threshold": trial.suggest_float("confidence_threshold", 0.3, 0.7),
            "seed_min_distance": trial.suggest_int("seed_min_distance", 2, 16),
            "simplify_m": trial.suggest_float("simplify_m", 0.5, 3.0),
            "regularize_area_threshold": trial.suggest_float("regularize_area_threshold", 0.4, 0.8),
            "regularize_overlap_tol_m2": trial.suggest_float("regularize_overlap_tol_m2", 0.0, 5.0),
            "min_area_m2": trial.suggest_float("min_area_m2", 0.0, 5.0),
        }
        pred = _trial_predictions(cache, params)
        if not len(pred) or not len(gt):
            return 0.0
        r = polymetrics.evaluate(gt, pred, iou_threshold=0.5, compute_map=False)
        return r.f1 + 0.3 * r.mean_iou

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=split_info["seed"]),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    recommended = dict(study.best_params)
    log_metadata(
        metadata={
            "fair/recommended_inference_params": recommended,
            "fair/tune_postprocess_best_score": float(study.best_value),
        }
    )
    return recommended


@step
def run_inference(
    model_uri: str,
    input_images: str,
    inference_params: dict[str, Any],
) -> Annotated[dict[str, Any], "predictions"]:
    from fair.serve.base import load_session

    return predict(load_session(model_uri), input_images, inference_params)


@step(output_materializers={"onnx_model": ONNXMaterializer})
def export_onnx(
    trained_model: Any,
    hyperparameters: dict[str, Any],
    num_classes: int = 2,
) -> Annotated[bytes, "onnx_model"]:
    """Export decoder + frozen encoder to single-file ONNX (opset 18); aux head dropped."""
    import os

    import onnx
    import torch
    from torch import nn

    class _MainOnly(nn.Module):
        def __init__(self, wrapped: nn.Module):
            super().__init__()
            self.wrapped = wrapped

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.wrapped(x)[0]

    model = _MainOnly(trained_model.cpu().eval()).eval()
    dummy = torch.randn(1, 3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)
    fd, path = tempfile.mkstemp(suffix=".onnx")
    os.close(fd)
    try:
        torch.onnx.export(
            model,
            (dummy,),
            path,
            input_names=["image"],
            output_names=["logits"],
            dynamic_shapes={"x": {0: torch.export.Dim("batch")}},
            dynamo=True,
        )
        # Dynamo writes weights to a sidecar .onnx.data that disappears with the temp dir;
        # reload + save inline so the returned bytes are self-contained (~1.4 GB, under the 2 GB protobuf limit).
        onnx.save(onnx.load(path), path, save_as_external_data=False)
        onnx.checker.check_model(path)
        return Path(path).read_bytes()
    finally:
        Path(path).unlink(missing_ok=True)
        Path(f"{path}.data").unlink(missing_ok=True)


@pipeline
def training_pipeline(
    base_model_weights: str,
    dataset_chips: str,
    dataset_labels: str,
    num_classes: int,
    hyperparameters: dict[str, Any],
) -> None:
    split_info = split_dataset(
        dataset_chips=dataset_chips, dataset_labels=dataset_labels, hyperparameters=hyperparameters
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
    inference_params: dict[str, Any] | None = None,
) -> None:
    run_inference(model_uri=model_uri, input_images=input_images, inference_params=inference_params or {})
