import hashlib
import random
import shutil
import tempfile
from pathlib import Path
from typing import Annotated, Any

from zenml import log_metadata, pipeline, step

from fair.zenml.instrumentation import log_evaluation_results, mlflow_training_context
from fair.zenml.materializers import CheckpointBytesMaterializer, ONNXMaterializer

MODEL_INPUT_SIZE = 128
CELL_SIZE_M = 5.0
CLASS_NAMES = ("background", "waste")


def _get_device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _download_checkpoint(url: str) -> Path:
    from upath import UPath

    print(url)
    local_path = Path(tempfile.mkdtemp()) / UPath(url).name
    local_path.write_bytes(UPath(url).read_bytes())
    return local_path


def _log_yolo_loss_history(model: Any) -> None:
    import csv

    from fair.zenml.metrics import log_loss_history

    save_dir = getattr(model.trainer, "save_dir", None) if hasattr(model, "trainer") else None
    if save_dir is None:
        return
    results_csv = Path(save_dir) / "results.csv"
    if not results_csv.exists():
        return

    train_losses: list[float] = []
    val_losses: list[float] = []
    with results_csv.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            stripped = {k.strip(): v.strip() for k, v in row.items()}
            train_loss = stripped.get("train/loss")
            val_loss = stripped.get("val/loss")
            if train_loss is not None and val_loss is not None:
                train_losses.append(float(train_loss))
                val_losses.append(float(val_loss))

    if train_losses:
        import mlflow

        for epoch, (tl, vl) in enumerate(zip(train_losses, val_losses, strict=True)):
            mlflow.log_metric("train_loss", tl, step=epoch)  # ty: ignore[possibly-missing-attribute]
            mlflow.log_metric("val_loss", vl, step=epoch)  # ty: ignore[possibly-missing-attribute]
        log_loss_history(train_losses, val_losses)


def _restore_checkpoint(trained_model: Any):
    from ultralytics import YOLO

    if isinstance(trained_model, YOLO):
        return trained_model
    if isinstance(trained_model, bytes):
        checkpoint = Path(tempfile.mkdtemp()) / "best.pt"
        checkpoint.write_bytes(trained_model)
        return YOLO(str(checkpoint))
    return YOLO(trained_model)


def preprocess(image_path: Any, chip_size: int = MODEL_INPUT_SIZE) -> Any:
    import numpy as np
    import rasterio
    import torch
    import torch.nn.functional as F

    with rasterio.open(image_path) as src:
        arr = src.read([1, 2, 3]).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).unsqueeze(0)
    if tensor.shape[-2:] != (chip_size, chip_size):
        tensor = F.interpolate(tensor, size=(chip_size, chip_size), mode="bilinear", align_corners=False)
    return tensor


def postprocess(output: Any) -> list[dict[str, Any]]:
    import numpy as np

    probs = np.atleast_2d(np.asarray(output))
    class_ids = probs.argmax(axis=1)
    return [
        {
            "class": int(cls),
            "label": CLASS_NAMES[int(cls)],
            "confidence": float(probs[i, int(cls)]),
        }
        for i, cls in enumerate(class_ids)
    ]


def predict(session: Any, input_images: str, params: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    import rasterio
    from PIL import Image
    from pyproj import Transformer
    from shapely.geometry import mapping
    from shapely.ops import transform as shapely_transform

    from fair.utils.data import resolve_directory

    confidence_threshold = float(params["confidence_threshold"])
    cell_size_m = float(params.get("cell_size_m", CELL_SIZE_M))
    input_name = session.get_inputs()[0].name
    waste_idx = CLASS_NAMES.index("waste")

    input_path = resolve_directory(input_images)
    chip_paths = (
        [input_path]
        if input_path.is_file()
        else sorted(p for pat in ("*.tif", "*.tiff") for p in input_path.rglob(pat))
    )
    if not chip_paths:
        msg = f"No georeferenced (.tif/.tiff) chips found in {input_path}"
        raise FileNotFoundError(msg)

    mosaic_path = build_mosaic(chip_paths)
    features: list[dict[str, Any]] = []
    with rasterio.open(mosaic_path) as mosaic:
        mosaic_crs = mosaic.crs
        bounds = mosaic.bounds
        nodata = mosaic.nodata

        centroid_lon = (bounds.left + bounds.right) / 2
        centroid_lat = (bounds.bottom + bounds.top) / 2
        target_crs = pick_utm_crs(centroid_lon, centroid_lat) if mosaic_crs.is_geographic else mosaic_crs

        to_proj = Transformer.from_crs(mosaic_crs, target_crs, always_xy=True)
        xs = [bounds.left, bounds.left, bounds.right, bounds.right]
        ys = [bounds.bottom, bounds.top, bounds.bottom, bounds.top]
        px, py = to_proj.transform(xs, ys)
        bounds_proj = (min(px), min(py), max(px), max(py))

        grid = build_grid_gdf(bounds_proj, cell_size_m, target_crs)
        utm_to_mosaic = Transformer.from_crs(target_crs, mosaic_crs, always_xy=True)
        to_wgs84 = Transformer.from_crs(target_crs, "EPSG:4326", always_xy=True)

        cell_ids = grid.index
        if params.get("show_progress"):
            from tqdm.auto import tqdm

            cell_ids = tqdm(cell_ids, desc="Predicting cells", unit="cell")
        for idx in cell_ids:
            cell_geom = grid.loc[idx, "geometry"]
            arr = read_cell_array_from_mosaic(mosaic, cell_geom, utm_to_mosaic, nodata)
            if arr is None or arr.size == 0:
                continue

            arr = arr.astype(np.float32) / 255.0
            resized = [
                np.asarray(
                    Image.fromarray(arr[c]).resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), Image.Resampling.BILINEAR)
                )
                for c in range(arr.shape[0])
            ]
            batch = np.stack(resized, axis=0)[np.newaxis, ...].astype(np.float32)

            probs = np.asarray(session.run(None, {input_name: batch})[0]).reshape(-1)
            waste_confidence = float(probs[waste_idx])
            label = "waste" if waste_confidence >= confidence_threshold else "background"

            geom_wgs84 = shapely_transform(lambda x, y, _z=None: to_wgs84.transform(x, y), cell_geom)
            features.append(
                {
                    "type": "Feature",
                    "properties": {
                        "cell_id": int(grid.loc[idx, "cell_id"]),
                        "label": label,
                        "confidence": round(waste_confidence, 4),
                    },
                    "geometry": mapping(geom_wgs84),
                }
            )
    return {"type": "FeatureCollection", "features": features}


def dataset_cache_dir(chips_path: str, labels_path: str, threshold: float, cell_size_m: float) -> Path:
    key = hashlib.sha256(f"{chips_path}|{labels_path}|{threshold}|{cell_size_m}".encode()).hexdigest()[:16]
    return Path(tempfile.gettempdir()) / f"yolo_cls_dataset_{key}"


def _subset_chips_dir(chips_path: str, fraction: float) -> str:
    """Return an evenly distributed, deterministic subset of chip files."""
    if not 0.0 < fraction <= 1.0:
        msg = "sample_fraction must be greater than 0 and no greater than 1"
        raise ValueError(msg)
    if fraction == 1.0:
        return chips_path
    from fair.utils.data import resolve_directory

    chips = sorted(resolve_directory(chips_path).rglob("*.tif"))
    sample_size = min(len(chips), max(1, round(len(chips) * fraction)))
    if sample_size == len(chips):
        return chips_path

    subset = Path(tempfile.mkdtemp(prefix="yolo_chips_subset_"))
    if sample_size == 1:
        selected_chips = [chips[0]]
    else:
        selected_chips = [chips[round(index * (len(chips) - 1) / (sample_size - 1))] for index in range(sample_size)]
    for chip in selected_chips:
        (subset / chip.name).symlink_to(chip)
        sidecar = chip.with_name(chip.name + ".aux.xml")
        if sidecar.exists():
            (subset / sidecar.name).symlink_to(sidecar)
    return str(subset)


def pick_utm_crs(lon: float, lat: float):
    import geopandas as gpd

    gdf = gpd.GeoSeries(gpd.points_from_xy([lon], [lat]), crs="EPSG:4326")
    return gdf.estimate_utm_crs()


def build_mosaic(chip_paths: list[Path]) -> Path:
    import rasterio
    from rasterio.merge import merge

    out_dir = Path(tempfile.mkdtemp(prefix="yolo_cls_mosaic_"))
    out_path = out_dir / "mosaic.tif"

    with rasterio.open(chip_paths[0]) as ref:
        profile = ref.profile

    photometric = "rgb" if profile.get("count", 1) >= 3 else "minisblack"

    merge(
        [str(p) for p in chip_paths],
        dst_path=str(out_path),
        dst_kwds={
            **profile,
            "driver": "GTiff",
            "tiled": True,
            "blockxsize": 512,
            "blockysize": 512,
            "compress": "deflate",
            "photometric": photometric,
        },
        mem_limit=512,
    )
    return out_path


def load_labels(labels_path: Path, target_crs):
    import geopandas as gpd

    labels = gpd.read_file(labels_path)
    if labels.crs is None:
        labels = labels.set_crs("EPSG:4326")
    labels = labels.to_crs(target_crs)

    if "label" not in labels.columns:
        labels["label"] = 1
    return labels.reset_index(drop=True)


def load_labels_merged(labels_path: Path, target_crs):
    labels = load_labels(labels_path, target_crs)

    waste = labels[labels["label"] == 1]
    background = labels[labels["label"] == 0]
    return waste.geometry.union_all(), None if background.empty else background.geometry.union_all()


def split_cells(cells, source_polygons, val_ratio: float, test_ratio: float, seed: int) -> dict[int, str]:
    """Split complete source label-polygon groups; splits per-cell if there are no source polygons to group by."""
    if source_polygons.empty:
        groups = [[cell_id] for cell_id in cells.index]
    else:
        groups_by_polygon: dict[int, list[int]] = {}
        for cell_id, cell in cells.iterrows():
            overlap = source_polygons.geometry.intersection(cell.geometry).area
            if overlap.max() <= 0:
                msg = "Selected cells must overlap a source label polygon"
                raise ValueError(msg)
            polygon_id = int(overlap.idxmax())
            groups_by_polygon.setdefault(polygon_id, []).append(cell_id)
        groups = list(groups_by_polygon.values())

    random.Random(seed).shuffle(groups)
    val_count = round(len(groups) * val_ratio)
    test_count = round(len(groups) * test_ratio)
    val_count = max(1, val_count) if val_ratio else 0
    test_count = max(1, test_count) if test_ratio else 0
    if val_count + test_count >= len(groups):
        msg = "Not enough source label polygons to create train, validation, and test splits"
        raise ValueError(msg)

    split_by_cell = {cell_id: "train" for group in groups for cell_id in group}
    for group in groups[:val_count]:
        split_by_cell.update(dict.fromkeys(group, "val"))
    for group in groups[val_count : val_count + test_count]:
        split_by_cell.update(dict.fromkeys(group, "test"))
    return split_by_cell


def build_grid_gdf(bounds_proj, cell_size: float, crs):
    import math

    import geopandas as gpd
    from shapely.geometry import box

    minx, miny, maxx, maxy = bounds_proj
    cols = max(1, math.ceil((maxx - minx) / cell_size))
    rows = max(1, math.ceil((maxy - miny) / cell_size))

    cells = []
    cell_id = 0
    for r in range(rows):
        for c in range(cols):
            x = minx + c * cell_size
            y = miny + r * cell_size
            cells.append({"cell_id": cell_id, "geometry": box(x, y, x + cell_size, y + cell_size)})
            cell_id += 1
    return gpd.GeoDataFrame(cells, crs=crs)


def classify_cells(
    grid_gdf,
    labels_union,
    threshold: float,
    background_union=None,
    seed: int = 42,
    mosaic_ds=None,
    utm_to_mosaic=None,
    nodata=None,
):
    grid_gdf = grid_gdf.copy()
    intersections = grid_gdf.geometry.intersection(labels_union)
    grid_gdf["overlap_fraction"] = (intersections.area / grid_gdf.geometry.area).fillna(0.0)
    grid_gdf["label"] = (grid_gdf["overlap_fraction"] >= threshold).astype(int)

    waste = grid_gdf[grid_gdf["label"] == 1]
    background = None
    if background_union is not None:
        background_intersections = grid_gdf.geometry.intersection(background_union)
        background_overlap = (background_intersections.area / grid_gdf.geometry.area).fillna(0.0)
        background = grid_gdf[(background_overlap >= threshold) & (grid_gdf["label"] == 0)]

    if background is None or background.empty:
        candidates = grid_gdf[grid_gdf["label"] == 0]
        if nodata is not None and mosaic_ds is not None and utm_to_mosaic is not None:
            candidates = candidates[
                candidates.geometry.apply(lambda geom: cell_nodata_share(mosaic_ds, geom, utm_to_mosaic, nodata) <= 0.2)
            ]
        background = candidates.sample(n=min(len(waste), len(candidates)), random_state=seed)

    if background.empty:
        msg = "No background cells available"
        raise ValueError(msg)
    return grid_gdf.loc[[*waste.index, *background.index]].copy()


def cell_nodata_share(mosaic_ds, cell_geom, utm_to_mosaic, nodata) -> float:
    arr = read_cell_array_from_mosaic(mosaic_ds, cell_geom, utm_to_mosaic, nodata)
    if arr is None or arr.size == 0:
        return 1.0
    return float(((arr == nodata).all(axis=0)).mean())


def read_cell_array_from_mosaic(mosaic_ds, cell_geom, utm_to_mosaic, nodata) -> Any:
    from rasterio.windows import from_bounds

    minx, miny, maxx, maxy = cell_geom.bounds
    xs = [minx, minx, maxx, maxx]
    ys = [miny, maxy, miny, maxy]
    lons, lats = utm_to_mosaic.transform(xs, ys)
    west, east = min(lons), max(lons)
    south, north = min(lats), max(lats)

    window = from_bounds(west, south, east, north, transform=mosaic_ds.transform)

    arr = mosaic_ds.read(
        [1, 2, 3],
        window=window,
        boundless=True,
        fill_value=0,
        out_dtype="uint8",
    )
    if arr.size == 0:
        return None

    if nodata is not None:
        mask = (arr == nodata).all(axis=0)
        if mask.mean() > 0.8:  # min covered area, same as min threshold as for waste intersetcion maybe?
            return None
    return arr


def save_arr_as_png(arr: Any, out_path: Path) -> None:
    import numpy as np
    from PIL import Image

    rgb = np.transpose(arr, (1, 2, 0)).astype(np.uint8)
    Image.fromarray(rgb, mode="RGB").save(out_path, format="PNG")


def resolve_chip_paths(chips_path: str) -> list[Path]:
    from fair.utils.data import resolve_directory

    return sorted(resolve_directory(chips_path).rglob("*.tif"))


def resolve_label_file(labels_path: str) -> Path:
    from fair.utils.data import resolve_directory

    labels_dir = resolve_directory(labels_path)
    if labels_dir.is_file():
        return labels_dir
    return sorted([*labels_dir.rglob("*.geojson"), *labels_dir.rglob("*.gpkg")])[0]


def reset_yolo_dirs(yolo_dir: Path) -> None:
    if yolo_dir.exists():
        shutil.rmtree(yolo_dir)
    for split in ("train", "val", "test"):
        for cls in CLASS_NAMES:
            (yolo_dir / split / cls).mkdir(parents=True)


def projected_bounds(bounds, source_crs, target_crs):
    from pyproj import Transformer

    to_proj = Transformer.from_crs(source_crs, target_crs, always_xy=True)
    xs = [bounds.left, bounds.left, bounds.right, bounds.right]
    ys = [bounds.bottom, bounds.top, bounds.bottom, bounds.top]
    px, py = to_proj.transform(xs, ys)
    return min(px), min(py), max(px), max(py)


def _prepare_yolo_classification_dataset(
    chips_path: str,
    labels_path: str,
    waste_overlap_threshold: float,
    cell_size_m: float = CELL_SIZE_M,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[Path, dict[str, int], dict[str, int], dict[str, int]]:
    import rasterio
    from pyproj import Transformer

    yolo_dir = dataset_cache_dir(chips_path, labels_path, waste_overlap_threshold, cell_size_m)
    chip_paths = resolve_chip_paths(chips_path)
    label_file = resolve_label_file(labels_path)
    reset_yolo_dirs(yolo_dir)
    train_counts: dict[str, int] = {cls: 0 for cls in CLASS_NAMES}
    val_counts: dict[str, int] = {cls: 0 for cls in CLASS_NAMES}
    test_counts: dict[str, int] = {cls: 0 for cls in CLASS_NAMES}

    with rasterio.open(build_mosaic(chip_paths)) as mosaic:
        mosaic_crs = mosaic.crs
        bounds = mosaic.bounds
        nodata = mosaic.nodata

        centroid_lon = (bounds.left + bounds.right) / 2
        centroid_lat = (bounds.bottom + bounds.top) / 2
        target_crs = pick_utm_crs(centroid_lon, centroid_lat) if mosaic_crs.is_geographic else mosaic_crs

        grid = build_grid_gdf(projected_bounds(bounds, mosaic_crs, target_crs), cell_size_m, target_crs)
        utm_to_mosaic = Transformer.from_crs(target_crs, mosaic_crs, always_xy=True)
        label_polygons = load_labels(label_file, target_crs)
        waste_polygons = label_polygons[label_polygons["label"] == 1]
        if waste_polygons.empty:
            msg = "At least one label=1 waste polygon is required"
            raise ValueError(msg)
        waste_union = waste_polygons.geometry.union_all()
        background_polygons = label_polygons[label_polygons["label"] == 0]
        background_union = None if background_polygons.empty else background_polygons.geometry.union_all()
        grid = classify_cells(
            grid,
            waste_union,
            waste_overlap_threshold,
            background_union=background_union,
            seed=seed,
            mosaic_ds=mosaic,
            utm_to_mosaic=utm_to_mosaic,
            nodata=nodata,
        )
        for label_value, cls_name in enumerate(CLASS_NAMES):
            class_cells = grid[grid["label"] == label_value]
            source_polygons = waste_polygons if label_value == 1 else background_polygons
            split_by_cell = split_cells(class_cells, source_polygons, val_ratio, test_ratio, seed)

            for cell_id, cell in class_cells.iterrows():
                arr = read_cell_array_from_mosaic(mosaic, cell.geometry, utm_to_mosaic, nodata)
                if arr is None or arr.size == 0:
                    continue
                split = split_by_cell[cell_id]
                out_path = yolo_dir / split / cls_name / f"cell_{int(cell.cell_id):08d}.png"
                save_arr_as_png(arr, out_path)
                if split == "val":
                    val_counts[cls_name] += 1
                elif split == "test":
                    test_counts[cls_name] += 1
                else:
                    train_counts[cls_name] += 1

    return yolo_dir, train_counts, val_counts, test_counts


def _resolve_yolo_dir_for_step(
    dataset_chips: str,
    dataset_labels: str,
    hyperparameters: dict[str, Any],
    split_info: dict[str, Any],
) -> Path:
    """Resolve the generated dataset, rebuilding it in an isolated step container."""
    yolo_dir = Path(split_info["_yolo_dir"])
    if all((yolo_dir / split / cls).is_dir() for split in ("train", "val", "test") for cls in CLASS_NAMES):
        return yolo_dir

    chips_path = _subset_chips_dir(dataset_chips, hyperparameters.get("sample_fraction", 1.0))
    yolo_dir, _, _, _ = _prepare_yolo_classification_dataset(
        chips_path,
        dataset_labels,
        waste_overlap_threshold=split_info.get("waste_overlap_threshold", 0.8),
        cell_size_m=split_info.get("cell_size_m", CELL_SIZE_M),
        val_ratio=split_info["val_ratio"],
        test_ratio=split_info["test_ratio"],
        seed=split_info.get("seed", 42),
    )
    return yolo_dir


@step
def split_dataset(
    dataset_chips: str,
    dataset_labels: str,
    hyperparameters: dict[str, Any],
) -> Annotated[dict[str, Any], "split_info_artifact"]:
    val_ratio = hyperparameters.get("val_ratio", 0.1)
    test_ratio = hyperparameters.get("test_ratio", 0.1)
    seed = hyperparameters.get("split_seed", 42)
    waste_overlap_threshold = hyperparameters.get("waste_overlap_threshold", 0.8)
    cell_size_m = hyperparameters.get("cell_size_m", CELL_SIZE_M)

    chips_path = _subset_chips_dir(dataset_chips, hyperparameters.get("sample_fraction", 1.0))
    yolo_dir, train_counts, val_counts, test_counts = _prepare_yolo_classification_dataset(
        chips_path,
        dataset_labels,
        waste_overlap_threshold=waste_overlap_threshold,
        cell_size_m=cell_size_m,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    train_count = sum(train_counts.values())
    val_count = sum(val_counts.values())
    test_count = sum(test_counts.values())
    split_info = {
        "strategy": "grid_5m_stratified_grouped",
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "seed": seed,
        "train_count": train_count,
        "val_count": val_count,
        "test_count": test_count,
        "train_counts_per_class": train_counts,
        "val_counts_per_class": val_counts,
        "test_counts_per_class": test_counts,
        "waste_overlap_threshold": waste_overlap_threshold,
        "cell_size_m": cell_size_m,
        "class_names": list(CLASS_NAMES),
        "description": (
            f"5 m x 5 m grid over the chip mosaic; cells with >= "
            f"{waste_overlap_threshold:.0%} label coverage are 'waste', "
            f"else 'background'. Waste and background cells are grouped by their source "
            f"label polygon. Validation and test target "
            f"{val_ratio:.0%} and {test_ratio:.0%} of source-polygon groups."
        ),
        "_yolo_dir": str(yolo_dir),
    }
    log_metadata(metadata={"fair/split": {k: v for k, v in split_info.items() if not k.startswith("_")}})
    return split_info


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
) -> Annotated[Any, "trained_model_artifact"]:
    from ultralytics import YOLO
    from ultralytics import settings as yolo_settings

    epochs = hyperparameters["epochs"]
    batch_size = hyperparameters.get("batch_size", 4)
    chip_size = hyperparameters.get("chip_size", MODEL_INPUT_SIZE)
    learning_rate = hyperparameters.get("learning_rate", 0.001)
    weight_decay = hyperparameters.get("weight_decay", 0.0001)
    optimizer = hyperparameters.get("optimizer", "AdamW")
    use_cos_scheduler = hyperparameters.get("scheduler", "cosine")
    freeze_layers = hyperparameters.get("freeze_layers", 10)

    if use_cos_scheduler not in {"cosine", "none"}:
        msg = "use_cos_scheduler must be 'cosine' or 'none'"
        raise ValueError(msg)

    yolo_dir = _resolve_yolo_dir_for_step(
        dataset_chips,
        dataset_labels,
        hyperparameters,
        split_info,
    )

    yolo_settings.update({"mlflow": False})
    local_weights = _download_checkpoint(base_model_weights)
    device = _get_device()

    with mlflow_training_context(
        hyperparameters,
        model_name,
        base_model_id,
        dataset_id,
    ):
        model = YOLO(str(local_weights), task="classify")
        results = model.train(
            data=str(yolo_dir),
            epochs=epochs,
            batch=batch_size,
            imgsz=chip_size,
            device=device,
            lr0=learning_rate,
            weight_decay=weight_decay,
            optimizer=optimizer,
            freeze=freeze_layers,
            cos_lr=use_cos_scheduler == "cosine",
            verbose=False,
        )
        if results and hasattr(results, "results_dict"):
            top1 = results.results_dict.get("metrics/accuracy_top1", 0.0)
            log_metadata(metadata={"accuracy_top1": float(top1), "epoch": epochs})

        _log_yolo_loss_history(model)

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
    chip_size = hyperparameters.get("chip_size", MODEL_INPUT_SIZE)

    yolo_dir = _resolve_yolo_dir_for_step(
        dataset_chips,
        dataset_labels,
        hyperparameters,
        split_info,
    )

    model = _restore_checkpoint(trained_model)
    results = model.val(data=str(yolo_dir), imgsz=chip_size, verbose=False, split="test")

    if not hasattr(results, "results_dict") or not results.results_dict:
        msg = "YOLO validation produced no results"
        raise RuntimeError(msg)

    accuracy = results.results_dict.get("metrics/accuracy_top1")
    if accuracy is None:
        msg = "YOLO validation did not report top-1 accuracy"
        raise RuntimeError(msg)
    metrics_dict: dict[str, Any] = {"accuracy": float(accuracy)}
    log_evaluation_results(metrics_dict)
    return metrics_dict


@step(output_materializers={"onnx_model": ONNXMaterializer})
def export_onnx(trained_model: Any) -> Annotated[bytes, "onnx_model"]:
    import onnx

    model = _restore_checkpoint(trained_model)
    onnx_path = model.export(format="onnx")
    proto = onnx.load(onnx_path)
    onnx.save_model(proto, onnx_path, save_as_external_data=False)
    onnx.checker.check_model(onnx_path)
    try:
        return Path(onnx_path).read_bytes()
    finally:
        Path(onnx_path).unlink(missing_ok=True)


@step
def run_inference(
    model_uri: str,
    input_images: str,
    inference_params: dict[str, Any],
) -> Annotated[dict[str, Any], "predictions"]:
    from fair.serve.base import load_session

    session = load_session(model_uri)
    return predict(session, input_images, inference_params)


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
    run_inference(
        model_uri=model_uri,
        input_images=input_images,
        inference_params=inference_params,
    )
