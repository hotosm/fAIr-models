import tempfile
import zipfile
from pathlib import Path
from typing import Annotated, Any

from zenml import log_metadata, pipeline, step

from fair.zenml.instrumentation import log_evaluation_results, mlflow_training_context
from fair.zenml.materializers import ONNXMaterializer
from fair.zenml.metrics import log_loss_history

CLASS_NAMES = ("unpaved", "paved")

LABEL_COLUMN = "label"
SCENE_ID = "scene"
MATERIALIZE_SEED = 0

# The weights derive from DINOv3: its license (1.b.i) requires a copy of it with every copy of the weights.
DINOV3_LICENSE = Path(__file__).with_name("LICENSE-DINOv3.md")

RAW_DATA_API_URL = "https://api-prod.raw-data.hotosm.org/v1"

HIGHWAY_VALUES = (
    "motorway,motorway_link,trunk,trunk_link,primary,primary_link,secondary,secondary_link,tertiary,tertiary_link,"
    "unclassified,residential,living_street,service,track,road,footway,path,pedestrian,cycleway,bridleway"
)
EXCLUDE_TAGS = "footway=sidewalk,footway=crossing"

PATCH_CONFIG_KWARGS: dict[str, Any] = {
    "patch_size_m": 15.0,
    "patch_size_px": 256,
    "meters_per_patch": 20.0,
    "min_patches_per_road": 1,
    "max_patches_per_road": 4,
    "n_bags": 5,
    "endpoint_margin_fraction": 0.1,
    "junction_clearance_m": 7.5,
    "lateral_jitter_m": 1.5,
    "bearing_jitter_deg": 10.0,
    "max_nodata_fraction": 0.1,
    "max_attempts_per_patch": 10,
    "fill_tolerance": 1,
    "junction_exclude_highway_values": [
        "footway",
        "path",
        "cycleway",
        "steps",
        "pedestrian",
        "corridor",
        "platform",
        "bus_stop",
        "crossing",
        "give_way",
        "stop",
        "traffic_signals",
        "traffic_mirror",
        "street_lamp",
        "milestone",
        "emergency_access_point",
        "construction",
    ],
}

# Inference / evaluation: one bag centred along the road (same patch budget), cut on the road centreline.
INFERENCE_PATCH_CONFIG_KWARGS: dict[str, Any] = {
    **PATCH_CONFIG_KWARGS,
    "n_bags": 1,
    "lateral_jitter_m": 0.0,
    "bearing_jitter_deg": 0.0,
}


def build_mosaic(chip_paths: list, out_dir: Path) -> Path:
    import rasterio
    from rasterio.merge import merge

    out_path = out_dir / "mosaic.tif"
    with rasterio.open(chip_paths[0]) as ref:
        profile = ref.profile

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
            "photometric": "rgb",
        },
        mem_limit=512,
    )
    return out_path


def load_labelled_roads(dataset_labels: str, id_column: str) -> Any:
    """The labelled roads with the `road_id` materialize gives them (it selects them with the same functions) and
    their `id_column` value, e.g. the OSM way id; several roads can share one id."""
    from road_surface_classifier.data.materialize import load_lines, roads_to_cut

    from fair.utils.data import resolve_directory

    lines = load_lines(resolve_directory(dataset_labels, "*.geojson") / "labels.geojson", "EPSG:4326")
    roads = roads_to_cut(lines, LABEL_COLUMN).rename_axis("road_id").reset_index()
    roads[id_column] = roads["source_row"].map(lines.drop_duplicates("source_row").set_index("source_row")[id_column])
    return roads.drop(columns="source_row").astype({LABEL_COLUMN: int})


def split_road_ids(dataset_labels: str, split_info: dict, split: str) -> list:
    """The `road_id`s of the roads in `split` ("train", "val" or "test") of `split_info`."""
    roads = load_labelled_roads(dataset_labels, split_info["id_column"])
    return roads.loc[roads[split_info["id_column"]].isin(split_info[f"{split}_ids"]), "road_id"].tolist()


def materialize(
    dataset_chips: str,
    dataset_labels: str,
    work_dir: Path,
    patch_config_kwargs: dict = PATCH_CONFIG_KWARGS,
    label_column: str | None = LABEL_COLUMN,
) -> Path:
    """Cut bags along every labelled road into the source repo's dataset layout; deterministic, so each step can
    rebuild it locally. A single bag sits centred along the road. Lines without a label only count as junctions;
    with `label_column=None` every line is cut (inference)."""
    from road_surface_classifier.data.materialize import PatchConfig, materialize_dataset

    from fair.utils.data import resolve_directory

    chips = sorted(resolve_directory(dataset_chips, "*.tif").glob("*.tif"))
    raster = chips[0] if len(chips) == 1 else build_mosaic(chips, work_dir)

    dataset_dir = work_dir / "dataset"
    materialize_dataset(
        [(SCENE_ID, raster, resolve_directory(dataset_labels, "*.geojson") / "labels.geojson")],
        dataset_dir,
        label_column,
        PatchConfig(**patch_config_kwargs),
        seed=MATERIALIZE_SEED,
    )
    return dataset_dir


def road_subset(dataset: Any, road_ids: list) -> Any:
    from torch.utils.data import Subset

    wanted = set(road_ids)
    return Subset(dataset, [i for i, road_id in enumerate(dataset.roads["road_id"]) if road_id in wanted])


def even_clusters(xy: Any, n_clusters: int) -> Any:
    import numpy as np

    cluster = np.zeros(len(xy), dtype=int)

    def bisect(points: Any, first: int, n: int) -> None:
        if n == 1:
            cluster[points] = first
            return
        ordered = points[np.argsort(xy[points, np.ptp(xy[points], axis=0).argmax()], kind="stable")]
        cut = round(len(points) * (n // 2) / n)
        bisect(ordered[:cut], first, n // 2)
        bisect(ordered[cut:], first + n // 2, n - n // 2)

    bisect(np.arange(len(xy)), 0, n_clusters)
    return cluster


def split_roads(roads: Any, id_column: str, val_ratio: float, test_ratio: float, seed: int, n_clusters: int) -> tuple:
    import numpy as np

    ways = roads.dissolve(by=id_column)
    centroids = ways.to_crs(ways.estimate_utm_crs()).geometry.centroid
    n_clusters = min(n_clusters, len(ways))
    cluster_of_way = even_clusters(np.column_stack([centroids.x, centroids.y]), n_clusters)

    order = np.random.default_rng(seed).permutation(n_clusters)
    n_test, n_val = max(1, round(test_ratio * n_clusters)), max(1, round(val_ratio * n_clusters))
    test_ids, val_ids, train_ids = (
        sorted(ways.index[np.isin(cluster_of_way, clusters)].tolist())
        for clusters in (order[:n_test], order[n_test : n_test + n_val], order[n_test + n_val :])
    )
    return train_ids, val_ids, test_ids, n_clusters


def _device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def replay_bags(bank: dict) -> tuple:
    """The replay bank's bags: per-patch backbone features [n_patches, 384] of each bag, and its label. Label 2
    (paved_segmental) is paved."""
    import torch

    patches = bank["patches"]
    bags = {}
    for row, key in enumerate(zip(patches["scene_id"], patches["road_id"], patches["bag"], strict=True)):
        bags.setdefault(key, []).append(row)
    features = bank["features"].float()
    labels = torch.tensor([min(int(patches["label"][rows[0]]), 1) for rows in bags.values()])
    return [features[rows] for rows in bags.values()], labels


def load_dataset(dataset_dir: Path, road_ids: list, seed: int, augment: bool = False) -> Any:
    from road_surface_classifier.data.augmentation import build_bag_augmentations, build_patch_augmentations
    from road_surface_classifier.data.dataset import PatchDataset

    label_map = {i: i for i in range(len(CLASS_NAMES))}
    if augment:
        dataset = PatchDataset(
            dataset_dir,
            label_map,
            seed=seed,
            bag_transform=build_bag_augmentations(),
            patch_transform=build_patch_augmentations(PATCH_CONFIG_KWARGS["patch_size_px"]),
        )
    else:
        dataset = PatchDataset(dataset_dir, label_map, seed=seed, mode="eval")
    return road_subset(dataset, road_ids)


def load_base_model(head_path: Path, encoder_path: Path) -> Any:
    from road_surface_classifier.model import load_model

    model = load_model(head_path, encoder_path, trainable="classifier")
    model.eval().requires_grad_(False)
    model.classifier.requires_grad_(True)
    return model


def evaluate_dataset(model: Any, dataset: Any, device: str) -> tuple:
    """Loss and accuracy of the model on `dataset`, all its bags in one batch."""
    import torch
    import torch.nn.functional as F
    from road_surface_classifier.data.dataset import collate_bags

    images, sizes, labels = collate_bags([dataset[i] for i in range(len(dataset))])
    with torch.no_grad():
        logits = model(images.to(device), sizes)[0].cpu()
    loss = F.cross_entropy(logits, labels, weight=model.class_weights.cpu()).item()
    return loss, (logits.argmax(1) == labels).float().mean().item()


def finetune(
    model: Any,
    train_dataset: Any,
    val_dataset: Any,
    replay_features: list,
    replay_labels: Any,
    epochs: int,
    batch_size: int,
    replay_bags_per_step: int,
    learning_rate: float,
    device: str,
) -> None:
    import mlflow
    import torch
    import torch.nn.functional as F
    from road_surface_classifier.data.dataset import collate_bags

    optimizer = torch.optim.AdamW(
        model.classifier.parameters(), lr=learning_rate, weight_decay=model.hparams.weight_decay
    )
    train_losses, val_losses = [], []
    step = 0
    for _ in range(epochs):
        epoch_losses = []
        order = torch.randperm(len(train_dataset)).tolist()
        for start in range(0, len(order), batch_size):
            images, sizes, labels = collate_bags([train_dataset[i] for i in order[start : start + batch_size]])
            replay = torch.randint(len(replay_labels), (replay_bags_per_step,))
            replay_pooled = torch.stack([model.pool(replay_features[i].to(device))[0] for i in replay.tolist()])
            logits = torch.cat([model(images.to(device), sizes)[0], model.classifier(replay_pooled)])
            labels = torch.cat([labels, replay_labels[replay]]).to(device)
            loss = F.cross_entropy(
                logits, labels, weight=model.class_weights, label_smoothing=model.hparams.label_smoothing
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.classifier.parameters(), 1.0)
            optimizer.step()
            step += 1
            mlflow.log_metric("train_loss", loss.item(), step=step)  # ty: ignore[possibly-missing-attribute]
            epoch_losses.append(loss.item())

        val_loss, val_accuracy = evaluate_dataset(model, val_dataset, device)
        mlflow.log_metrics({"val_loss": val_loss, "val_accuracy": val_accuracy}, step=step)  # ty: ignore[possibly-missing-attribute]
        train_losses.append(sum(epoch_losses) / len(epoch_losses))
        val_losses.append(val_loss)
    log_loss_history(train_losses, val_losses)


def fetch_roads(bbox: list[float], highway_values: list[str], exclude_tags: list[str]) -> Any:
    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    from geomltoolkits.downloader.osm import download_osm_data

    from fair.datasets import _osm_filters

    download = download_osm_data(
        bbox=bbox,
        api_url=RAW_DATA_API_URL,
        filters=_osm_filters([{"name": "highway", "classes": highway_values}], "line"),
        geometry_types=["line"],
        use_st_within=False,  # ST_Intersects: keep the roads crossing the AOI edge
    )
    with ThreadPoolExecutor(1) as pool:
        osm = pool.submit(asyncio.run, download).result()
    return validated_roads(osm, highway_values, exclude_tags)


def validated_roads(osm: Any, highway_values: list[str], exclude_tags: list[str]) -> Any:
    import geopandas as gpd

    roads = gpd.GeoDataFrame.from_features(osm, crs="EPSG:4326", columns=["geometry", "osm_id", "osm_type", "tags"])
    roads = roads[roads["osm_type"] == "ways_line"].assign(highway=roads["tags"].str.get("highway"))
    excluded = roads["tags"].apply(lambda tags: any(f"{key}={value}" in exclude_tags for key, value in tags.items()))
    roads = roads[roads["highway"].isin(highway_values) & ~excluded][["geometry", "osm_id", "highway"]]
    if roads.empty:
        msg = f"No OSM roads to classify: none in highway_values {highway_values} without exclude_tags {exclude_tags}"
        raise ValueError(msg)
    return roads


def predict(session: Any, input_images: str, params: dict) -> dict:
    import geopandas as gpd
    import pandas as pd
    import rasterio
    from road_surface_classifier.data.materialize import PATCHES_FILENAME
    from shapely.geometry import box

    from fair.utils.data import resolve_directory

    with tempfile.TemporaryDirectory() as tmpdir:
        work_dir = Path(tmpdir)
        mosaic_path = build_mosaic(sorted(resolve_directory(input_images, "*.tif").glob("*.tif")), work_dir)
        with rasterio.open(mosaic_path) as mosaic:
            footprint = gpd.GeoSeries([box(*mosaic.bounds)], crs=mosaic.crs)

        utm = footprint.estimate_utm_crs()
        clip_area = footprint.to_crs(utm).buffer(-INFERENCE_PATCH_CONFIG_KWARGS["patch_size_m"] / 2)
        highway_values = params.get("highway_values", HIGHWAY_VALUES).split(",")
        exclude_tags = params.get("exclude_tags", EXCLUDE_TAGS).split(",")
        roads = fetch_roads(list(footprint.to_crs(4326).total_bounds), highway_values, exclude_tags)
        roads = roads.to_crs(utm).clip(clip_area, keep_geom_type=True).explode(ignore_index=True).to_crs(4326)

        labels_dir = work_dir / "labels"
        labels_dir.mkdir()
        roads.to_file(labels_dir / "labels.geojson", driver="GeoJSON")
        dataset_dir = materialize(
            str(work_dir), str(labels_dir), work_dir, INFERENCE_PATCH_CONFIG_KWARGS, label_column=None
        )
        scene_dir = dataset_dir / SCENE_ID

        patches = pd.read_parquet(scene_dir / PATCHES_FILENAME, columns=["road_id", "slot", "path"])
        logits = {}
        for road_id, road_patches in patches.sort_values("slot").groupby("road_id"):
            patch_paths = [scene_dir / path for path in road_patches["path"]]
            logits[road_id] = session.run(None, {"image": preprocess(patch_paths)})[0]

    return postprocess(roads, logits, params["confidence_threshold"])


def preprocess(patch_paths: list) -> Any:
    import numpy as np
    from PIL import Image

    patches = [np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) for path in patch_paths]
    return np.stack(patches).transpose(0, 3, 1, 2) / 255


def postprocess(roads: Any, logits: dict, confidence_threshold: float) -> dict:
    import json

    import numpy as np
    import pandas as pd

    road_ids = list(logits)
    stacked = np.array([logits[road_id][0] for road_id in road_ids]).reshape(-1, len(CLASS_NAMES))
    scores = np.exp(stacked - stacked.max(axis=1, keepdims=True))
    probabilities = scores / scores.sum(axis=1, keepdims=True)
    class_ids = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    known = confidence >= confidence_threshold

    classified = roads.loc[road_ids].assign(
        **{"class": pd.Series(class_ids, index=road_ids, dtype="Int64").where(known)},
        label=np.where(known, np.asarray(CLASS_NAMES)[class_ids], "unknown"),
        confidence=confidence,
    )
    return json.loads(classified.to_json(drop_id=True))


@step
def split_dataset(
    dataset_chips: str,
    dataset_labels: str,
    hyperparameters: dict,
) -> dict:
    id_column = hyperparameters.get("id_column", "osm_id")
    roads = load_labelled_roads(dataset_labels, id_column)

    val_ratio = hyperparameters.get("val_ratio", 0.2)
    test_ratio = hyperparameters.get("test_ratio", 0.2)
    seed = hyperparameters.get("split_seed", 42)
    n_clusters = hyperparameters.get("n_clusters", 10)
    train_ids, val_ids, test_ids, n_clusters = split_roads(roads, id_column, val_ratio, test_ratio, seed, n_clusters)

    def counts_per_class(ids: list) -> dict:
        labels = roads.loc[roads[id_column].isin(ids), LABEL_COLUMN].tolist()
        return {name: labels.count(i) for i, name in enumerate(CLASS_NAMES)}

    split_info = {
        "strategy": "spatial",
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "seed": seed,
        "n_clusters": n_clusters,
        "id_column": id_column,
        "train_count": len(train_ids),
        "val_count": len(val_ids),
        "test_count": len(test_ids),
        "train_counts_per_class": counts_per_class(train_ids),
        "val_counts_per_class": counts_per_class(val_ids),
        "test_counts_per_class": counts_per_class(test_ids),
        "train_ids": train_ids,
        "val_ids": val_ids,
        "test_ids": test_ids,
        "class_names": list(CLASS_NAMES),
        "description": (
            f"Roads grouped by {id_column} (all roads of one id stay together); the ids form {n_clusters} spatial "
            f"clusters of equal size by their centroid (recursive bisection), and in seeded random order test gets "
            f"{test_ratio:.0%} and val {val_ratio:.0%} of the clusters (at least one each), the rest is train."
        ),
    }
    log_metadata(metadata={"fair/split": split_info})
    return split_info


@step
def train_model(
    dataset_chips: str,
    dataset_labels: str,
    base_model_weights: str,
    hyperparameters: dict,
    split_info: dict,
    num_classes: int = 2,
    model_name: str | None = None,
    base_model_id: str | None = None,
    dataset_id: str | None = None,
) -> Annotated[Any, "trained_model_artifact"]:
    import mlflow
    import torch

    from fair.utils.data import resolve_path

    device = _device()
    dataset_dir = materialize(dataset_chips, dataset_labels, Path(tempfile.mkdtemp()))
    train_road_ids, val_road_ids, test_road_ids = (
        split_road_ids(dataset_labels, split_info, split) for split in ("train", "val", "test")
    )
    train_dataset = load_dataset(dataset_dir, train_road_ids, split_info["seed"], augment=True)
    val_dataset = load_dataset(dataset_dir, val_road_ids, split_info["seed"])
    head_path = resolve_path(base_model_weights)
    encoder_path = resolve_path(hyperparameters["encoder_uri"])
    if encoder_path.suffix == ".zip":
        with zipfile.ZipFile(encoder_path) as archive:
            encoder_path = Path(archive.extract("encoder.pt", tempfile.mkdtemp()))
    model = load_base_model(head_path, encoder_path)
    bank = torch.load(resolve_path(hyperparameters["replay_bank_uri"]), map_location="cpu")
    replay_features, replay_labels = replay_bags(bank)
    model.to(device)

    with mlflow_training_context(hyperparameters, model_name, base_model_id, dataset_id):
        all_road_ids = train_road_ids + val_road_ids + test_road_ids
        _, base_accuracy = evaluate_dataset(model, load_dataset(dataset_dir, all_road_ids, split_info["seed"]), device)
        mlflow.log_metric("base_accuracy", base_accuracy)  # ty: ignore[possibly-missing-attribute]
        if base_accuracy >= hyperparameters.get("gate_accuracy", 0.95):
            return model.cpu()

        finetune(
            model,
            train_dataset,
            val_dataset,
            replay_features,
            replay_labels,
            epochs=hyperparameters.get("epochs", 10),
            batch_size=hyperparameters.get("batch_size", 8),
            replay_bags_per_step=hyperparameters.get("replay_bags_per_step", 20),
            learning_rate=hyperparameters.get("learning_rate", 1e-3),
            device=device,
        )

    return model.cpu()


@step
def evaluate_model(
    trained_model: Any,
    dataset_chips: str,
    dataset_labels: str,
    hyperparameters: dict,
    split_info: dict,
    class_names: list | None = None,
) -> dict:

    device = _device()
    model = trained_model.to(device)
    dataset_dir = materialize(dataset_chips, dataset_labels, Path(tempfile.mkdtemp()), INFERENCE_PATCH_CONFIG_KWARGS)
    loss, accuracy = evaluate_dataset(
        model, load_dataset(dataset_dir, split_road_ids(dataset_labels, split_info, "test"), split_info["seed"]), device
    )

    metrics = {"accuracy": accuracy, "loss": loss}
    log_evaluation_results(metrics)
    return metrics


@step(output_materializers={"onnx_model": ONNXMaterializer})
def export_onnx(trained_model: Any) -> Annotated[bytes, "onnx_model"]:
    import onnx
    import torch
    from torch import nn

    class _MainOnly(nn.Module):
        def __init__(self, wrapped: nn.Module) -> None:
            super().__init__()
            self.wrapped = wrapped

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # All patches of x form one bag + drop the attention
            return self.wrapped(x, [x.shape[0]])[0]

    model = _MainOnly(trained_model.cpu().eval()).eval()
    max_patches = INFERENCE_PATCH_CONFIG_KWARGS["max_patches_per_road"]
    size = INFERENCE_PATCH_CONFIG_KWARGS["patch_size_px"]
    dummy = torch.rand(max_patches, 3, size, size)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = str(Path(tmpdir) / "model.onnx")
        torch.onnx.export(
            model,
            (dummy,),
            path,
            input_names=["image"],
            output_names=["logits"],
            dynamic_shapes={"x": {0: torch.export.Dim("patches", min=1, max=max_patches)}},
            dynamo=True,
        )
        onnx_model = onnx.load(path)
        # Bundle the DINOv3 license into the model file, so it travels wherever the ONNX is copied.
        onnx.helper.set_model_props(
            onnx_model, {"license": DINOV3_LICENSE.read_text(encoding="utf-8"), "notice": "Built with DINOv3"}
        )
        onnx.save(onnx_model, path, save_as_external_data=False)
        onnx.checker.check_model(path)
        return Path(path).read_bytes()


@step
def run_inference(
    model_uri: str,
    input_images: str,
    inference_params: dict,
) -> Annotated[dict, "predictions"]:
    from fair.serve.base import load_session

    session = load_session(model_uri)
    return predict(session, input_images, inference_params)


@pipeline
def training_pipeline(
    base_model_weights: str,
    dataset_chips: str,
    dataset_labels: str,
    num_classes: int,
    hyperparameters: dict,
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
    inference_params: dict,
) -> None:
    run_inference(
        model_uri=model_uri,
        input_images=input_images,
        inference_params=inference_params,
    )
