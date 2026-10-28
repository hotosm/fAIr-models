import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from pyproj import Transformer
from rasterio.crs import CRS
from rasterio.features import rasterize
from rasterio.transform import from_origin
from shapely.geometry import LineString, mapping

CHIP_SIZE = 256
GRID = 2
RESOLUTION_M = 0.3
# EPSG:3857, upper left of the chip grid. predict fetches the real OSM roads of the chips' footprint, so this sits in
# Banepa (inside the CI smoke-test bbox) where about 14 residential, unclassified, track and path ways are mapped.
ORIGIN_X, ORIGIN_Y = 9_520_000.0, 3_203_000.0
EXTENT_M = GRID * CHIP_SIZE * RESOLUTION_M

# Parallel roads 17 m apart (label 0 unpaved, 1 paved) and one unlabelled cross street (junction context only).
ROAD_LABELS = [0, 1, 0, 1, 0, 1, 0, 1]
ROAD_COLORS = {0: (150, 110, 70), 1: (90, 90, 95)}
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
_TO_WGS84 = Transformer.from_crs(3857, 4326, always_xy=True)


def _road_lines() -> list:
    roads = []
    for i, label in enumerate(ROAD_LABELS):
        y = ORIGIN_Y - 10 - 17 * i
        roads.append((LineString([(ORIGIN_X + 10, y), (ORIGIN_X + EXTENT_M - 10, y)]), label))
    cross_x = ORIGIN_X + EXTENT_M / 2
    roads.append((LineString([(cross_x, ORIGIN_Y - 5), (cross_x, ORIGIN_Y - EXTENT_M + 5)]), None))
    return roads


def _to_wgs84(line: LineString) -> LineString:
    xs, ys = _TO_WGS84.transform(*line.xy)
    return LineString(zip(xs, ys, strict=True))


def create_toy_data(root: Path) -> dict:
    rng = np.random.default_rng(0)
    size = GRID * CHIP_SIZE
    transform = from_origin(ORIGIN_X, ORIGIN_Y, RESOLUTION_M, RESOLUTION_M)
    # Vegetation-like background, kept clear of the near-black/near-white values materialize treats as nodata.
    mosaic = np.stack([rng.integers(low, low + 40, (size, size)) for low in (60, 100, 50)]).astype(np.uint8)
    for line, label in _road_lines():
        color = ROAD_COLORS.get(label, (120, 120, 120))
        mask = rasterize([line.buffer(3.0)], out_shape=(size, size), transform=transform).astype(bool)
        for band, value in enumerate(color):
            mosaic[band][mask] = value + rng.integers(-10, 10, int(mask.sum()))

    chips_dir = root / "chips"
    chips_dir.mkdir()
    for row in range(GRID):
        for col in range(GRID):
            window = mosaic[:, row * CHIP_SIZE : (row + 1) * CHIP_SIZE, col * CHIP_SIZE : (col + 1) * CHIP_SIZE]
            chip_transform = from_origin(
                ORIGIN_X + col * CHIP_SIZE * RESOLUTION_M,
                ORIGIN_Y - row * CHIP_SIZE * RESOLUTION_M,
                RESOLUTION_M,
                RESOLUTION_M,
            )
            with rasterio.open(
                chips_dir / f"OAM-{col}-{row}-20.tif",
                "w",
                driver="GTiff",
                width=CHIP_SIZE,
                height=CHIP_SIZE,
                count=3,
                dtype="uint8",
                crs=CRS.from_epsg(3857),
                transform=chip_transform,
            ) as dst:
                dst.write(window)

    labels_dir = root / "labels"
    labels_dir.mkdir()
    features = [
        {"type": "Feature", "properties": {"osm_id": 1000 + i, "label": label}, "geometry": mapping(_to_wgs84(line))}
        for i, (line, label) in enumerate(_road_lines())
    ]
    (labels_dir / "labels.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": features}))

    stac_path = root / "dataset-stac-item.json"
    stac_path.write_text(json.dumps(_build_dataset_stac_item(chips_dir, labels_dir), indent=2))
    return {"chips": chips_dir, "labels": labels_dir, "dataset_stac_item": stac_path}


@pytest.fixture(scope="session")
def weights(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The head, encoder and replay bank the STAC item points to, downloaded once into fAIr's cache."""
    import zipfile

    from fair.utils.data import resolve_path

    item = json.loads((DATA_DIR.parent / "stac-item.json").read_text())
    hyperparameters = item["properties"]["mlm:hyperparameters"]
    root = tmp_path_factory.mktemp("weights")
    (root / "head.pt").symlink_to(resolve_path(item["assets"]["checkpoint"]["href"]))
    with zipfile.ZipFile(resolve_path(hyperparameters["training.encoder_uri"])) as archive:
        archive.extract("encoder.pt", root)
    (root / "replay_bank_2000_features.pt").symlink_to(resolve_path(hyperparameters["training.replay_bank_uri"]))
    return root


@pytest.fixture(scope="session")
def example_pair(tmp_path_factory: pytest.TempPathFactory, weights: Path) -> dict:
    """data/example_pair (one OAM scene + labelled roads) in fAIr's chips/labels layout, plus the released weights."""
    import geopandas as gpd

    if not (DATA_DIR / "example_pair").exists():
        pytest.skip(f"{DATA_DIR / 'example_pair'} not available")
    root = tmp_path_factory.mktemp("example_pair")
    (root / "chips").mkdir()
    (root / "labels").mkdir()
    for tif in (DATA_DIR / "example_pair").glob("*.tif"):
        (root / "chips" / tif.name).symlink_to(tif)
    for gpkg in (DATA_DIR / "example_pair").glob("*.gpkg"):
        gpd.read_file(gpkg).to_file(root / "labels" / "labels.geojson", driver="GeoJSON")
    return {"chips": root / "chips", "labels": root / "labels", "weights": weights}


@pytest.fixture(scope="session")
def generate_toy_dataset(tmp_path_factory: pytest.TempPathFactory) -> dict:
    return create_toy_data(tmp_path_factory.mktemp("toy_dinov3_road_surface"))


def _build_dataset_stac_item(chips_dir: Path, labels_dir: Path) -> dict:
    west, north = _TO_WGS84.transform(ORIGIN_X, ORIGIN_Y)
    east, south = _TO_WGS84.transform(ORIGIN_X + EXTENT_M, ORIGIN_Y - EXTENT_M)
    return {
        "type": "Feature",
        "stac_version": "1.1.0",
        "stac_extensions": ["https://stac-extensions.github.io/label/v1.0.1/schema.json"],
        "id": "toy-dinov3-road-surface",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[west, south], [east, south], [east, north], [west, north], [west, south]]],
        },
        "bbox": [west, south, east, north],
        "properties": {
            "datetime": "2026-10-14T00:00:00Z",
            "description": "Toy road surface dataset",
            "label:type": "vector",
            "label:tasks": ["classification"],
            "label:classes": [{"name": "label", "classes": ["0", "1"]}],
            "label:description": "Road lines labelled 0 unpaved, 1 paved",
            "keywords": ["road"],
            "fair:user_id": "test",
            "version": "1",
            "deprecated": False,
            "license": "CC-BY-4.0",
            "providers": [{"name": "HOTOSM", "roles": ["producer"], "url": "https://www.hotosm.org"}],
        },
        "assets": {
            "chips": {"href": str(chips_dir), "type": "image/tiff", "roles": ["data"]},
            "labels": {"href": str(labels_dir), "type": "application/geo+json", "roles": ["labels"]},
        },
        "links": [],
    }
