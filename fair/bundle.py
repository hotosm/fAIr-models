import io
import json
import logging
import subprocess
import tempfile
from collections import Counter
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from upath import UPath

logger = logging.getLogger(__name__)

LABELS_LAYER = "labels"
LABELS_PARQUET = "labels.parquet"
TILES_PARQUET = "tiles.parquet"
IMAGERY_PMTILES = "imagery.pmtiles"
LABELS_PMTILES = "labels.pmtiles"
IMAGERY_QUALITIES: dict[str, int | None] = {"lossless": None, "high": 90, "medium": 80}
DEFAULT_IMAGERY_QUALITY = "high"

Chips = dict[str, Path | UPath]
Bounds = tuple[float, float, float, float]


def index_chips(chip_paths: Iterable[Path | UPath]) -> Chips:
    """One `.tif` chip per chip id (its file stem); the first path wins a shared id."""
    chips: Chips = {}
    for chip in chip_paths:
        if chip.suffix.lower() in {".tif", ".tiff"}:
            chips.setdefault(chip.stem, chip)
    if not chips:
        raise ValueError("No .tif chips found")
    return chips


def collect_chips(chips_dirs: Iterable[UPath]) -> Chips:
    return index_chips(chip for chips_dir in chips_dirs for chip in chips_dir.iterdir())


@contextmanager
def _local_chip(chip: Path | UPath) -> Iterator[Path]:
    # geomltoolkits georeferences JPEG chips in a GDAL .aux.xml sidecar, found only beside the opened file.
    if isinstance(chip, Path):
        yield chip
        return
    with tempfile.TemporaryDirectory(prefix="fair-chip-") as tmp:
        chip.fs.get(f"{chip.path}*", f"{tmp}/")
        yield Path(tmp) / chip.name


def chip_bounds(chips: Chips) -> dict[str, Bounds]:
    import rasterio
    from rasterio.warp import transform_bounds

    bounds = {}
    for chip_id, chip in chips.items():
        with _local_chip(chip) as path, rasterio.open(path) as src:
            if src.crs is None:
                raise ValueError(f"Chip {chip.name} is not georeferenced")
            bounds[chip_id] = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
    return bounds


def grid_tile(bounds: Bounds) -> Any:
    """The web-mercator `mercantile.Tile` that EPSG:4326 `bounds` cover exactly, else None."""
    import mercantile

    west, south, east, north = bounds
    inset = (east - west) * 1e-3
    tile = mercantile.bounding_tile(west + inset, south + inset, east - inset, north - inset)
    return tile if all(abs(a - b) <= inset for a, b in zip(bounds, mercantile.bounds(tile), strict=True)) else None


def grid_tiles(bounds: dict[Any, Bounds]) -> dict[Any, Any] | None:
    """Each key by its `mercantile.Tile` if every bounds is a distinct web-mercator tile at one zoom, else None."""
    tiles = {grid_tile(value): key for key, value in bounds.items()}
    if None in tiles or len(tiles) != len(bounds) or len({tile.z for tile in tiles}) != 1:
        return None
    return tiles


def _rgb8(chip: Path | UPath) -> bool:
    import rasterio

    with _local_chip(chip) as path, rasterio.open(path) as src:
        return src.count >= 3 and src.dtypes[0] == "uint8"


def pmtiles_tile_id(tile: Any) -> int:
    from pmtiles.tile import zxy_to_tileid

    return zxy_to_tileid(tile.z, tile.x, tile.y)


def _webp(chip: Path | UPath, quality: int | None) -> bytes:
    import numpy as np
    import rasterio
    from PIL import Image

    with rasterio.MemoryFile(chip.read_bytes()) as memfile, memfile.open() as src:
        rgb = np.moveaxis(src.read([1, 2, 3]), 0, -1)
    out = io.BytesIO()
    options: dict[str, Any] = {"lossless": True} if quality is None else {"quality": quality}
    Image.fromarray(rgb).save(out, format="WEBP", **options)
    return out.getvalue()


def write_imagery_pmtiles(tiles: dict[Any, Path | UPath], target: Path, quality: str = DEFAULT_IMAGERY_QUALITY) -> None:
    if quality not in IMAGERY_QUALITIES:
        raise ValueError(f"imagery quality must be one of {sorted(IMAGERY_QUALITIES)}, got {quality!r}")
    import mercantile
    from pmtiles.tile import Compression, TileType
    from pmtiles.writer import Writer

    ordered = sorted(tiles, key=pmtiles_tile_id)
    zoom = ordered[0].z
    west, north = mercantile.ul(min(tile.x for tile in ordered), min(tile.y for tile in ordered), zoom)
    east, south = mercantile.ul(max(tile.x for tile in ordered) + 1, max(tile.y for tile in ordered) + 1, zoom)
    with target.open("wb") as handle:
        writer = Writer(handle)
        for tile in ordered:
            writer.write_tile(pmtiles_tile_id(tile), _webp(tiles[tile], IMAGERY_QUALITIES[quality]))
        header = {"tile_type": TileType.WEBP, "tile_compression": Compression.NONE, "center_zoom": zoom}
        corners = {"min_lon_e7": west, "min_lat_e7": south, "max_lon_e7": east, "max_lat_e7": north}
        writer.finalize(
            header | {key: int(value * 1e7) for key, value in corners.items()}, {"zoom": zoom, "tiles": len(ordered)}
        )


def write_labels_parquet(features: list[dict[str, Any]], properties: list[str], target: Path) -> None:
    """Write the labels as GeoParquet 1.1 (bbox column, zstd, Hilbert-sorted), keeping `properties` as strings."""
    import geopandas as gpd
    import pandas as pd

    frame = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326", columns=["geometry"])
    for name in properties:
        values = (feature["properties"].get(name) for feature in features)
        frame[name] = pd.array([None if value is None else str(value) for value in values], dtype="string")
    frame = frame.iloc[frame.geometry.hilbert_distance().argsort()]
    # Settings from https://github.com/opengeospatial/geoparquet/blob/v2.0.0/format-specs/distributing-geoparquet.md
    frame.to_parquet(
        target,
        index=False,
        schema_version="1.1.0",
        write_covering_bbox=True,
        compression="zstd",
        compression_level=15,
        row_group_size=100_000,
    )


def read_labels(data: bytes, area: Any, class_property: str) -> Any:
    """GeoParquet or GeoJSON labels intersecting `area`, in EPSG:4326, with `class_property` only."""
    import geopandas as gpd

    if data[:4] == b"PAR1":
        frame = gpd.read_parquet(io.BytesIO(data))
    else:
        # Parsed with json, not GDAL, which also opens VRT files that read other files on the server.
        try:
            collection = json.loads(data)
        except (ValueError, RecursionError) as error:
            raise ValueError(f"labels are neither GeoParquet nor GeoJSON: {error}") from error
        if not isinstance(collection, dict) or not isinstance(collection.get("features"), list):
            raise ValueError("labels must be a GeoJSON FeatureCollection")
        frame = gpd.GeoDataFrame.from_features(collection["features"], crs="EPSG:4326")
    if class_property not in frame.columns:
        raise ValueError(f"labels have no {class_property!r} column")
    frame = frame.to_crs("EPSG:4326")
    return frame.loc[frame.intersects(area), [class_property, frame.geometry.name]]


def read_tile_ids(tiles_parquet: UPath) -> list[int]:
    """The sorted PMTiles tile ids of a bundle's chips; fails unless they are web-mercator tiles at one zoom."""
    import geopandas as gpd

    frame = gpd.read_parquet(tiles_parquet.path, filesystem=tiles_parquet.fs, columns=["geometry"])
    tiles = grid_tiles(dict(enumerate(frame.geometry.bounds.itertuples(index=False))))
    if tiles is None:
        raise ValueError(f"{tiles_parquet}: the chips are not distinct web-mercator tiles at one zoom")
    return sorted(pmtiles_tile_id(tile) for tile in tiles)


def write_tiles_parquet(
    chips: Chips, bounds: dict[str, Bounds], labels: Path, class_property: str, target: Path
) -> None:
    """Write one row per chip: `chip_id`, footprint, `num_features` and `class_counts` as JSON, Hilbert-sorted."""
    import geopandas as gpd
    import numpy as np
    import pandas as pd
    from shapely.geometry import box

    frame = gpd.GeoDataFrame({"chip_id": list(chips)}, geometry=[box(*bounds[chip]) for chip in chips], crs="EPSG:4326")
    frame = frame.iloc[frame.geometry.hilbert_distance().argsort()].reset_index(drop=True)
    label_frame = gpd.read_parquet(labels, columns=["geometry", class_property])
    chip_rows, label_rows = label_frame.sindex.query(frame.geometry, predicate="intersects")
    classes = pd.Series(label_frame[class_property].to_numpy()[label_rows]).groupby(chip_rows).agg(_class_counts)
    frame["num_features"] = np.bincount(chip_rows, minlength=len(frame))
    frame["class_counts"] = classes.reindex(range(len(frame)), fill_value="{}").to_numpy()
    frame = frame[["chip_id", "num_features", "class_counts", "geometry"]]
    frame.to_parquet(
        target, index=False, schema_version="1.1.0", write_covering_bbox=True, compression="zstd", compression_level=15
    )


def _class_counts(values: Any) -> str:
    return json.dumps(dict(sorted(Counter(str(value) for value in values.dropna()).items())))


def write_labels_pmtiles(labels: Iterable[Path], zoom: int | None, target: Path) -> None:
    """Merge EPSG:4326 GeoParquet or GeoJSON label files into one `labels` layer at `zoom` (None: tippecanoe picks)."""
    import geopandas as gpd
    import pandas as pd

    frames = [gpd.read_parquet(path) if Path(path).suffix == ".parquet" else gpd.read_file(path) for path in labels]
    merged = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs="EPSG:4326")
    merged = merged[merged.geometry.notna() & ~merged.geometry.is_empty]
    zooms = ["-zg"] if zoom is None else ["-z", str(zoom), "-Z", str(zoom)]
    with tempfile.TemporaryDirectory(prefix="fair-labels-") as tmp:
        source = Path(tmp) / "labels.fgb"
        merged.to_file(source, driver="FlatGeobuf")
        flags = ["-l", LABELS_LAYER, "--force", "--no-tile-size-limit", "--no-feature-limit", *zooms]
        try:
            subprocess.run(["tippecanoe", "-o", str(target), *flags, str(source)], check=True, capture_output=True)
        except subprocess.CalledProcessError as error:
            raise RuntimeError(f"tippecanoe failed: {error.stderr.decode()[-2000:]}") from error


def write_bundle(
    chips_by_input: dict[str, Chips],
    features: list[dict[str, Any]],
    properties: list[str],
    out_dir: Path,
    imagery_quality: str = DEFAULT_IMAGERY_QUALITY,
) -> dict[str, Path]:
    """Write `labels.parquet`, `tiles.parquet`, `labels.pmtiles`, and imagery PMTiles for each tile-grid input."""
    paths = {
        "labels_parquet": out_dir / LABELS_PARQUET,
        "tiles_parquet": out_dir / TILES_PARQUET,
        "labels_pmtiles": out_dir / LABELS_PMTILES,
    }
    write_labels_parquet(features, properties, paths["labels_parquet"])
    zoom = None
    for index, (name, chips) in enumerate(chips_by_input.items()):
        bounds = chip_bounds(chips)
        tiles = grid_tiles(bounds)
        if index == 0:
            write_tiles_parquet(chips, bounds, paths["labels_parquet"], properties[0], paths["tiles_parquet"])
            zoom = next(iter(tiles)).z if tiles else None
        if tiles is None or not all(_rgb8(chip) for chip in chips.values()):
            logger.warning(
                "Input %s: chips are not distinct 8-bit RGB web-mercator tiles at one zoom; skipping imagery", name
            )
            continue
        single = len(chips_by_input) == 1
        key = "imagery_pmtiles" if single else f"imagery_pmtiles_{name}"
        paths[key] = out_dir / (IMAGERY_PMTILES if single else f"imagery-{name}.pmtiles")
        write_imagery_pmtiles({tile: chips[chip_id] for tile, chip_id in tiles.items()}, paths[key], imagery_quality)
    write_labels_pmtiles([paths["labels_parquet"]], zoom, paths["labels_pmtiles"])
    return paths
