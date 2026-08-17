"""Unit tests for the grid-cell labeling rules used before training."""

import geopandas as gpd
import pytest
from shapely.geometry import box

from models.yolo_swag_waste_grid_segmentation.pipeline import classify_cells


def _grid(cell_count: int) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {"cell_id": range(cell_count)},
        geometry=[box(index, 0, index + 1, 1) for index in range(cell_count)],
        crs="EPSG:32645",
    )


def test_cells_at_or_above_the_overlap_threshold_are_waste() -> None:
    grid = _grid(3)
    waste = box(0, 0, 0.5, 1).union(box(1, 0, 2, 1))
    explicit_background = box(2, 0, 3, 1)

    cells = classify_cells(grid, waste, threshold=0.5, background_union=explicit_background)

    assert dict(zip(cells.cell_id, cells.label, strict=True)) == {0: 1, 1: 1, 2: 0}


def test_unlabeled_background_is_balanced_and_deterministic() -> None:
    grid = _grid(5)
    waste = box(0, 0, 2, 1)

    first = classify_cells(grid, waste, threshold=1.0, seed=42)
    second = classify_cells(grid, waste, threshold=1.0, seed=42)

    assert first.cell_id.tolist() == second.cell_id.tolist()
    assert (first.label == 1).sum() == 2
    assert (first.label == 0).sum() == 2


def test_explicit_background_excludes_unlabeled_background_cells() -> None:
    grid = _grid(4)
    waste = box(0, 0, 1, 1)
    explicit_background = box(2, 0, 3, 1)

    cells = classify_cells(grid, waste, threshold=1.0, background_union=explicit_background)

    assert dict(zip(cells.cell_id, cells.label, strict=True)) == {0: 1, 2: 0}


def test_grid_without_background_cells_fails_clearly() -> None:
    grid = _grid(2)

    with pytest.raises(ValueError, match="No background cells available"):
        classify_cells(grid, box(0, 0, 2, 1), threshold=1.0)
