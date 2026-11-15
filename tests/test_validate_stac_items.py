import importlib.util
import json
import shutil
from pathlib import Path

import pystac
import pytest

_spec = importlib.util.spec_from_file_location(
    "validate_stac_items", Path(__file__).parent.parent / "scripts" / "validate_stac_items.py"
)
assert _spec is not None and _spec.loader is not None
validate_stac_items = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(validate_stac_items)

_MODEL = Path(__file__).parent.parent / "models" / "sklearn_rgb_segmentation"


@pytest.fixture
def model_dir(tmp_path: Path) -> Path:
    target = tmp_path / "sklearn_rgb_segmentation"
    shutil.copytree(_MODEL / "sample", target / "sample")
    shutil.copy(_MODEL / "stac-item.json", target / "stac-item.json")
    return target


def _errors(model_dir: Path) -> list[str]:
    model = pystac.Item.from_file(str(model_dir / "stac-item.json"))
    return validate_stac_items.sample_errors(model, model_dir)


def test_shipped_sample_passes(model_dir: Path) -> None:
    assert _errors(model_dir) == []


def test_missing_sample_fails(model_dir: Path) -> None:
    shutil.rmtree(model_dir / "sample")

    assert any("no sample dataset" in error for error in _errors(model_dir))


def test_sample_with_another_geometry_fails(model_dir: Path) -> None:
    item_path = model_dir / "sample" / "stac-item.json"
    item = json.loads(item_path.read_text())
    item["properties"]["keywords"] = ["building", "semantic-segmentation", "line"]
    item["properties"]["fair:geometry_type"] = "line"
    item_path.write_text(json.dumps(item))

    assert any("Geometry type mismatch" in error for error in _errors(model_dir))


def test_sample_without_chips_fails(model_dir: Path) -> None:
    for chip in (model_dir / "sample" / "chips-image").glob("*.tif"):
        chip.unlink()

    assert any("no .tif chips" in error for error in _errors(model_dir))


def test_sample_without_labels_file_fails(model_dir: Path) -> None:
    (model_dir / "sample" / "labels" / "labels.geojson").unlink()

    assert any("labels file is missing" in error for error in _errors(model_dir))


def test_oversized_sample_fails(model_dir: Path) -> None:
    (model_dir / "sample" / "padding.bin").write_bytes(b"0" * (validate_stac_items.SAMPLE_MAX_BYTES + 1))

    assert any("over the" in error for error in _errors(model_dir))


def test_sample_classes_outside_the_model_fail(model_dir: Path) -> None:
    item_path = model_dir / "sample" / "stac-item.json"
    item = json.loads(item_path.read_text())
    item["properties"]["label:properties"] = ["highway"]
    item["properties"]["label:classes"] = [{"name": "highway", "classes": ["primary"]}]
    item_path.write_text(json.dumps(item))

    assert any("do not fit the model classes" in error and "highway=primary" in error for error in _errors(model_dir))
