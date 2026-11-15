"""Check example dataset selection without optional model dependencies."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from examples import run


def test_model_example_uses_generated_road_data(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    module_path = tmp_path / "models/chesapeake_rsc_v2/example_data.py"
    module_path.parent.mkdir(parents=True)
    module_path.touch()
    root = Path("examples/data/chesapeake_rsc_v2")
    factory = Mock(return_value={"dataset_stac_item": root / "item.json", "chips": root / "chips"})
    importer = Mock(return_value=SimpleNamespace(create_toy_data=factory))
    monkeypatch.setattr(run.importlib, "import_module", importer)

    dataset, images = run.example_data("chesapeake_rsc_v2", "semantic-segmentation")

    importer.assert_called_once_with("models.chesapeake_rsc_v2.example_data")
    factory.assert_called_once_with(root)
    assert dataset == str(root / "item.json")
    assert images == {"image": str(root / "chips")}


def test_other_examples_keep_the_banepa_dataset(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    importer = Mock(side_effect=AssertionError("Default data needs no model import"))
    monkeypatch.setattr(run.importlib, "import_module", importer)

    dataset, images = run.example_data("sklearn_rgb_segmentation", "semantic-segmentation")

    assert dataset == "data/sample/buildings-banepa-semantic-segmentation/stac-item.json"
    assert images == {"image": "data/sample/test/oam"}
