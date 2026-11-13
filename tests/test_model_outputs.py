import json
from pathlib import Path

import pytest

from fair.stac.validators import prediction_output

MODEL_ITEMS = sorted(Path(__file__).resolve().parents[1].glob("models/*/stac-item.json"))


@pytest.mark.parametrize("item_path", MODEL_ITEMS, ids=lambda path: path.parent.name)
def test_model_predictions_leave_out_a_label_that_repeats_the_class(item_path: Path) -> None:
    output = prediction_output(json.loads(item_path.read_text())["properties"])

    assert output is not None
    assert "label" not in output["variables"]
