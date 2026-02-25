from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from fair.stac.builders import build_base_model_item, build_dataset_item
from fair.zenml.config import generate_inference_config, generate_training_config


def _base_model(**overrides: Any):
    defaults: dict[str, Any] = {
        "item_id": "unet",
        "dt": datetime(2024, 1, 1, tzinfo=UTC),
        "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]},
        "mlm_name": "unet",
        "mlm_architecture": "UNet",
        "mlm_tasks": ["semantic-segmentation"],
        "mlm_framework": "pytorch",
        "mlm_framework_version": "2.1.0",
        "mlm_input": [
            {
                "name": "RGB",
                "bands": [{"name": "red"}, {"name": "green"}, {"name": "blue"}],
                "input": {
                    "shape": [-1, 3, 512, 512],
                    "dim_order": ["batch", "bands", "height", "width"],
                    "data_type": "float32",
                },
                "pre_processing_function": {"format": "python", "expression": "mod:preprocess"},
            }
        ],
        "mlm_output": [
            {
                "name": "mask",
                "tasks": ["semantic-segmentation"],
                "result": {
                    "shape": [-1, 2, 512, 512],
                    "dim_order": ["batch", "channel", "height", "width"],
                    "data_type": "float32",
                },
                "classification:classes": [{"name": "bg", "value": 0}, {"name": "building", "value": 1}],
                "post_processing_function": {"format": "python", "expression": "mod:postprocess"},
            }
        ],
        "mlm_hyperparameters": {"epochs": 15, "batch_size": 4, "learning_rate": 0.0001},
        "keywords": ["building"],
        "model_href": "weights.pt",
        "model_artifact_type": "torch.save",
        "mlm_pretrained": True,
        "mlm_pretrained_source": "OAM-TCD",
        "source_code_href": "https://github.com/example",
        "source_code_entrypoint": "mod:train",
        "training_runtime_href": "ghcr.io/hotosm/fair-unet:v1",
        "inference_runtime_href": "ghcr.io/hotosm/fair-unet:v1",
    }
    defaults.update(overrides)
    return build_base_model_item(**defaults)


def _dataset(tmp_path):
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[85.0, 27.0], [85.1, 27.0], [85.1, 27.1], [85.0, 27.1], [85.0, 27.0]]],
                },
                "properties": {},
            }
        ],
    }
    path = tmp_path / "labels.geojson"
    path.write_text(json.dumps(geojson))
    return build_dataset_item(
        item_id="ds",
        dt=datetime(2024, 6, 1, tzinfo=UTC),
        label_type="vector",
        label_tasks=["segmentation"],
        label_classes=[{"name": "building", "classes": ["building"]}],
        keywords=["building"],
        chips_href="data/oam/",
        labels_href=str(path),
    )


def test_training_config(tmp_path):
    cfg = generate_training_config(_base_model(), _dataset(tmp_path), model_name="finetuned")
    p = cfg["parameters"]
    assert cfg["model"]["name"] == "finetuned"
    assert p["epochs"] == 15 and p["batch_size"] == 4
    assert p["dataset_chips"] == "data/oam/"
    assert p["chip_size"] == 512 and p["num_classes"] == 2
    assert p["base_model_weights"] == "weights.pt"
    assert cfg["settings"]["docker"]["parent_image"] == "ghcr.io/hotosm/fair-unet:v1"


def test_training_overrides(tmp_path):
    cfg = generate_training_config(
        _base_model(),
        _dataset(tmp_path),
        model_name="t",
        overrides={"epochs": 1},
    )
    assert cfg["parameters"]["epochs"] == 1
    assert cfg["parameters"]["batch_size"] == 4  # non-overridden default preserved


def test_inference_config():
    cfg = generate_inference_config(_base_model(), "/data/input/")
    p = cfg["parameters"]
    assert p["model_uri"] == "weights.pt"
    assert p["input_images"] == "/data/input/"
    assert p["chip_size"] == 512 and p["num_classes"] == 2
    assert cfg["settings"]["docker"]["parent_image"] == "ghcr.io/hotosm/fair-unet:v1"
    assert "zenml_artifact_version_id" not in p


def test_inference_config_with_artifact_id():
    """When model asset has fair:zenml_artifact_version_id, it flows into config."""
    from fair.stac.builders import build_local_model_item

    base = _base_model()
    local = build_local_model_item(
        base_model_item=base,
        item_id="local-v1",
        dt=datetime(2024, 6, 1, tzinfo=UTC),
        model_href="s3://store/model/abc",
        mlm_hyperparameters={},
        keywords=["building"],
        base_model_item_id="unet",
        dataset_item_id="ds-1",
        version="1",
        zenml_artifact_version_id="uuid-123",
    )
    cfg = generate_inference_config(local, "/data/input/")
    p = cfg["parameters"]
    assert p["model_uri"] == "s3://store/model/abc"
    assert p["zenml_artifact_version_id"] == "uuid-123"
