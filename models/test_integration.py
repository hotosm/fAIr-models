"""Shared FairClient integration test for all model pipelines.

Runs inside Docker CI after build. Uses local ZenML server, STAC catalog,
and auto-generated toy data derived from the model's stac-item.json.
Validates the full workflow: setup -> register -> finetune -> promote -> predict.

Pass --model-dir=models/<name> to select which model to test.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture(scope="module")
def work_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("integration")


@pytest.fixture(scope="module")
def model_dir(request: pytest.FixtureRequest) -> Path:
    raw = request.config.getoption("--model-dir", default=None)
    if raw:
        return Path(raw).resolve()
    msg = "Pass --model-dir=models/<name> for integration tests"
    raise pytest.UsageError(msg)


@pytest.fixture(scope="module")
def model_stac(model_dir: Path) -> dict[str, Any]:
    return json.loads((model_dir / "stac-item.json").read_text())


@pytest.fixture(scope="module")
def task_type(model_stac: dict[str, Any]) -> str:
    tasks = model_stac["properties"].get("mlm:tasks", [])
    return tasks[0] if tasks else "unknown"


@pytest.fixture(scope="module")
def _toy_data(work_dir: Path, task_type: str) -> tuple[Path, Path]:
    import numpy as np
    from PIL import Image

    chip_size = 256
    chips_dir = work_dir / "chips"
    chips_dir.mkdir()

    for i in range(6):
        img = np.random.randint(0, 255, (chip_size, chip_size, 3), dtype=np.uint8)
        Image.fromarray(img).save(chips_dir / f"chip_{i:03d}.png")

    if task_type == "classification":
        labels = work_dir / "labels.csv"
        with open(labels, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["filename", "class_name"])
            writer.writeheader()
            for i, chip in enumerate(sorted(chips_dir.glob("*.png"))):
                writer.writerow(
                    {
                        "filename": chip.name,
                        "class_name": "building" if i % 2 == 0 else "no_building",
                    }
                )
        return chips_dir, labels

    if task_type == "object-detection":
        labels = work_dir / "labels.json"
        chip_files = sorted(chips_dir.glob("*.png"))
        images = []
        annotations = []
        for img_id, chip in enumerate(chip_files, start=1):
            images.append(
                {
                    "id": img_id,
                    "file_name": chip.name,
                    "width": chip_size,
                    "height": chip_size,
                }
            )
            box = chip_size // 3
            annotations.append(
                {
                    "id": img_id,
                    "image_id": img_id,
                    "category_id": 0,
                    "bbox": [2, 2, box, box],
                    "area": box * box,
                    "iscrowd": 0,
                }
            )
        coco = {
            "images": images,
            "annotations": annotations,
            "categories": [{"id": 0, "name": "building"}],
        }
        labels.write_text(json.dumps(coco))
        return chips_dir, labels

    labels_dir = work_dir / "labels"
    labels_dir.mkdir()
    features = [
        {
            "type": "Feature",
            "properties": {"building": "yes"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]],
            },
        }
        for _ in range(6)
    ]
    (labels_dir / "buildings.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": features}))
    return chips_dir, labels_dir


def _label_info(task_type: str) -> tuple[str, list[str], list[dict[str, Any]]]:
    if task_type == "classification":
        return "raster", ["classification"], [{"name": None, "classes": ["building"]}]
    if task_type == "object-detection":
        return "vector", ["object-detection"], [{"name": "building", "classes": ["building"]}]
    return "vector", ["segmentation"], [{"name": "building", "classes": ["yes"]}]


def _labels_media_type(task_type: str) -> str:
    if task_type == "classification":
        return "text/csv"
    if task_type == "object-detection":
        return "application/json"
    return "application/geo+json"


@pytest.fixture(scope="module")
def dataset_stac_item(work_dir: Path, task_type: str, _toy_data: tuple[Path, Path]) -> str:
    chips_dir, labels_path = _toy_data
    label_type, label_tasks, label_classes = _label_info(task_type)
    item = {
        "type": "Feature",
        "stac_version": "1.1.0",
        "stac_extensions": [
            "https://stac-extensions.github.io/label/v1.0.1/schema.json",
        ],
        "id": "integration-test-dataset",
        "geometry": None,
        "bbox": None,
        "properties": {
            "datetime": "2024-01-01T00:00:00Z",
            "label:type": label_type,
            "label:tasks": label_tasks,
            "label:classes": label_classes,
            "fair:user_id": "test",
            "version": "1",
            "deprecated": False,
            "license": "CC-BY-4.0",
        },
        "assets": {
            "chips": {"href": str(chips_dir), "type": "image/png", "roles": ["data"]},
            "labels": {"href": str(labels_path), "type": _labels_media_type(task_type), "roles": ["labels"]},
        },
        "links": [],
    }
    stac_path = work_dir / "dataset-stac-item.json"
    stac_path.write_text(json.dumps(item, indent=2))
    return str(stac_path)


@pytest.fixture(scope="module")
def predict_images(_toy_data: tuple[Path, Path]) -> str:
    return str(_toy_data[0])


@pytest.fixture(scope="module")
def zenml_server(work_dir: Path):
    zenml_bin = Path(sys.executable).parent / "zenml"
    env = {**os.environ, "ZENML_CONFIG_PATH": str(work_dir / ".zen")}
    subprocess.run([str(zenml_bin), "init"], check=True, capture_output=True, cwd=str(work_dir), env=env)
    proc = subprocess.Popen(
        [str(zenml_bin), "login", "--local"],
        cwd=str(work_dir),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    import time

    time.sleep(5)
    yield env
    proc.terminate()
    proc.wait(timeout=10)


@pytest.fixture(scope="module")
def fair_client(work_dir: Path, zenml_server: dict[str, str]) -> Any:
    from fair.client import FairClient

    catalog_path = str(work_dir / "stac_catalog" / "catalog.json")
    config_dir = str(work_dir / "config")
    return FairClient(
        catalog_path=catalog_path,
        user_id="test",
        config_dir=config_dir,
    )


@pytest.mark.slow
def test_full_pipeline(
    fair_client: Any,
    model_dir: Path,
    dataset_stac_item: str,
    predict_images: str,
    zenml_server: dict[str, str],
) -> None:
    os.environ.update(zenml_server)

    fair_client.setup()

    base_model_id = fair_client.register_base_model(str(model_dir / "stac-item.json"))
    assert base_model_id

    dataset_id = fair_client.register_dataset(dataset_stac_item)
    assert dataset_id

    model_name = f"test-{model_dir.name}"
    finetuned_id = fair_client.finetune(
        base_model_id=base_model_id,
        dataset_id=dataset_id,
        model_name=model_name,
        overrides={"epochs": 1, "batch_size": 2},
    )
    assert finetuned_id

    local_model_id = fair_client.promote(finetuned_id, description="integration test")
    assert local_model_id

    fair_client.predict(local_model_id, image_path=predict_images)
