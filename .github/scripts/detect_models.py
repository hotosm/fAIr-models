"""Detect models with Dockerfiles for CI/CD."""

import json
from pathlib import Path

import pystac


def find_models():
    models_dir = Path("models")
    if not models_dir.exists():
        return []

    models = []
    for model_path in sorted(models_dir.iterdir()):
        if not model_path.is_dir() or not (model_path / "Dockerfile").exists():
            continue

        stac_file = model_path / "stac-item.json"
        if not stac_file.exists():
            raise FileNotFoundError(f"Missing stac-item.json for {model_path.name}")

        item = pystac.Item.from_file(str(stac_file))
        version = item.properties["version"]

        models.append({"name": model_path.name, "version": version, "path": str(model_path)})

    return models


if __name__ == "__main__":
    print(json.dumps({"include": find_models()}))
