"""Detect models with Dockerfiles for CI/CD."""

import json
import os
import subprocess
from pathlib import Path
from typing import TypedDict

import pystac


class ModelInfo(TypedDict):
    name: str
    version: str
    path: str


def get_changed_models() -> set[str] | None:
    if os.getenv("EVENT_NAME") != "pull_request":
        return None
    base_sha = os.getenv("BASE_SHA")
    head_sha = os.getenv("HEAD_SHA")
    if not base_sha or not head_sha:
        return None
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", base_sha, head_sha],
            capture_output=True,
            text=True,
            check=True,
        )
        return {
            Path(f).parts[0] + "/" + Path(f).parts[1]
            for f in result.stdout.splitlines()
            if f.startswith("models/") and len(Path(f).parts) > 1
        }
    except subprocess.CalledProcessError:
        return None


def find_models() -> list[ModelInfo]:
    models_dir = Path("models")
    if not models_dir.exists():
        return []

    changed = get_changed_models()
    models: list[ModelInfo] = []

    for model_path in sorted(models_dir.iterdir()):
        if not model_path.is_dir() or not (model_path / "Dockerfile").exists():
            continue
        if changed and str(model_path) not in changed:
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
