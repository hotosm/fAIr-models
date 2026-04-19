from __future__ import annotations

from pathlib import Path


def test_every_model_has_required_test_suite() -> None:
    model_root = Path("models")
    model_dirs = sorted(
        path
        for path in model_root.iterdir()
        if path.is_dir() and (path / "pipeline.py").exists() and (path / "stac-item.json").exists()
    )

    assert model_dirs

    for model_dir in model_dirs:
        tests_dir = model_dir / "tests"
        assert tests_dir.is_dir(), f"Missing tests directory for {model_dir.name}"
        assert (tests_dir / "test_serve.py").exists(), f"Missing serve tests for {model_dir.name}"
        assert (tests_dir / "test_steps.py").exists(), f"Missing step tests for {model_dir.name}"
