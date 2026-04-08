"""Validate model contributions have required pipeline entrypoints.

Uses AST parsing (no imports, no runtime dependencies) to check that
every model's pipeline.py defines the required @pipeline and @step
decorated functions.
"""

from __future__ import annotations

import ast
from pathlib import Path

REQUIRED_PIPELINES = frozenset({"training_pipeline", "inference_pipeline"})
REQUIRED_STEPS = frozenset({"split_dataset"})
REQUIRED_FILES = frozenset({"pipeline.py", "README.md", "stac-item.json"})


def _has_decorator(func: ast.FunctionDef, name: str) -> bool:
    for dec in func.decorator_list:
        if isinstance(dec, ast.Name) and dec.id == name:
            return True
        if isinstance(dec, ast.Attribute) and dec.attr == name:
            return True
    return False


def _find_decorated_names(source: str, decorator: str) -> set[str]:
    tree = ast.parse(source)
    return {
        node.name
        for node in ast.iter_child_nodes(tree)
        if isinstance(node, ast.FunctionDef) and _has_decorator(node, decorator)
    }


def validate_model(model_dir: Path) -> list[str]:
    errors: list[str] = []

    for filename in sorted(REQUIRED_FILES):
        if not (model_dir / filename).exists():
            errors.append(f"{model_dir.name}: missing {filename}")

    pipeline_file = model_dir / "pipeline.py"
    if not pipeline_file.exists():
        return errors

    try:
        source = pipeline_file.read_text()
    except OSError as exc:
        errors.append(f"{model_dir.name}: cannot read pipeline.py: {exc}")
        return errors

    try:
        pipelines = _find_decorated_names(source, "pipeline")
        steps = _find_decorated_names(source, "step")
    except SyntaxError as exc:
        errors.append(f"{model_dir.name}: syntax error in pipeline.py: {exc}")
        return errors

    for name in sorted(REQUIRED_PIPELINES - pipelines):
        errors.append(f"{model_dir.name}: missing @pipeline function '{name}'")

    for name in sorted(REQUIRED_STEPS - steps):
        errors.append(f"{model_dir.name}: missing @step function '{name}'")

    return errors
