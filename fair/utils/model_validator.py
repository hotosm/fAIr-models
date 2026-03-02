"""Validate model contributions have required pipeline entrypoints.

Uses AST parsing (no imports, no runtime dependencies) to check that
every model's pipeline.py defines @pipeline-decorated functions named
training_pipeline and inference_pipeline.
"""

from __future__ import annotations

import ast
from pathlib import Path

REQUIRED_PIPELINES = frozenset({"training_pipeline", "inference_pipeline"})
REQUIRED_FILES = frozenset({"pipeline.py", "README.md", "stac-item.json"})


def _has_pipeline_decorator(func: ast.FunctionDef) -> bool:
    """Check if function has @pipeline decorator (bare or dotted)."""
    for dec in func.decorator_list:
        if isinstance(dec, ast.Name) and dec.id == "pipeline":
            return True
        if isinstance(dec, ast.Attribute) and dec.attr == "pipeline":
            return True
    return False


def _find_pipeline_names(source: str) -> set[str]:
    """Extract top-level @pipeline-decorated function names via AST."""
    tree = ast.parse(source)
    return {
        node.name
        for node in ast.iter_child_nodes(tree)
        if isinstance(node, ast.FunctionDef) and _has_pipeline_decorator(node)
    }


def validate_model(model_dir: Path) -> list[str]:
    """Validate a model directory has required pipeline entrypoints.

    Static analysis only — no imports needed, works in lightweight CI.

    Args:
        model_dir: Path to model directory (e.g. models/example_unet).

    Returns:
        List of error strings. Empty means valid.
    """
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
        found = _find_pipeline_names(source)
    except SyntaxError as exc:
        errors.append(f"{model_dir.name}: syntax error in pipeline.py: {exc}")
        return errors

    for name in sorted(REQUIRED_PIPELINES - found):
        errors.append(f"{model_dir.name}: missing @pipeline function '{name}'")

    return errors
