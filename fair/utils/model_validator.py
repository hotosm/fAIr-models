"""Validate model contributions have required pipeline entrypoints.

Uses AST parsing (no imports, no runtime dependencies) to check that
every model's pipeline.py defines the required @pipeline and @step
decorated functions. Also validates that models with non-URL weight
references provide a resolve_weights function.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

REQUIRED_PIPELINES = frozenset({"training_pipeline", "inference_pipeline"})
REQUIRED_STEPS = frozenset({"split_dataset"})
REQUIRED_FILES = frozenset({"pipeline.py", "README.md", "stac-item.json"})
REQUIRED_TEST_FILE = Path("tests") / "test_steps.py"
REQUIRED_TEST_FUNCTIONS = frozenset(
    {"test_split_dataset", "test_train_model", "test_evaluate_model", "test_export_onnx"}
)


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


def _find_top_level_functions(source: str) -> set[str]:
    tree = ast.parse(source)
    return {node.name for node in ast.iter_child_nodes(tree) if isinstance(node, ast.FunctionDef)}


def _model_href_is_url(model_dir: Path) -> bool | None:
    stac_path = model_dir / "stac-item.json"
    if not stac_path.exists():
        return None
    try:
        item = json.loads(stac_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    href = item.get("assets", {}).get("model", {}).get("href", "")
    if not href:
        return None
    return "://" in href


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

    is_url = _model_href_is_url(model_dir)
    if is_url is False:
        all_functions = _find_top_level_functions(source)
        if "resolve_weights" not in all_functions:
            errors.append(
                f"{model_dir.name}: model asset href is not a URL; "
                f"pipeline.py must define a resolve_weights(weight_id: str) -> Path function"
            )

    errors.extend(_validate_test_steps(model_dir))

    return errors


def _validate_test_steps(model_dir: Path) -> list[str]:
    errors: list[str] = []

    test_file = model_dir / REQUIRED_TEST_FILE
    if not test_file.exists():
        errors.append(f"{model_dir.name}: missing {REQUIRED_TEST_FILE}")
        return errors

    try:
        test_source = test_file.read_text()
    except OSError as exc:
        errors.append(f"{model_dir.name}: cannot read {REQUIRED_TEST_FILE}: {exc}")
        return errors

    try:
        test_functions = _find_top_level_functions(test_source)
    except SyntaxError as exc:
        errors.append(f"{model_dir.name}: syntax error in {REQUIRED_TEST_FILE}: {exc}")
        return errors

    for name in sorted(REQUIRED_TEST_FUNCTIONS - test_functions):
        errors.append(f"{model_dir.name}: missing test function '{name}' in {REQUIRED_TEST_FILE}")

    return errors
