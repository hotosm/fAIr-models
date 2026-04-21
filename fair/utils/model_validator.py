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


def _asset_href_is_https_url(model_dir: Path, asset_name: str) -> bool | None:
    stac_path = model_dir / "stac-item.json"
    if not stac_path.exists():
        return None
    try:
        item = json.loads(stac_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    href = item.get("assets", {}).get(asset_name, {}).get("href", "")
    if not href:
        return None
    return href.startswith(("http://", "https://"))


def _return_annotation_name(func: ast.FunctionDef) -> str | None:
    ann = func.returns
    if ann is None:
        return None
    if isinstance(ann, ast.Name):
        return ann.id
    if isinstance(ann, ast.Constant):
        return str(ann.value)
    if isinstance(ann, ast.Subscript):
        if isinstance(ann.value, ast.Name):
            return ann.value.id
        if isinstance(ann.slice, ast.Name):
            return ann.slice.id
        if isinstance(ann.slice, ast.Tuple) and ann.slice.elts:
            first = ann.slice.elts[0]
            if isinstance(first, ast.Name):
                return first.id
    return None


def _validate_step_return_types(source: str, model_name: str) -> list[str]:
    errors: list[str] = []
    tree = ast.parse(source)
    for node in ast.iter_child_nodes(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if not _has_decorator(node, "step"):
            continue

        ret = _return_annotation_name(node)
        if node.name == "export_onnx" and ret == "str":
            errors.append(
                f"{model_name}: export_onnx must return bytes (not str); "
                f"ZenML persists the path string, not the ONNX file"
            )
        if node.name == "train_model" and ret == "str":
            errors.append(f"{model_name}: train_model must not return str; return the model object or serialized bytes")
    return errors


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

    checkpoint_is_url = _asset_href_is_https_url(model_dir, "checkpoint")
    if checkpoint_is_url is False:
        errors.append(f"{model_dir.name}: checkpoint asset href must be an https URL")

    model_is_url = _asset_href_is_https_url(model_dir, "model")
    if model_is_url is None:
        errors.append(f"{model_dir.name}: model asset href is required and must be an https URL")
    elif model_is_url is False:
        errors.append(f"{model_dir.name}: model asset href must be an https URL")

    errors.extend(_validate_step_return_types(source, model_dir.name))
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
