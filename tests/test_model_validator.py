from __future__ import annotations

from pathlib import Path

from fair.utils.model_validator import _find_pipeline_names, validate_model


class TestFindPipelineNames:
    def test_finds_decorated_functions(self) -> None:
        source = """
from zenml import pipeline

@pipeline
def training_pipeline(): ...

@pipeline
def inference_pipeline(): ...

def helper(): ...
"""
        assert _find_pipeline_names(source) == {"training_pipeline", "inference_pipeline"}

    def test_dotted_decorator(self) -> None:
        source = """
import zenml

@zenml.pipeline
def training_pipeline(): ...
"""
        assert _find_pipeline_names(source) == {"training_pipeline"}

    def test_ignores_non_pipeline(self) -> None:
        source = """
@step
def train_model(): ...
"""
        assert _find_pipeline_names(source) == set()


class TestValidateModel:
    def test_valid_model(self, tmp_path: Path) -> None:
        pipeline_file = tmp_path / "pipeline.py"
        pipeline_file.write_text("""
from zenml import pipeline

@pipeline
def training_pipeline(): ...

@pipeline
def inference_pipeline(): ...
""")
        assert validate_model(tmp_path) == []

    def test_missing_pipeline_py(self, tmp_path: Path) -> None:
        errors = validate_model(tmp_path)
        assert len(errors) == 1
        assert "missing pipeline.py" in errors[0]

    def test_missing_inference(self, tmp_path: Path) -> None:
        (tmp_path / "pipeline.py").write_text("""
from zenml import pipeline

@pipeline
def training_pipeline(): ...
""")
        errors = validate_model(tmp_path)
        assert len(errors) == 1
        assert "inference_pipeline" in errors[0]

    def test_missing_both(self, tmp_path: Path) -> None:
        (tmp_path / "pipeline.py").write_text("x = 1\n")
        errors = validate_model(tmp_path)
        assert len(errors) == 2

    def test_syntax_error(self, tmp_path: Path) -> None:
        (tmp_path / "pipeline.py").write_text("def (broken")
        errors = validate_model(tmp_path)
        assert len(errors) == 1
        assert "syntax error" in errors[0]

    def test_validates_real_example(self) -> None:
        """Validate the actual example_unet model in the repo."""
        model_dir = Path(__file__).resolve().parent.parent / "models" / "example_unet"
        if model_dir.exists():
            assert validate_model(model_dir) == []
