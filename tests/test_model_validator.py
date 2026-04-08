from __future__ import annotations

from pathlib import Path

from fair.utils.model_validator import REQUIRED_FILES, _find_decorated_names, validate_model


class TestFindDecoratedNames:
    def test_finds_pipeline_functions(self) -> None:
        source = """
from zenml import pipeline

@pipeline
def training_pipeline(): ...

@pipeline
def inference_pipeline(): ...

def helper(): ...
"""
        assert _find_decorated_names(source, "pipeline") == {"training_pipeline", "inference_pipeline"}

    def test_dotted_decorator(self) -> None:
        source = """
import zenml

@zenml.pipeline
def training_pipeline(): ...
"""
        assert _find_decorated_names(source, "pipeline") == {"training_pipeline"}

    def test_finds_step_functions(self) -> None:
        source = """
from zenml import step

@step
def split_dataset(): ...

@step
def train_model(): ...
"""
        assert _find_decorated_names(source, "step") == {"split_dataset", "train_model"}

    def test_ignores_other_decorators(self) -> None:
        source = """
@step
def train_model(): ...
"""
        assert _find_decorated_names(source, "pipeline") == set()


def _scaffold(tmp_path: Path) -> None:
    """Create all required files except pipeline.py."""
    (tmp_path / "README.md").write_text("# Model\n")
    (tmp_path / "stac-item.json").write_text("{}\n")


class TestValidateModel:
    def test_valid_model(self, tmp_path: Path) -> None:
        _scaffold(tmp_path)
        pipeline_file = tmp_path / "pipeline.py"
        pipeline_file.write_text("""
from zenml import pipeline, step

@step
def split_dataset(): ...

@pipeline
def training_pipeline(): ...

@pipeline
def inference_pipeline(): ...
""")
        assert validate_model(tmp_path) == []

    def test_missing_pipeline_py(self, tmp_path: Path) -> None:
        _scaffold(tmp_path)
        errors = validate_model(tmp_path)
        assert len(errors) == 1
        assert "missing pipeline.py" in errors[0]

    def test_missing_inference(self, tmp_path: Path) -> None:
        _scaffold(tmp_path)
        (tmp_path / "pipeline.py").write_text("""
from zenml import pipeline, step

@step
def split_dataset(): ...

@pipeline
def training_pipeline(): ...
""")
        errors = validate_model(tmp_path)
        assert len(errors) == 1
        assert "inference_pipeline" in errors[0]

    def test_missing_split_dataset(self, tmp_path: Path) -> None:
        _scaffold(tmp_path)
        (tmp_path / "pipeline.py").write_text("""
from zenml import pipeline

@pipeline
def training_pipeline(): ...

@pipeline
def inference_pipeline(): ...
""")
        errors = validate_model(tmp_path)
        assert len(errors) == 1
        assert "split_dataset" in errors[0]

    def test_missing_all_entrypoints(self, tmp_path: Path) -> None:
        _scaffold(tmp_path)
        (tmp_path / "pipeline.py").write_text("x = 1\n")
        errors = validate_model(tmp_path)
        assert len(errors) == 3

    def test_syntax_error(self, tmp_path: Path) -> None:
        _scaffold(tmp_path)
        (tmp_path / "pipeline.py").write_text("def (broken")
        errors = validate_model(tmp_path)
        assert len(errors) == 1
        assert "syntax error" in errors[0]

    def test_missing_all_files(self, tmp_path: Path) -> None:
        errors = validate_model(tmp_path)
        assert len(errors) == len(REQUIRED_FILES)
        for filename in REQUIRED_FILES:
            assert any(filename in e for e in errors)

    def test_missing_readme_only(self, tmp_path: Path) -> None:
        (tmp_path / "stac-item.json").write_text("{}\n")
        (tmp_path / "pipeline.py").write_text("""
from zenml import pipeline, step

@step
def split_dataset(): ...

@pipeline
def training_pipeline(): ...

@pipeline
def inference_pipeline(): ...
""")
        errors = validate_model(tmp_path)
        assert len(errors) == 1
        assert "README.md" in errors[0]

    def test_validates_real_example(self) -> None:
        """Validate the actual unet_segmentation model in the repo."""
        model_dir = Path(__file__).resolve().parent.parent / "models" / "unet_segmentation"
        if model_dir.exists():
            assert validate_model(model_dir) == []
