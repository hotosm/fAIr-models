"""fAIr-models: Model registry and ML pipeline orchestration for fAIr."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("fair-models")
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"
