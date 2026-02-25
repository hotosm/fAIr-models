"""fair-py-ops: Model registry and ML pipeline orchestration for fAIr."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("fair-py-ops")
except PackageNotFoundError:
    __version__ = "0.1.0.dev0"
