"""Custom ZenML materializers used by fAIr model pipelines.

These exist to give downstream consumers (e.g. the promote step that copies
artifacts to a stable S3 prefix for the STAC catalog) a deterministic, public
filename to read. Without them, ZenML's BuiltInMaterializer for bytes writes
the artifact as `data.txt`, which is a private constant
(`zenml.materializers.built_in_materializer.DEFAULT_BYTES_FILENAME`) and would
silently break promotion if ZenML renamed it.
"""

import os
from typing import Any

from zenml.enums import ArtifactType
from zenml.io import fileio
from zenml.materializers.base_materializer import BaseMaterializer

ONNX_FILENAME = "model.onnx"
CHECKPOINT_FILENAME = "checkpoint.pt"


class ONNXMaterializer(BaseMaterializer):
    """Persists ONNX bytes as `model.onnx` under the artifact URI.

    Apply via `@step(output_materializers={"onnx_model": ONNXMaterializer})` so
    only steps that explicitly opt in use this materializer; ZenML's built-in
    bytes materializer continues to serve every other `bytes` output.
    """

    ASSOCIATED_TYPES = (bytes,)
    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.MODEL

    def save(self, data: bytes) -> None:
        if not isinstance(data, bytes):
            raise TypeError(f"ONNXMaterializer expected bytes, got {type(data).__name__}")
        with fileio.open(os.path.join(self.uri, ONNX_FILENAME), "wb") as f:
            f.write(data)

    def load(self, data_type: type[Any]) -> bytes:
        with fileio.open(os.path.join(self.uri, ONNX_FILENAME), "rb") as f:
            return f.read()


class CheckpointBytesMaterializer(BaseMaterializer):
    """Persists raw checkpoint bytes as `checkpoint.pt` under the artifact URI.

    Used by step outputs that already produce a serialized PyTorch checkpoint
    (for example Ultralytics YOLO's `model.save()` blob). Aligns the on-disk
    filename with what fair.zenml.promotion expects when copying the artifact
    out of the ZenML store.
    """

    ASSOCIATED_TYPES = (bytes,)
    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.MODEL

    def save(self, data: bytes) -> None:
        if not isinstance(data, bytes):
            raise TypeError(f"CheckpointBytesMaterializer expected bytes, got {type(data).__name__}")
        with fileio.open(os.path.join(self.uri, CHECKPOINT_FILENAME), "wb") as f:
            f.write(data)

    def load(self, data_type: type[Any]) -> bytes:
        with fileio.open(os.path.join(self.uri, CHECKPOINT_FILENAME), "rb") as f:
            return f.read()
