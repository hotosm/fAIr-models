from __future__ import annotations

from pathlib import Path

import pytest

from fair.zenml.materializers import (
    CHECKPOINT_FILENAME,
    ONNX_FILENAME,
    CheckpointBytesMaterializer,
    ONNXMaterializer,
)


def test_onnx_materializer_round_trip(tmp_path: Path) -> None:
    materializer = ONNXMaterializer(uri=str(tmp_path))
    materializer.save(b"onnx-bytes")
    assert (tmp_path / ONNX_FILENAME).read_bytes() == b"onnx-bytes"
    assert materializer.load(bytes) == b"onnx-bytes"


def test_onnx_materializer_rejects_non_bytes(tmp_path: Path) -> None:
    materializer = ONNXMaterializer(uri=str(tmp_path))
    with pytest.raises(TypeError, match="ONNXMaterializer expected bytes"):
        materializer.save("not-bytes")  # type: ignore[arg-type]


def test_checkpoint_materializer_round_trip(tmp_path: Path) -> None:
    materializer = CheckpointBytesMaterializer(uri=str(tmp_path))
    materializer.save(b"weights-bytes")
    assert (tmp_path / CHECKPOINT_FILENAME).read_bytes() == b"weights-bytes"
    assert materializer.load(bytes) == b"weights-bytes"


def test_checkpoint_materializer_rejects_non_bytes(tmp_path: Path) -> None:
    materializer = CheckpointBytesMaterializer(uri=str(tmp_path))
    with pytest.raises(TypeError, match="CheckpointBytesMaterializer expected bytes"):
        materializer.save("not-bytes")  # type: ignore[arg-type]
