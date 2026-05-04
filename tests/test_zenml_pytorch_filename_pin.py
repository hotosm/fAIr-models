"""Pin the ZenML torch-materializer filename our promote step depends on.

`fair.zenml.promotion._CHECKPOINT_SOURCE_FILENAME` is hardcoded as `checkpoint.pt`
because the promote container deliberately runs without torch installed (artifact
copy goes via UPath bytes, not torch.load). That makes it impossible to import
the constant from `zenml.integrations.pytorch.materializers.pytorch_module_materializer`
at runtime: the import would fail on missing torch.

This test runs in the dev/CI environment where torch IS available. It asserts
the hardcoded value still matches ZenML's. A ZenML upgrade that renames the
constant fails this test instead of silently breaking promote() in production.
"""

import pytest


def test_checkpoint_filename_matches_zenml() -> None:
    pytorch = pytest.importorskip("torch")
    del pytorch  # only here to ensure the materializer module imports cleanly
    from zenml.integrations.pytorch.materializers.pytorch_module_materializer import (
        CHECKPOINT_FILENAME,
    )

    from fair.zenml.promotion import _CHECKPOINT_SOURCE_FILENAME

    assert _CHECKPOINT_SOURCE_FILENAME == CHECKPOINT_FILENAME, (
        f"ZenML renamed CHECKPOINT_FILENAME to {CHECKPOINT_FILENAME!r}. "
        f"Update fair.zenml.promotion._CHECKPOINT_SOURCE_FILENAME to match."
    )
