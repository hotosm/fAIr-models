"""Protocol conformance: the concrete backends satisfy StacBackend."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import pytest

from fair.stac.api_backend import StacApiBackend
from fair.stac.backend import StacBackend
from fair.stac.catalog_manager import StacCatalogManager


@runtime_checkable
class _RuntimeStacBackend(StacBackend, Protocol):
    """Runtime-checkable view of StacBackend; inherits its members so it stays in sync."""


# PgStacBackend needs the optional [pgstac] extra; its conformance is checked in
# test_pgstac_backend.py, where pypgstac is mocked at import time.
@pytest.mark.parametrize("backend_cls", [StacCatalogManager, StacApiBackend])
def test_backend_satisfies_stac_backend_protocol(backend_cls: type) -> None:
    assert issubclass(backend_cls, _RuntimeStacBackend)
