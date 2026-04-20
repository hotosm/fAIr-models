from __future__ import annotations

import importlib
import sys
from types import ModuleType, SimpleNamespace
from typing import Any, ClassVar

import pytest


def _make_module(name: str, **attrs: Any) -> Any:
    module = ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def test_fair_version_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib.metadata as metadata

    import fair

    monkeypatch.setattr(metadata, "version", lambda _: "1.2.3")
    assert importlib.reload(fair).__version__ == "1.2.3"

    def _raise_not_found(_: str) -> str:
        raise metadata.PackageNotFoundError

    monkeypatch.setattr(metadata, "version", _raise_not_found)
    assert importlib.reload(fair).__version__ == "0.0.0.dev0"


def test_install_s3_cleanup_handler(monkeypatch: pytest.MonkeyPatch) -> None:
    from fair.utils import install_s3_cleanup_handler

    registered: list[Any] = []
    default_calls: list[dict[str, object]] = []
    handlers: list[Any] = []

    class FakeLoop:
        def set_exception_handler(self, handler: Any) -> None:
            handlers.append(handler)

        def default_exception_handler(self, context: dict[str, object]) -> None:
            default_calls.append(context)

    monkeypatch.setattr("fair.utils.atexit.register", lambda func: registered.append(func))
    monkeypatch.setattr("fair.utils.asyncio.get_event_loop", lambda: FakeLoop())

    class FakeFS:
        cleared = False

        @classmethod
        def clear_instance_cache(cls) -> None:
            cls.cleared = True

    fake_fsspec = SimpleNamespace(AbstractFileSystem=FakeFS)
    monkeypatch.setitem(sys.modules, "fsspec", fake_fsspec)

    install_s3_cleanup_handler()

    assert len(registered) == 1
    assert len(handlers) == 1

    handler = handlers[0]
    handler(FakeLoop(), {"exception": AssertionError("Session was never entered")})
    assert default_calls == []

    context = {"exception": RuntimeError("boom")}
    handler(FakeLoop(), context)
    assert default_calls == [context]

    registered[0]()
    assert FakeFS.cleared is True


def test_install_s3_cleanup_handler_tolerates_missing_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    from fair.utils import install_s3_cleanup_handler

    registered: list[Any] = []

    monkeypatch.setattr("fair.utils.atexit.register", lambda func: registered.append(func))
    monkeypatch.setattr(
        "fair.utils.asyncio.get_event_loop",
        lambda: (_ for _ in ()).throw(RuntimeError("no loop")),
    )

    install_s3_cleanup_handler()
    assert len(registered) == 1
    registered[0]()


def test_load_model_step_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    from fair.zenml.steps import load_model

    expected = object()
    artifact = SimpleNamespace(load=lambda: expected)
    client = SimpleNamespace(
        get_artifact_version=lambda _: artifact,
        list_artifact_versions=lambda **_: [artifact],
    )
    monkeypatch.setattr("fair.zenml.steps.Client", lambda: client)

    assert load_model.entrypoint("s3://model", "artifact-id") is expected
    assert load_model.entrypoint("s3://model") is expected

    empty_client = SimpleNamespace(
        get_artifact_version=lambda _: artifact,
        list_artifact_versions=lambda **_: [],
    )
    monkeypatch.setattr("fair.zenml.steps.Client", lambda: empty_client)
    with pytest.raises(RuntimeError):
        load_model.entrypoint("s3://missing")


def test_apply_zenml_patch_updates_server_database_type(monkeypatch: pytest.MonkeyPatch) -> None:
    from enum import Enum

    import fair._patch_zenml as patch_module

    class FakeServerDatabaseType(Enum):
        SQLITE = "sqlite"

    class FakeField:
        def __init__(self) -> None:
            self.annotation: Any = None

    rebuilt: list[bool] = []

    class FakeServerModel:
        model_fields: ClassVar[dict[str, FakeField]] = {"database_type": FakeField()}

        @classmethod
        def model_rebuild(cls, *, force: bool) -> None:
            rebuilt.append(force)

    fake_server_models = _make_module(
        "zenml.models.v2.misc.server_models",
        ServerDatabaseType=FakeServerDatabaseType,
        ServerModel=FakeServerModel,
    )
    fake_enum_utils = _make_module("zenml.utils.enum_utils", StrEnum=Enum)
    fake_misc = _make_module("zenml.models.v2.misc", server_models=fake_server_models)
    fake_v2 = _make_module("zenml.models.v2", misc=fake_misc)
    fake_models = _make_module("zenml.models", v2=fake_v2)
    fake_utils = _make_module("zenml.utils", enum_utils=fake_enum_utils)
    fake_zenml = _make_module("zenml", __version__="0.0-test", models=fake_models, utils=fake_utils)

    monkeypatch.delenv("FAIR_SKIP_ZENML_PATCH", raising=False)
    monkeypatch.setitem(sys.modules, "zenml", fake_zenml)
    monkeypatch.setitem(sys.modules, "zenml.utils", fake_utils)
    monkeypatch.setitem(sys.modules, "zenml.utils.enum_utils", fake_enum_utils)
    monkeypatch.setitem(sys.modules, "zenml.models", fake_models)
    monkeypatch.setitem(sys.modules, "zenml.models.v2", fake_v2)
    monkeypatch.setitem(sys.modules, "zenml.models.v2.misc", fake_misc)
    monkeypatch.setitem(sys.modules, "zenml.models.v2.misc.server_models", fake_server_models)

    patch_module._apply()

    values = {member.value for member in fake_server_models.ServerDatabaseType}
    assert "postgresql" in values
    assert rebuilt == [True]

    monkeypatch.setenv("FAIR_SKIP_ZENML_PATCH", "1")
    patch_module._apply()
