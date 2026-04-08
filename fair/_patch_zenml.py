"""Patch ZenML ServerDatabaseType to accept 'postgresql'.

Triggered at interpreter startup via .pth before any ZenML import.
https://docs.python.org/3/library/site.html
Set FAIR_SKIP_ZENML_PATCH=1 to disable.
"""

import os


def _apply() -> None:
    if os.environ.get("FAIR_SKIP_ZENML_PATCH"):
        return

    try:
        from zenml.models.v2.misc.server_models import ServerDatabaseType
    except (ImportError, ModuleNotFoundError):
        return

    if any(m.value == "postgresql" for m in ServerDatabaseType):
        return

    # stdlib enum.StrEnum != zenml.utils.enum_utils.StrEnum; must use theirs
    from zenml.utils.enum_utils import StrEnum as ZenMLStrEnum

    new_members = {m.name: m.value for m in ServerDatabaseType}
    new_members["POSTGRESQL"] = "postgresql"

    patched = ZenMLStrEnum("ServerDatabaseType", new_members)

    patched.__module__ = ServerDatabaseType.__module__
    patched.__qualname__ = ServerDatabaseType.__qualname__  # type: ignore[attr-defined]

    if not any(m.value == "postgresql" for m in patched):  # ty: ignore[unresolved-attribute]
        import zenml

        msg = f"ZenML patch failed: POSTGRESQL member not added (zenml {zenml.__version__})"
        raise RuntimeError(msg)

    import zenml.models.v2.misc.server_models as mod

    mod.ServerDatabaseType = patched  # type: ignore[misc]

    if hasattr(mod, "ServerModel"):
        for field_name, field_info in mod.ServerModel.model_fields.items():
            if field_name == "database_type":
                field_info.annotation = patched
        mod.ServerModel.model_rebuild(force=True)


_apply()
