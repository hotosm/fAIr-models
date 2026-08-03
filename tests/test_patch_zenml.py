import os
import subprocess
import sys

import pytest


class TestZenMLPostgresPatch:
    def test_postgresql_enum_member_exists(self):
        from zenml.models.v2.misc.server_models import ServerDatabaseType

        member = ServerDatabaseType("postgresql")
        assert member.value == "postgresql"

    def test_original_members_preserved(self):
        from zenml.models.v2.misc.server_models import ServerDatabaseType

        assert ServerDatabaseType("sqlite").name == "SQLITE"
        assert ServerDatabaseType("mysql").name == "MYSQL"
        assert ServerDatabaseType("other").name == "OTHER"

    def test_server_model_accepts_postgresql(self):
        from zenml.models.v2.misc.server_models import ServerModel

        data = {
            "id": "00000000-0000-0000-0000-000000000000",
            "version": "0.93.3",
            "deployment_type": "other",
            "database_type": "postgresql",
            "auth_scheme": "NO_AUTH",
            "server_url": "",
            "dashboard_url": "",
        }
        model = ServerModel.model_validate(data)
        assert model.database_type.value == "postgresql"

    def test_server_model_still_accepts_other(self):
        from zenml.models.v2.misc.server_models import ServerModel

        data = {
            "id": "00000000-0000-0000-0000-000000000000",
            "version": "0.93.3",
            "deployment_type": "other",
            "database_type": "other",
            "auth_scheme": "NO_AUTH",
            "server_url": "",
            "dashboard_url": "",
        }
        model = ServerModel.model_validate(data)
        assert model.database_type.value == "other"

    def test_server_model_rejects_invalid(self):
        from zenml.models.v2.misc.server_models import ServerModel

        data = {
            "id": "00000000-0000-0000-0000-000000000000",
            "version": "0.93.3",
            "deployment_type": "other",
            "database_type": "oracle",
            "auth_scheme": "NO_AUTH",
            "server_url": "",
            "dashboard_url": "",
        }
        with pytest.raises(ValueError):
            ServerModel.model_validate(data)

    def test_enum_inherits_from_zenml_strenum(self):
        from zenml.models.v2.misc.server_models import ServerDatabaseType
        from zenml.utils.enum_utils import StrEnum as ZenMLStrEnum

        assert issubclass(ServerDatabaseType, ZenMLStrEnum)

    def test_patch_idempotent(self):
        import fair._patch_zenml

        fair._patch_zenml._apply()
        fair._patch_zenml._apply()

        from zenml.models.v2.misc.server_models import ServerDatabaseType

        assert ServerDatabaseType("postgresql").value == "postgresql"
        assert len([m for m in ServerDatabaseType if m.value == "postgresql"]) == 1

    def test_env_var_skips_patch(self):
        # Needs a fresh interpreter: this process is already patched, so an in-process
        # _apply() returns early whether or not the env var is set.
        code = (
            "import fair._patch_zenml as p; p._apply();"
            "from zenml.models.v2.misc.server_models import ServerDatabaseType;"
            "print(any(m.value == 'postgresql' for m in ServerDatabaseType))"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env={**os.environ, "FAIR_SKIP_ZENML_PATCH": "1"},
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "False"
