import json
from types import SimpleNamespace
from unittest.mock import MagicMock

from typer.testing import CliRunner

from fair.cli import _client, app

runner = CliRunner()

_SQUARE = [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]]


def _stub_client(monkeypatch) -> tuple[MagicMock, MagicMock]:
    client = MagicMock()
    backend = MagicMock()
    client._get_backend.return_value = backend
    monkeypatch.setattr("fair.cli._client", lambda: client)
    return client, backend


def test_client_reads_env(monkeypatch) -> None:
    monkeypatch.setenv("FAIR_USER_ID", "ops")
    monkeypatch.setenv("FAIR_STAC_API_URL", "https://stac.example/api")
    client = _client()
    assert client.user_id == "ops"
    assert client._stac_api_url == "https://stac.example/api"


def test_dataset_register_delegates(tmp_path, monkeypatch) -> None:
    client, _ = _stub_client(monkeypatch)
    client.register_dataset.return_value = "ds-1"
    draft = tmp_path / "draft.json"
    draft.write_text("{}")

    result = runner.invoke(app, ["dataset", "register", str(draft)])

    assert result.exit_code == 0
    assert "ds-1" in result.output
    client.register_dataset.assert_called_once_with(str(draft))


def test_basemodel_register_delegates(tmp_path, monkeypatch) -> None:
    client, _ = _stub_client(monkeypatch)
    client.register_base_model.return_value = "bm-1"
    item = tmp_path / "base.json"
    item.write_text("{}")

    result = runner.invoke(app, ["basemodel", "register", str(item)])

    assert result.exit_code == 0
    client.register_base_model.assert_called_once_with(str(item))


def test_model_train_passes_overrides(monkeypatch) -> None:
    client, _ = _stub_client(monkeypatch)
    client.finetune.return_value = "my-model"

    result = runner.invoke(
        app,
        ["model", "train", "--base", "b-1", "--dataset", "d-1", "--name", "my-model", "--epochs", "2"],
    )

    assert result.exit_code == 0
    assert "my-model" in result.output
    client.finetune.assert_called_once_with(
        base_model_id="b-1",
        dataset_id="d-1",
        model_name="my-model",
        overrides={"epochs": 2},
    )


def test_model_promote_without_out(monkeypatch) -> None:
    client, _ = _stub_client(monkeypatch)
    client.promote.return_value = "lm-1"

    result = runner.invoke(app, ["model", "promote", "my-model", "--description", "great", "--keywords", "a,b"])

    assert result.exit_code == 0
    assert "lm-1" in result.output
    client.promote.assert_called_once_with(
        "my-model",
        base_model_id=None,
        dataset_id=None,
        title=None,
        description="great",
        keywords=["a", "b"],
    )


def test_model_promote_writes_item(tmp_path, monkeypatch) -> None:
    client, backend = _stub_client(monkeypatch)
    client.promote.return_value = "lm-1"
    backend.get_item.return_value = SimpleNamespace(to_dict=lambda transform_hrefs: {"id": "lm-1"})

    result = runner.invoke(app, ["model", "promote", "my-model", "--out", str(tmp_path)])

    assert result.exit_code == 0
    written = tmp_path / "model-lm-1.json"
    assert json.loads(written.read_text()) == {"id": "lm-1"}


def test_item_get_to_stdout(monkeypatch) -> None:
    _, backend = _stub_client(monkeypatch)
    backend.get_item.return_value = SimpleNamespace(to_dict=lambda transform_hrefs: {"id": "x"})

    result = runner.invoke(app, ["item", "get", "datasets", "x"])

    assert result.exit_code == 0
    assert json.loads(result.output) == {"id": "x"}
    backend.get_item.assert_called_once_with("datasets", "x")


def test_item_get_rejects_unknown_collection(monkeypatch) -> None:
    _stub_client(monkeypatch)
    result = runner.invoke(app, ["item", "get", "nope", "x"])
    assert result.exit_code != 0


def test_item_list_formats_rows(monkeypatch) -> None:
    _, backend = _stub_client(monkeypatch)
    backend.list_items.return_value = [
        SimpleNamespace(id="a", properties={"version": "2", "title": "Alpha", "deprecated": False}),
        SimpleNamespace(id="b", properties={"version": "1", "title": "Beta", "deprecated": True}),
    ]

    result = runner.invoke(app, ["item", "list", "local-models", "--limit", "5"])

    assert result.exit_code == 0
    assert "a\tv2\tAlpha" in result.output
    assert "b\tv1\tBeta [deprecated]" in result.output
    backend.list_items.assert_called_once_with("local-models", limit=5)


def test_item_patch_set_builds_merge_patch(monkeypatch) -> None:
    _, backend = _stub_client(monkeypatch)
    backend.patch_item.return_value = SimpleNamespace(id="x")

    result = runner.invoke(
        app,
        ["item", "patch", "datasets", "x", "--set", "fair:pinned=true", "--set", "title=Hello"],
    )

    assert result.exit_code == 0
    backend.patch_item.assert_called_once_with("datasets", "x", {"properties": {"fair:pinned": True, "title": "Hello"}})


def test_item_patch_from_file(tmp_path, monkeypatch) -> None:
    _, backend = _stub_client(monkeypatch)
    backend.patch_item.return_value = SimpleNamespace(id="x")
    body = tmp_path / "patch.json"
    body.write_text(json.dumps({"properties": {"a": 1}}))

    result = runner.invoke(app, ["item", "patch", "datasets", "x", "--file", str(body)])

    assert result.exit_code == 0
    backend.patch_item.assert_called_once_with("datasets", "x", {"properties": {"a": 1}})


def test_item_patch_rejects_set_and_file(tmp_path, monkeypatch) -> None:
    _stub_client(monkeypatch)
    body = tmp_path / "patch.json"
    body.write_text("{}")
    result = runner.invoke(app, ["item", "patch", "datasets", "x", "--set", "a=1", "--file", str(body)])
    assert result.exit_code != 0


def test_item_patch_requires_input(monkeypatch) -> None:
    _stub_client(monkeypatch)
    result = runner.invoke(app, ["item", "patch", "datasets", "x"])
    assert result.exit_code != 0


def test_item_patch_rejects_bad_set(monkeypatch) -> None:
    _stub_client(monkeypatch)
    result = runner.invoke(app, ["item", "patch", "datasets", "x", "--set", "noequals"])
    assert result.exit_code != 0


def test_item_deprecate_delegates(monkeypatch) -> None:
    _, backend = _stub_client(monkeypatch)
    backend.deprecate_item.return_value = SimpleNamespace(id="x")

    result = runner.invoke(app, ["item", "deprecate", "base-models", "x"])

    assert result.exit_code == 0
    backend.deprecate_item.assert_called_once_with("base-models", "x")


def test_dataset_build_writes_draft(tmp_path, monkeypatch) -> None:
    aoi = tmp_path / "aoi.geojson"
    aoi.write_text(json.dumps({"type": "Feature", "geometry": {"type": "Polygon", "coordinates": _SQUARE}}))
    out = tmp_path / "drafts"

    geometry = {"type": "Polygon", "coordinates": _SQUARE}
    monkeypatch.setattr(
        "fair.cli.materialize_dataset",
        lambda **kwargs: ("/data/chips", "/data/labels/labels.geojson", geometry, [0.0, 0.0, 1.0, 1.0]),
    )

    result = runner.invoke(
        app,
        [
            "dataset",
            "build",
            "--tms",
            "https://tiles/{z}/{x}/{y}",
            "--aoi",
            str(aoi),
            "--zoom",
            "19",
            "--out",
            str(out),
            "--osm-url",
            "https://osm",
        ],
    )

    assert result.exit_code == 0, result.output
    drafts = list(out.glob("dataset-*.json"))
    assert len(drafts) == 1
    item = json.loads(drafts[0].read_text())
    assert item["assets"]["chips"]["href"] == "/data/chips"
    assert item["assets"]["labels"]["href"] == "/data/labels/labels.geojson"
    assert item["properties"]["fair:source_imagery"] == "https://tiles/{z}/{x}/{y}"
    assert any(link["rel"] == "source" for link in item["links"])


def test_dataset_build_requires_osm_url(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("FAIR_RAW_DATA_API_URL", raising=False)
    aoi = tmp_path / "aoi.geojson"
    aoi.write_text(json.dumps({"type": "Feature", "geometry": {"type": "Polygon", "coordinates": _SQUARE}}))

    result = runner.invoke(
        app,
        ["dataset", "build", "--tms", "https://t/{z}/{x}/{y}", "--aoi", str(aoi), "--zoom", "19"],
    )
    assert result.exit_code != 0
