import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import click
import pytest
from typer.testing import CliRunner

from fair.cli import _client, app
from fair.datasets import MaterializedDataset

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


def test_basemodel_pin_rewrites_image_tags(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("fair.infra.registry.resolve_digest", lambda ref: ref.rsplit(":", 1)[0] + "@sha256:abc")
    item = tmp_path / "base.json"
    item.write_text(
        json.dumps(
            {
                "assets": {
                    "mlm:training": {"href": "ghcr.io/hotosm/demo:v1"},
                    "mlm:inference": {"href": "ghcr.io/hotosm/demo:v1-inference"},
                    "model": {"href": "https://example.com/model.onnx"},
                }
            }
        )
    )

    result = runner.invoke(app, ["basemodel", "pin", str(item)])

    assert result.exit_code == 0
    assets = json.loads(item.read_text())["assets"]
    assert assets["mlm:training"]["href"] == "ghcr.io/hotosm/demo@sha256:abc"
    assert assets["mlm:inference"]["href"] == "ghcr.io/hotosm/demo@sha256:abc"
    assert assets["model"]["href"] == "https://example.com/model.onnx"


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
        class_map=None,
    )


def test_model_train_passes_the_class_map(monkeypatch) -> None:
    client, _ = _stub_client(monkeypatch)
    command = ["model", "train", "--base", "b", "--dataset", "d", "--name", "m", "--class-map"]

    result = runner.invoke(app, [*command, '{"natural=palm": "tree"}'])

    assert result.exit_code == 0, result.output
    assert client.finetune.call_args.kwargs["class_map"] == {"natural=palm": "tree"}
    assert runner.invoke(app, [*command, "natural=palm"]).exit_code != 0
    assert runner.invoke(app, [*command, '{"natural=palm": 1}']).exit_code != 0


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
        lambda *, out_dir, **kwargs: MaterializedDataset(
            chips_dirs={"image": f"{out_dir}/chips-image"},
            labels_file=f"{out_dir}/labels/labels.geojson",
            geometry=geometry,
            bbox=[0.0, 0.0, 1.0, 1.0],
            label_properties=["building"],
            label_classes=[{"name": "building", "classes": ["yes"]}],
        ),
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
    assert item["assets"]["chips-image"]["href"] == "aoi-polygon/chips-image"
    assert item["assets"]["labels"]["href"] == "aoi-polygon/labels/labels.geojson"
    assert item["properties"]["fair:source_imagery"] == "https://tiles/{z}/{x}/{y}"
    assert item["properties"]["license"] == "ODbL-1.0"
    assert item["properties"]["label:properties"] == ["building"]
    assert item["properties"]["label:classes"] == [{"name": "building", "classes": ["yes"]}]
    assert any(link["rel"] == "source" for link in item["links"])


def _model_with_preview(tmp_path, inputs: list[str]):
    preview = {
        "center": [85.52, 27.63],
        "zoom": {"recommended": 18},
        "imagery": {"url": "https://post/{z}/{x}/{y}"},
        "pre_imagery": {"url": "https://pre/{z}/{x}/{y}"},
    }
    model = tmp_path / "stac-item.json"
    model.write_text(
        json.dumps(
            {
                "id": "demo-model",
                "properties": {
                    "mlm:input": [{"name": name} for name in inputs],
                    "mlm:output": [
                        {
                            "classification:classes": [
                                {"name": "background", "value": 0},
                                {"name": "building", "value": 1, "fair:osm_tags": {"building": ["*"]}},
                            ]
                        }
                    ],
                    "mlm:tasks": ["semantic-segmentation"],
                    "keywords": ["building", "polygon"],
                    "fair:preview": preview,
                },
            }
        )
    )
    return model


def _capture_materialize(monkeypatch) -> dict:
    captured: dict = {}

    def fake(*, out_dir, tms_urls, **kwargs):
        captured.update(out_dir=out_dir, tms_urls=tms_urls, **kwargs)
        return MaterializedDataset(
            chips_dirs={name: f"{out_dir}/chips-{name}" for name in tms_urls},
            labels_file=f"{out_dir}/labels/labels.geojson",
            geometry={"type": "Polygon", "coordinates": _SQUARE},
            bbox=[0, 0, 1, 1],
            label_properties=["building"],
            label_classes=[{"name": "building", "classes": ["yes"]}],
        )

    monkeypatch.setattr("fair.cli.materialize_dataset", fake)
    return captured


def test_dataset_build_from_preview_writes_a_sample_item(tmp_path, monkeypatch) -> None:
    captured = _capture_materialize(monkeypatch)
    model = _model_with_preview(tmp_path, ["post", "pre"])
    out = tmp_path / "sample"

    result = runner.invoke(
        app,
        ["dataset", "build", "--from-preview", str(model), "--out", str(out)],
        env={"FAIR_RAW_DATA_API_URL": "https://osm"},
    )

    assert result.exit_code == 0, result.output
    assert captured["tms_urls"] == {"pre": "https://pre/{z}/{x}/{y}", "post": "https://post/{z}/{x}/{y}"}
    assert captured["zoom"] == 18
    assert captured["osm_tags"] == {"building": ["*"]}
    item = json.loads((out / "stac-item.json").read_text())
    assert item["assets"]["chips-pre"]["href"] == "chips-pre"
    assert item["properties"]["fair:source_imagery"] == "https://post/{z}/{x}/{y}"


def test_dataset_build_own_labels_need_a_license(tmp_path, monkeypatch) -> None:
    _capture_materialize(monkeypatch)
    labels = tmp_path / "labels.geojson"
    labels.write_text(json.dumps({"type": "FeatureCollection", "features": []}))
    model = _model_with_preview(tmp_path, ["image"])
    command = ["dataset", "build", "--from-preview", str(model), "--labels", str(labels), "--out", str(tmp_path / "s")]

    missing_license = runner.invoke(app, command)
    assert missing_license.exit_code != 0
    assert "--labels-license" in click.unstyle(missing_license.output)

    result = runner.invoke(app, [*command, "--labels-license", "CC-BY-4.0"])
    assert result.exit_code == 0, result.output
    assert json.loads((tmp_path / "s" / "stac-item.json").read_text())["properties"]["license"] == "CC-BY-4.0"


def test_dataset_build_tms_override_needs_a_single_input_model(tmp_path, monkeypatch) -> None:
    _capture_materialize(monkeypatch)
    model = _model_with_preview(tmp_path, ["post", "pre"])

    result = runner.invoke(
        app,
        [
            "dataset",
            "build",
            "--from-preview",
            str(model),
            "--tms",
            "https://other/{z}/{x}/{y}",
            "--osm-url",
            "u",
            "--out",
            str(tmp_path / "s"),
        ],
    )

    assert result.exit_code != 0
    assert "replaces one input" in result.output


@pytest.mark.parametrize(
    ("env_url", "flag_url", "expected"),
    [
        (None, None, "https://api-prod.raw-data.hotosm.org/v1"),
        ("https://env-osm", None, "https://env-osm"),
        ("https://env-osm", "https://flag-osm", "https://flag-osm"),
    ],
)
def test_dataset_build_raw_data_url_defaults_and_overrides(tmp_path, monkeypatch, env_url, flag_url, expected) -> None:
    captured = _capture_materialize(monkeypatch)
    monkeypatch.delenv("FAIR_RAW_DATA_API_URL", raising=False)
    if env_url:
        monkeypatch.setenv("FAIR_RAW_DATA_API_URL", env_url)
    aoi = tmp_path / "aoi.geojson"
    aoi.write_text(json.dumps({"type": "Polygon", "coordinates": _SQUARE}))
    command = ["dataset", "build", "--tms", "https://t/{z}/{x}/{y}", "--aoi", str(aoi), "--zoom", "19"]
    command += ["--out", str(tmp_path / "d"), *(["--osm-url", flag_url] if flag_url else [])]

    result = runner.invoke(app, command)

    assert result.exit_code == 0, result.output
    assert captured["osm_api_url"] == expected


def test_knative_register_applies_the_service(tmp_path, monkeypatch) -> None:
    import fair.infra.knative as knative_module

    item = SimpleNamespace(id="dinov3s-buildings", properties={"mlm:name": "dinov3s_buildings"})
    monkeypatch.setattr("fair.cli.pystac.Item.from_file", lambda _: item)
    applied: list[tuple[object, str | None]] = []
    monkeypatch.setattr(
        knative_module,
        "ensure_knative_service",
        lambda stac_item, namespace=None: applied.append((stac_item, namespace)),
    )
    stac_item_path = tmp_path / "stac-item.json"
    stac_item_path.write_text("{}")

    result = runner.invoke(app, ["knative", "register", str(stac_item_path), "--namespace", "serving"])

    assert result.exit_code == 0
    assert applied == [(item, "serving")]
    assert "dinov3s-buildings" in result.output


def test_knative_status_reports_readiness(monkeypatch) -> None:
    import fair.infra.knative as knative_module

    monkeypatch.setattr(
        knative_module,
        "knative_service_status",
        lambda name, namespace=None: ("True", f"https://{name}.predict.example.com"),
    )

    result = runner.invoke(app, ["knative", "status", "dinov3s-buildings"])

    assert result.exit_code == 0
    assert "ready=True" in result.output
    assert "https://dinov3s-buildings.predict.example.com" in result.output


def test_knative_delete_delegates(monkeypatch) -> None:
    import fair.infra.knative as knative_module

    deleted: list[tuple[str, str | None]] = []
    monkeypatch.setattr(
        knative_module,
        "delete_knative_service",
        lambda name, namespace=None: deleted.append((name, namespace)),
    )

    result = runner.invoke(app, ["knative", "delete", "dinov3s_buildings"])

    assert result.exit_code == 0
    assert deleted == [("dinov3s_buildings", None)]
    assert "deleted dinov3s-buildings" in result.output


def test_dataset_build_accepts_only_the_label_geometries_the_model_declares(tmp_path, monkeypatch) -> None:
    _capture_materialize(monkeypatch)
    model = _model_with_preview(tmp_path, ["image"])
    out = tmp_path / "sample"
    command = ["dataset", "build", "--from-preview", str(model), "--geometry-type", "point", "--out", str(out)]
    env = {"FAIR_RAW_DATA_API_URL": "https://osm"}

    rejected = runner.invoke(app, command, env=env)
    assert rejected.exit_code != 0
    assert "the model accepts" in rejected.output

    item = json.loads(model.read_text())
    item["properties"]["fair:label_geometry_types"] = ["point", "polygon"]
    model.write_text(json.dumps(item))
    accepted = runner.invoke(app, command, env=env)

    assert accepted.exit_code == 0, accepted.output
    properties = json.loads((out / "stac-item.json").read_text())["properties"]
    assert properties["keywords"] == ["building", "semantic-segmentation", "point"]
    assert properties["fair:geometry_type"] == "point"


@pytest.mark.parametrize(
    ("flag", "expected"),
    [
        ([], {"building": ["*"]}),
        (["--osm-tags", "natural=tree,natural=shrub,building"], {"natural": ["tree", "shrub"], "building": ["*"]}),
        (["--osm-tags", "building=house,building"], {"building": ["*"]}),
    ],
)
def test_dataset_build_osm_tags_default_and_override(tmp_path, monkeypatch, flag, expected) -> None:
    captured = _capture_materialize(monkeypatch)
    aoi = tmp_path / "aoi.geojson"
    aoi.write_text(json.dumps({"type": "Polygon", "coordinates": _SQUARE}))
    command = ["dataset", "build", "--tms", "https://t/{z}/{x}/{y}", "--aoi", str(aoi), "--zoom", "19"]

    result = runner.invoke(app, [*command, "--out", str(tmp_path / "d"), *flag], env={"FAIR_RAW_DATA_API_URL": "u"})

    assert result.exit_code == 0, result.output
    assert captured["osm_tags"] == expected


def test_dataset_build_from_a_model_without_osm_tags_needs_them(tmp_path, monkeypatch) -> None:
    _capture_materialize(monkeypatch)
    model = _model_with_preview(tmp_path, ["image"])
    item = json.loads(model.read_text())
    del item["properties"]["mlm:output"][0]["classification:classes"][1]["fair:osm_tags"]
    model.write_text(json.dumps(item))

    result = runner.invoke(app, ["dataset", "build", "--from-preview", str(model), "--out", str(tmp_path / "s")])

    assert result.exit_code != 0
    assert "no fair:osm_tags" in result.output


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--labels", "LABELS", "--osm-tags", "building"], "--osm-tags"),
        (["--class-property", "kind"], "--class-property"),
    ],
)
def test_dataset_build_rejects_options_that_do_not_apply(tmp_path, monkeypatch, extra, message) -> None:
    _capture_materialize(monkeypatch)
    labels = tmp_path / "labels.geojson"
    labels.write_text(json.dumps({"type": "FeatureCollection", "features": []}))
    model = _model_with_preview(tmp_path, ["image"])
    extra = [str(labels) if arg == "LABELS" else arg for arg in extra]

    result = runner.invoke(
        app,
        [
            "dataset",
            "build",
            "--from-preview",
            str(model),
            "--labels-license",
            "MIT",
            "--out",
            str(tmp_path / "s"),
            *extra,
        ],
    )

    assert result.exit_code != 0
    assert message in click.unstyle(result.output)
