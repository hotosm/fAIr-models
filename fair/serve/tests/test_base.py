"""Tests for the Starlette serving app and ONNX session loader."""

import asyncio
import importlib
import json
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

VALID_REQUEST = {
    "model_uri": "s3://bucket/model.onnx",
    "image_uri": "https://tiles.example.com/{z}/{x}/{y}.png",
    "bbox": [85.5, 27.6, 85.52, 27.63],
    "zoom": 18,
    "params": {"confidence_threshold": 0.5},
}


def _write_stub_pipeline(tmp_path: Path, module_name: str, return_payload: dict[str, Any]) -> None:
    module_dir = tmp_path
    for part in module_name.split(".")[:-1]:
        module_dir = module_dir / part
        module_dir.mkdir(exist_ok=True)
        (module_dir / "__init__.py").write_text("")
    final = module_dir / f"{module_name.rsplit('.', 1)[-1]}.py"
    final.write_text(
        textwrap.dedent(
            f"""
            def predict(session, input_images, params):
                return {return_payload!r}
            """
        )
    )


def _patch_fetch_chips(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[dict[str, Any]]:
    from fair.serve import base as serve_base

    chips_calls: list[dict[str, Any]] = []

    async def fake_fetch_chips(image_uri: str, bbox: list[float], zoom: int, out_dir: str) -> str:
        chips_calls.append({"image_uri": image_uri, "bbox": bbox, "zoom": zoom, "out_dir": out_dir})
        chips = Path(out_dir) / "chips"
        chips.mkdir(exist_ok=True)
        return str(chips)

    monkeypatch.setattr(serve_base, "_fetch_chips", fake_fetch_chips)
    return chips_calls


def test_load_session_is_cached(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from fair.serve import base as serve_base

    serve_base.load_session.cache_clear()
    calls: list[str] = []

    class FakeSession:
        def __init__(self, path: str) -> None:
            calls.append(path)

    monkeypatch.setattr(serve_base, "__name__", "fair.serve.base")

    fake_model = tmp_path / "model.onnx"
    fake_model.write_bytes(b"onnx")

    import onnxruntime

    monkeypatch.setattr(
        onnxruntime, "InferenceSession", lambda path, sess_options=None, providers=None: FakeSession(path)
    )

    serve_base.load_session(str(fake_model))
    serve_base.load_session(str(fake_model))
    assert len(calls) == 1
    serve_base.load_session.cache_clear()


def test_health_and_predict_routes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from starlette.testclient import TestClient

    module_name = "stubmodel.pipeline"
    stub_root = tmp_path / "stubsrc"
    stub_root.mkdir()
    _write_stub_pipeline(stub_root, module_name, {"type": "FeatureCollection", "features": []})
    sys.path.insert(0, str(stub_root))

    try:
        monkeypatch.setenv("MODEL_MODULE", module_name)
        from fair.serve import base as serve_base

        serve_base.load_session.cache_clear()

        fake_model = tmp_path / "model.onnx"
        fake_model.write_bytes(b"onnx")

        class FakeSession:
            pass

        import onnxruntime

        monkeypatch.setattr(
            onnxruntime,
            "InferenceSession",
            lambda path, sess_options=None, providers=None: FakeSession(),
        )

        chips_calls = _patch_fetch_chips(monkeypatch, tmp_path)

        importlib.invalidate_caches()
        app = serve_base.create_app()
        with TestClient(app) as client:
            health = client.get("/health")
            assert health.status_code == 200
            assert health.json() == {"status": "ok"}

            request = {**VALID_REQUEST, "model_uri": str(fake_model)}
            resp = client.post("/predict", json=request)
            assert resp.status_code == 200, resp.text
            assert resp.json() == {"type": "FeatureCollection", "features": []}

            assert len(chips_calls) == 1
            assert chips_calls[0]["image_uri"] == request["image_uri"]
            assert chips_calls[0]["bbox"] == request["bbox"]
            assert chips_calls[0]["zoom"] == request["zoom"]
    finally:
        sys.path.remove(str(stub_root))
        serve_base.load_session.cache_clear()


def test_predict_forwards_bbox_when_pipeline_accepts_it(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from starlette.testclient import TestClient

    module_name = "bboxmodel.pipeline"
    stub_root = tmp_path / "stubsrc_bbox"
    package_dir = stub_root / "bboxmodel"
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text("")
    (package_dir / "pipeline.py").write_text(
        textwrap.dedent(
            """
            def predict(session, input_images, params, bbox=None):
                return {"type": "FeatureCollection", "features": [], "echo_bbox": bbox}
            """
        )
    )
    sys.path.insert(0, str(stub_root))

    try:
        monkeypatch.setenv("MODEL_MODULE", module_name)
        from fair.serve import base as serve_base

        serve_base.load_session.cache_clear()

        fake_model = tmp_path / "model.onnx"
        fake_model.write_bytes(b"onnx")

        class FakeSession:
            pass

        import onnxruntime

        monkeypatch.setattr(
            onnxruntime,
            "InferenceSession",
            lambda path, sess_options=None, providers=None: FakeSession(),
        )
        _patch_fetch_chips(monkeypatch, tmp_path)

        importlib.invalidate_caches()
        app = serve_base.create_app()
        with TestClient(app) as client:
            request = {**VALID_REQUEST, "model_uri": str(fake_model)}
            resp = client.post("/predict", json=request)
            assert resp.status_code == 200, resp.text
            assert resp.json()["echo_bbox"] == VALID_REQUEST["bbox"]
    finally:
        sys.path.remove(str(stub_root))
        serve_base.load_session.cache_clear()


@pytest.mark.parametrize(
    ("mutation", "expected_fragment"),
    [
        ({"model_uri": ""}, "model_uri"),
        ({"image_uri": ""}, "image_uri"),
        ({"bbox": [1.0, 2.0, 3.0]}, "bbox"),
        ({"bbox": [5.0, 5.0, 1.0, 1.0]}, "west<east"),
        ({"bbox": ["a", "b", "c", "d"]}, "numbers"),
        ({"zoom": "high"}, "zoom"),
        ({"zoom": 99}, "zoom"),
        ({"params": [1, 2]}, "params"),
    ],
)
def test_predict_rejects_invalid_fields(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mutation: dict[str, Any],
    expected_fragment: str,
) -> None:
    from starlette.testclient import TestClient

    module_name = "stubmodel_invalid.pipeline"
    stub_root = tmp_path / "stubsrc"
    stub_root.mkdir(exist_ok=True)
    _write_stub_pipeline(stub_root, module_name, {"type": "FeatureCollection", "features": []})
    sys.path.insert(0, str(stub_root))

    try:
        monkeypatch.setenv("MODEL_MODULE", module_name)
        from fair.serve import base as serve_base

        app = serve_base.create_app()
        with TestClient(app) as client:
            resp = client.post("/predict", json={**VALID_REQUEST, **mutation})
            assert resp.status_code == 400
            assert expected_fragment in resp.json()["error"]
    finally:
        sys.path.remove(str(stub_root))


def test_predict_rejects_missing_required_field(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from starlette.testclient import TestClient

    module_name = "stubmodel_missing.pipeline"
    stub_root = tmp_path / "stubsrc"
    stub_root.mkdir()
    _write_stub_pipeline(stub_root, module_name, {"type": "FeatureCollection", "features": []})
    sys.path.insert(0, str(stub_root))

    try:
        monkeypatch.setenv("MODEL_MODULE", module_name)
        from fair.serve import base as serve_base

        app = serve_base.create_app()
        with TestClient(app) as client:
            resp = client.post("/predict", json={"image_uri": "x"})
            assert resp.status_code == 400
    finally:
        sys.path.remove(str(stub_root))


def test_create_app_requires_model_module(monkeypatch: pytest.MonkeyPatch) -> None:
    from fair.serve.base import ServeConfigError, create_app

    monkeypatch.delenv("MODEL_MODULE", raising=False)
    with pytest.raises(ServeConfigError):
        create_app()


def test_create_app_requires_predict_function(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module_root = tmp_path / "badsrc"
    module_root.mkdir()
    package_dir = module_root / "badmodel"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("")
    (package_dir / "pipeline.py").write_text("VALUE = 1\n")
    sys.path.insert(0, str(module_root))

    try:
        monkeypatch.setenv("MODEL_MODULE", "badmodel.pipeline")
        from fair.serve.base import ServeConfigError, create_app

        with pytest.raises(ServeConfigError):
            create_app()
    finally:
        sys.path.remove(str(module_root))


def test_predict_handles_invalid_json_and_pipeline_errors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from starlette.testclient import TestClient

    stub_root = tmp_path / "stubsrc3"
    stub_root.mkdir()
    _write_stub_pipeline(stub_root, "errormodel.pipeline", {"unused": True})
    sys.path.insert(0, str(stub_root))

    try:
        monkeypatch.setenv("MODEL_MODULE", "errormodel.pipeline")
        from fair.serve import base as serve_base

        def _raise_error(*_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("boom")

        app = serve_base.create_app()
        with TestClient(app) as client:
            bad_json = client.post(
                "/predict",
                content="{not-json}",
                headers={"content-type": "application/json"},
            )
            assert bad_json.status_code == 400

            serve_base.load_session.cache_clear()
            monkeypatch.setattr(serve_base, "load_session", _raise_error)

            failed = client.post("/predict", json=VALID_REQUEST)
            assert failed.status_code == 500
            assert failed.json()["error"] == "boom"
    finally:
        sys.path.remove(str(stub_root))


def _request(*, client_disconnects: bool) -> Any:
    """Starlette request whose receive reports a disconnect or blocks like a live client."""
    from starlette.requests import Request

    body_sent = False

    async def receive() -> dict[str, Any]:
        nonlocal body_sent
        if not body_sent:
            body_sent = True
            return {"type": "http.request", "body": json.dumps(VALID_REQUEST).encode(), "more_body": False}
        if client_disconnects:
            return {"type": "http.disconnect"}
        await asyncio.Event().wait()
        return {}

    return Request({"type": "http", "method": "POST", "headers": []}, receive)


class _RecordingPipeline:
    def __init__(self) -> None:
        self.calls = 0

    def predict(self, session: Any, input_images: str, params: dict[str, Any]) -> dict[str, Any]:
        self.calls += 1
        return {"type": "FeatureCollection", "features": []}


def test_disconnect_during_chip_download_cancels_it_and_skips_predict(monkeypatch: pytest.MonkeyPatch) -> None:
    from fair.serve import base as serve_base

    download = {"started": False, "finished": False}

    async def slow_fetch_chips(image_uri: str, bbox: list[float], zoom: int, out_dir: str) -> str:
        download["started"] = True
        await asyncio.sleep(30)
        download["finished"] = True
        return out_dir

    monkeypatch.setattr(serve_base, "_fetch_chips", slow_fetch_chips)
    monkeypatch.setattr(serve_base, "load_session", lambda model_uri: object())
    monkeypatch.setattr(serve_base, "_DISCONNECT_POLL_SECONDS", 0.01)
    pipeline = _RecordingPipeline()

    response = asyncio.run(serve_base._predict(_request(client_disconnects=True), pipeline, False))

    assert response.status_code == 499
    assert download == {"started": True, "finished": False}
    assert pipeline.calls == 0


def test_disconnect_after_chip_download_skips_predict(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from fair.serve import base as serve_base

    _patch_fetch_chips(monkeypatch, tmp_path)
    monkeypatch.setattr(serve_base, "load_session", lambda model_uri: object())
    pipeline = _RecordingPipeline()

    response = asyncio.run(serve_base._predict(_request(client_disconnects=True), pipeline, False))

    assert response.status_code == 499
    assert pipeline.calls == 0


def test_connected_client_gets_predictions(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from fair.serve import base as serve_base

    _patch_fetch_chips(monkeypatch, tmp_path)
    monkeypatch.setattr(serve_base, "load_session", lambda model_uri: object())
    pipeline = _RecordingPipeline()

    response = asyncio.run(serve_base._predict(_request(client_disconnects=False), pipeline, False))

    assert response.status_code == 200
    assert pipeline.calls == 1


def test_cancelled_handler_cancels_chip_download(monkeypatch: pytest.MonkeyPatch) -> None:
    from fair.serve import base as serve_base

    download_started = asyncio.Event()
    download_cancelled = False

    async def slow_fetch_chips(image_uri: str, bbox: list[float], zoom: int, out_dir: str) -> str:
        nonlocal download_cancelled
        download_started.set()
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            download_cancelled = True
            raise
        return out_dir

    monkeypatch.setattr(serve_base, "_fetch_chips", slow_fetch_chips)
    monkeypatch.setattr(serve_base, "load_session", lambda model_uri: object())

    async def cancel_mid_download() -> None:
        handler = asyncio.ensure_future(
            serve_base._predict(_request(client_disconnects=False), _RecordingPipeline(), False)
        )
        await download_started.wait()
        handler.cancel()
        with pytest.raises(asyncio.CancelledError):
            await handler
        assert download_cancelled

    asyncio.run(cancel_mid_download())
