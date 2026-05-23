"""End-to-end smoke test for a deployed fAIr API.
Usage:
    uv run python scripts/smoke_e2e.py                              # against api.fair.krschap.tech
    uv run python scripts/smoke_e2e.py --api https://api.example.com
    FAIR_DEV_TOKEN=... uv run python scripts/smoke_e2e.py

Token:
    1. --token CLI flag
    2. $FAIR_DEV_TOKEN env var
    3. kubectl get secret fair-backend-secrets -n fair -o jsonpath='{.data.FAIR_DEV_TOKEN}'
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime

import httpx

logging.getLogger("httpx").setLevel(logging.WARNING)

DEFAULT_API = "https://api.fair.krschap.tech"
DEFAULT_BASE_MODEL = "yolo11n-detection"
DEFAULT_IMAGERY = "https://tiles.openaerialmap.org/62d85d11d8499800053796c1/0/62d85d11d8499800053796c2/{z}/{x}/{y}"
AOI_POLYGON = [
    [85.51678, 27.63133],
    [85.52323, 27.63133],
    [85.52323, 27.63743],
    [85.51678, 27.63743],
    [85.51678, 27.63133],
]
BBOX = [85.51678, 27.63133, 85.52323, 27.63743]


@dataclass
class Config:
    api: str
    token: str
    base_model: str
    imagery: str
    build_timeout: int
    train_timeout: int
    predict_timeout: int
    poll_interval: int


class SmokeError(RuntimeError):
    """Step failure. The CLI entrypoint prints `exc` to stderr and exits 1."""


def _stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def log(msg: str) -> None:
    print(f"[smoke {_stamp()}] {msg}", flush=True)


def resolve_token(cli_token: str | None) -> str:
    if cli_token:
        return cli_token
    if env := os.environ.get("FAIR_DEV_TOKEN"):
        return env
    try:
        out = subprocess.check_output(
            [
                "kubectl",
                "get",
                "secret",
                "-n",
                "fair",
                "fair-backend-secrets",
                "-o",
                "jsonpath={.data.FAIR_DEV_TOKEN}",
            ],
            stderr=subprocess.PIPE,
            timeout=10,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as exc:
        raise SmokeError("no token: pass --token, set FAIR_DEV_TOKEN, or have kubectl access") from exc
    return base64.b64decode(out).decode().strip()


def _request(client: httpx.Client, method: str, path: str, *, expect: tuple[int, ...] = (200, 201, 202)) -> dict:
    return _request_with_body(client, method, path, body=None, expect=expect)


def _request_with_body(
    client: httpx.Client,
    method: str,
    path: str,
    *,
    body: dict | None,
    expect: tuple[int, ...],
) -> dict:
    resp = client.request(method, path, json=body)
    if resp.status_code not in expect:
        try:
            detail = json.dumps(resp.json(), indent=2)
        except ValueError:
            detail = resp.text
        raise SmokeError(f"{method} {path} -> HTTP {resp.status_code} (expected {expect})\n{detail}")
    return resp.json()


TERMINAL_FAILURE_STATES = frozenset({"failed", "cancelled", "canceled", "error", "errored"})


def _poll_until(
    client: httpx.Client,
    path: str,
    *,
    label: str,
    predicate,
    timeout: int,
    poll_interval: int,
    logs_path: str | None = None,
) -> dict:
    deadline = time.monotonic() + timeout
    last_status: str | None = None
    while time.monotonic() < deadline:
        try:
            body = _request(client, "GET", path)
        except SmokeError as exc:
            log(f"  …{label} (poll error, retrying): {exc}")
            time.sleep(poll_interval)
            continue
        status = (body.get("status") or body.get("build_status") or "?").lower()
        if predicate(body):
            log(f"  {label} ok (status={status})")
            return body
        if status in TERMINAL_FAILURE_STATES:
            zenml_run_id = body.get("zenml_run_id")
            tail = _fetch_logs(client, logs_path, zenml_run_id) if logs_path else ""
            raise SmokeError(f"{label} reached terminal state '{status}': {body}{tail}")
        if status != last_status:
            log(f"  …{label} (status={status})")
            last_status = status
        time.sleep(poll_interval)
    raise SmokeError(f"{label} did not finish within {timeout}s")


def _fetch_logs(client: httpx.Client, logs_path_template: str | None, zenml_run_id: str | None) -> str:
    if not logs_path_template or not zenml_run_id:
        return ""
    try:
        body = _request(client, "GET", logs_path_template.format(run_id=zenml_run_id))
    except SmokeError as exc:
        return f"\n(could not fetch logs: {exc})"
    raw = body.get("logs") if isinstance(body, dict) else None
    text = raw if isinstance(raw, str) else json.dumps(body, indent=2)
    return f"\n[last logs from /trainings/runs/{zenml_run_id}/logs/]\n{text}"


def run(cfg: Config) -> int:
    client = httpx.Client(
        base_url=cfg.api,
        headers={"Authorization": f"Bearer {cfg.token}"},
        timeout=30.0,
    )
    with client:
        log(f"API={cfg.api}  base_model={cfg.base_model}")

        health = _request(client, "GET", "/api/v1/health/")
        for k in ("postgresql", "s3", "stac_api", "zenml"):
            if not health.get(k):
                raise SmokeError(f"health check failed: {k}=false ({health})")
        log("0) health ok")

        aoi = _request_with_body(
            client,
            "POST",
            "/api/v1/aois/",
            body={
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [AOI_POLYGON]},
                "properties": {"dataset": None},
            },
            expect=(200, 201),
        )
        aoi_id = aoi.get("properties", {}).get("id") or aoi.get("id")
        if not isinstance(aoi_id, int):
            raise SmokeError(f"no AOI id in response: {aoi}")
        log(f"1) AOI created  id={aoi_id}")

        title = f"smoke-{_stamp()}"
        ds = _request_with_body(
            client,
            "POST",
            "/api/v1/datasets/build/",
            body={
                "title": title,
                "description": "e2e smoke",
                "source_imagery": cfg.imagery,
                "zoom": 19,
                "aoi_ids": [aoi_id],
                "label_tasks": ["object-detection"],
                "label_classes": [{"name": "building", "classes": ["*"]}],
                "keywords": ["building"],
                "label_type": "vector",
                "geometry_type": "polygon",
            },
            expect=(200, 201, 202),
        )
        dataset_id = ds["id"]
        stac_id = ds["stac_id"]
        log(f"2) dataset submitted  id={dataset_id}  stac={stac_id}")

        _poll_until(
            client,
            f"/api/v1/datasets/{dataset_id}/",
            label="dataset build",
            predicate=lambda b: b.get("build_status") == "published",
            timeout=cfg.build_timeout,
            poll_interval=cfg.poll_interval,
        )

        tr = _request_with_body(
            client,
            "POST",
            "/api/v1/trainings/submit/",
            body={
                "base_model_stac_id": cfg.base_model,
                "dataset_stac_id": stac_id,
                "model_name": f"yolo11n-detection-smoke-{_stamp()}",
                "overrides": {
                    "epochs": 3,
                    "batch_size": 2,
                    "learning_rate": 0.01,
                    "chip_size": 640,
                },
            },
            expect=(200, 201, 202),
        )
        tr_id = tr["id"]
        log(f"4) training submitted  id={tr_id}")

        _poll_until(
            client,
            f"/api/v1/trainings/{tr_id}/",
            label="training",
            predicate=lambda b: b.get("status") == "completed",
            timeout=cfg.train_timeout,
            poll_interval=cfg.poll_interval,
            logs_path="/api/v1/trainings/runs/{run_id}/logs/?tail=200",
        )

        pub = _request_with_body(
            client,
            "POST",
            f"/api/v1/trainings/{tr_id}/publish/",
            body={"description": "smoke"},
            expect=(200, 201),
        )
        lm_stac_id = pub.get("local_model_stac_id") or pub.get("stac_id")
        if not lm_stac_id:
            raise SmokeError(f"publish returned no stac id: {pub}")
        log(f"5) promoted  local_model_stac_id={lm_stac_id}")

        pr = _request_with_body(
            client,
            "POST",
            "/api/v1/predictions/submit/",
            body={
                "model_stac_id": lm_stac_id,
                "image_uri": cfg.imagery,
                "bbox": BBOX,
                "zoom": 19,
                "params": {"confidence_threshold": 0.25},
            },
            expect=(200, 201, 202),
        )
        pred_id = pr["id"]
        log(f"6) prediction submitted  id={pred_id}")

        _poll_until(
            client,
            f"/api/v1/predictions/{pred_id}/",
            label="prediction",
            predicate=lambda b: b.get("results_ready") is True,
            timeout=cfg.predict_timeout,
            poll_interval=cfg.poll_interval,
            logs_path="/api/v1/predictions/runs/{run_id}/logs/?tail=200",
        )

        result = _request(client, "GET", f"/api/v1/predictions/{pred_id}/result/")
        for key in ("geojson", "fgb", "pmtiles"):
            url = result.get(key)
            if not url or not url.startswith("http"):
                raise SmokeError(f"missing {key} URL in result: {result}")
            probe = httpx.get(
                url,
                headers={"Range": "bytes=0-0"},
                timeout=15.0,
                follow_redirects=True,
            )
            if probe.status_code not in (200, 206):
                raise SmokeError(f"{key} URL returned HTTP {probe.status_code}: {url}")
            log(f"7) {key} ok")

        log(
            "ALL PASSED  "
            f"aoi={aoi_id} dataset={dataset_id} training={tr_id} "
            f"local_model={lm_stac_id} prediction={pred_id}"
        )
    return 0


def parse_args() -> Config:
    summary = (__doc__ or "").split("\n\n", 1)[0]
    p = argparse.ArgumentParser(description=summary)
    p.add_argument("--api", default=os.environ.get("FAIR_API", DEFAULT_API))
    p.add_argument("--token", default=None, help="overrides $FAIR_DEV_TOKEN and kubectl lookup")
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    p.add_argument("--imagery", default=DEFAULT_IMAGERY)
    p.add_argument("--build-timeout", type=int, default=600)
    p.add_argument("--train-timeout", type=int, default=2400)
    p.add_argument("--predict-timeout", type=int, default=1200)
    p.add_argument("--poll-interval", type=int, default=15)
    args = p.parse_args()
    return Config(
        api=args.api.rstrip("/"),
        token=resolve_token(args.token),
        base_model=args.base_model,
        imagery=args.imagery,
        build_timeout=args.build_timeout,
        train_timeout=args.train_timeout,
        predict_timeout=args.predict_timeout,
        poll_interval=args.poll_interval,
    )


if __name__ == "__main__":
    try:
        sys.exit(run(parse_args()))
    except SmokeError as exc:
        print(f"[smoke {_stamp()}] FAIL: {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print(f"[smoke {_stamp()}] interrupted", file=sys.stderr)
        sys.exit(130)
