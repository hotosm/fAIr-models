from __future__ import annotations

import argparse
import json
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any
from urllib import error, request


def _wait_for_health(base_url: str, timeout_seconds: float) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None

    while time.monotonic() < deadline:
        try:
            with request.urlopen(f"{base_url}/health", timeout=5) as response:
                payload = json.load(response)
            if payload.get("status") == "ok":
                return
        except Exception as exc:  # pragma: no cover
            last_error = exc
        time.sleep(1)

    msg = f"Timed out waiting for {base_url}/health"
    raise RuntimeError(msg) from last_error


def _predict(
    *,
    base_url: str,
    model_uri: str,
    image_uri: str,
    bbox: list[float],
    zoom: int,
    confidence_threshold: float,
    iou_threshold: float,
    min_class_value: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    payload = {
        "model_uri": model_uri,
        "image_uri": image_uri,
        "bbox": bbox,
        "zoom": zoom,
        "params": {
            "confidence_threshold": confidence_threshold,
            "iou_threshold": iou_threshold,
            "min_class_value": min_class_value,
        },
    }
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        f"{base_url}/predict",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=timeout_seconds) as response:
            result = json.load(response)
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        msg = f"Prediction request failed with status {exc.code}: {body}"
        raise RuntimeError(msg) from exc

    if result.get("type") != "FeatureCollection":
        msg = f"Unexpected response type: {result}"
        raise RuntimeError(msg)
    if not isinstance(result.get("features"), list):
        msg = f"Prediction response is missing a features list: {result}"
        raise RuntimeError(msg)
    return result


def _cleanup_container(container_name: str) -> None:
    subprocess.run(
        ["docker", "rm", "-f", container_name],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--model-module", required=True)
    parser.add_argument("--model-uri", required=True)
    parser.add_argument("--image-uri", required=True, help="TMS/XYZ/WMS/WMTS template URL")
    parser.add_argument(
        "--bbox",
        required=True,
        help="Comma-separated west,south,east,north in EPSG:4326",
    )
    parser.add_argument("--zoom", type=int, required=True)
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--health-timeout", type=float, default=60.0)
    parser.add_argument("--predict-timeout", type=float, default=120.0)
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--iou-threshold", type=float, default=0.45)
    parser.add_argument("--min-class-value", type=int, default=1)
    parser.add_argument("--container-name", default="")
    args = parser.parse_args()

    workspace = Path(args.workspace).resolve()
    container_name = args.container_name or f"fair-smoke-{uuid.uuid4().hex[:8]}"
    _cleanup_container(container_name)

    try:
        subprocess.run(
            [
                "docker",
                "run",
                "-d",
                "--name",
                container_name,
                "-e",
                f"MODEL_MODULE={args.model_module}",
                "-v",
                f"{workspace}:/workspace:ro",
                "-w",
                "/workspace",
                "-p",
                f"{args.port}:8080",
                args.image,
            ],
            check=True,
        )

        base_url = f"http://127.0.0.1:{args.port}"
        _wait_for_health(base_url, args.health_timeout)
        bbox = [float(v) for v in args.bbox.split(",")]
        if len(bbox) != 4:
            raise ValueError("--bbox must have 4 comma-separated numbers")
        result = _predict(
            base_url=base_url,
            model_uri=args.model_uri,
            image_uri=args.image_uri,
            bbox=bbox,
            zoom=args.zoom,
            confidence_threshold=args.confidence_threshold,
            iou_threshold=args.iou_threshold,
            min_class_value=args.min_class_value,
            timeout_seconds=args.predict_timeout,
        )
        print(json.dumps({"status": "ok", "feature_count": len(result["features"])}))
        return 0
    except Exception:
        subprocess.run(["docker", "logs", container_name], check=False)
        raise
    finally:
        _cleanup_container(container_name)


if __name__ == "__main__":
    raise SystemExit(main())
