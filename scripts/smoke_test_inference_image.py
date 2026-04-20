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
    input_images: str,
    confidence_threshold: float,
    timeout_seconds: float,
) -> dict[str, Any]:
    payload = {
        "model_uri": model_uri,
        "input_images": input_images,
        "params": {"confidence_threshold": confidence_threshold},
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
    parser.add_argument("--input-images", required=True)
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--health-timeout", type=float, default=60.0)
    parser.add_argument("--predict-timeout", type=float, default=120.0)
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
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
                "--rm",
                "-d",
                "--name",
                container_name,
                "-e",
                f"MODEL_MODULE={args.model_module}",
                "-v",
                f"{workspace}:/workspace",
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
        result = _predict(
            base_url=base_url,
            model_uri=args.model_uri,
            input_images=args.input_images,
            confidence_threshold=args.confidence_threshold,
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
