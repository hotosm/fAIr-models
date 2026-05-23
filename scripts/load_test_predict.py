"""Locust step-load test for the three live `/predict` endpoints.

Run: uvx --from "locust==2.32.4" locust -f scripts/load_test_predict.py
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from typing import Any, ClassVar

from locust import FastHttpUser, LoadTestShape, between, events, task

TIMEOUT = 180
WARMUP_TIMEOUT = 240

PAYLOADS: dict[str, dict[str, Any]] = {
    "resnet18": {
        "host": "https://resnet18-classification.predict.fair.krschap.tech",
        "body": {
            "model_uri": "https://s3.fair.krschap.tech/zenml/local-models/27f6f5a6-d079-44ac-80b1-be9ff268c2cb/model/model.onnx",
            "image_uri": "https://tiles.openaerialmap.org/62d85d11d8499800053796c1/0/62d85d11d8499800053796c2/{z}/{x}/{y}",
            "bbox": [85.51678033745037, 27.6313353660439, 85.52323021107895, 27.637438390948745],
            "zoom": 18,
            "params": {"confidence_threshold": 0.5},
        },
    },
    "unet": {
        "host": "https://unet-segmentation.predict.fair.krschap.tech",
        "body": {
            "model_uri": "https://s3.fair.krschap.tech/zenml/local-models/08e20666-f8fa-4b8a-8fe8-72661a590fd0/model/model.onnx",
            "image_uri": "https://tiles.openaerialmap.org/62d85d11d8499800053796c1/0/62d85d11d8499800053796c2/{z}/{x}/{y}",
            "bbox": [85.51678033745037, 27.6313353660439, 85.52323021107895, 27.637438390948745],
            "zoom": 18,
            "params": {"confidence_threshold": 0.5, "min_class_value": 1},
        },
    },
    "yolo11n": {
        "host": "https://yolo11n-detection.predict.fair.krschap.tech",
        "body": {
            "model_uri": "https://s3.fair.krschap.tech/zenml/local-models/1e398477-2472-46ea-9286-cd89411e1c32/model/model.onnx",
            "image_uri": "https://tiles.openaerialmap.org/62d85d11d8499800053796c1/0/62d85d11d8499800053796c2/{z}/{x}/{y}",
            "bbox": [85.51678033745037, 27.6313353660439, 85.52323021107895, 27.637438390948745],
            "zoom": 18,
            "params": {"confidence_threshold": 0.3, "iou_threshold": 0.45},
        },
    },
}


def _make_user(model: str) -> type[FastHttpUser]:
    cfg = PAYLOADS[model]

    class _U(FastHttpUser):
        host = cfg["host"]
        wait_time = between(0.5, 1.5)
        network_timeout = float(TIMEOUT)
        connection_timeout = 30.0

        @task
        def predict(self) -> None:
            with self.client.post(
                "/predict",
                json=cfg["body"],
                name=f"{model} /predict",
                timeout=TIMEOUT,
                catch_response=True,
            ) as r:
                if r.status_code != 200:
                    body = r.text[:200] if r.text else "<empty>"
                    r.failure(f"{r.status_code}: {body}")

    _U.__name__ = f"{model.title()}User"
    return _U


ResnetUser = _make_user("resnet18")
UnetUser = _make_user("unet")
YoloUser = _make_user("yolo11n")


def _warmup_one(name: str, cfg: dict[str, Any]) -> None:
    body = json.dumps(cfg["body"]).encode()
    req = urllib.request.Request(
        f"{cfg['host']}/predict",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=WARMUP_TIMEOUT) as r:
            print(f"warmup {name}: HTTP {r.status}")
    except (urllib.error.URLError, TimeoutError) as e:
        print(f"warmup {name} failed (continuing): {e}")


@events.test_start.add_listener
def _warmup(environment: Any, **_kwargs: Any) -> None:
    print("warm-up: firing one request per service in parallel")
    threads = [threading.Thread(target=_warmup_one, args=(name, cfg)) for name, cfg in PAYLOADS.items()]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print(" warm-up done; starting load")


class StepLoad(LoadTestShape):
    stages: ClassVar[list[dict[str, int]]] = [
        {"duration": 60, "users": 3, "spawn_rate": 1},
        {"duration": 120, "users": 6, "spawn_rate": 1},
        {"duration": 180, "users": 12, "spawn_rate": 2},
        {"duration": 240, "users": 18, "spawn_rate": 2},
        {"duration": 300, "users": 24, "spawn_rate": 2},
    ]

    def tick(self) -> tuple[int, float] | None:
        elapsed = self.get_run_time()
        for stage in self.stages:
            if elapsed < stage["duration"]:
                return stage["users"], stage["spawn_rate"]
        return None
