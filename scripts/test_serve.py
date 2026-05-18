"""End-to-end test: build a model's inference container, load the latest trained
ONNX from PgStac, POST /predict against OAM TMS over a Banepa bbox, print response.

Usage: test_serve.py <model-dir-name>   (e.g. test_serve.py yolo11n_detection)
"""

import json
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen

STAC_API = "http://localhost:8082"
PORT = 8090
OAM_TMS = "https://tiles.openaerialmap.org/62d85d11d8499800053796c1/0/62d85d11d8499800053796c2/{z}/{x}/{y}"
BANEPA_BBOX = [85.5217, 27.6300, 85.5224, 27.6336]
ZOOM = 19


def latest_local_model(base_id: str) -> str:
    with urlopen(f"{STAC_API}/collections/local-models/items?limit=100") as r:
        feats = json.load(r).get("features", [])
    feats.sort(key=lambda x: x["properties"].get("datetime", ""), reverse=True)
    for f in feats:
        if f["properties"].get("fair:base_model_id") == base_id:
            return f["assets"]["model"]["href"]
    sys.exit(f"no local-model with fair:base_model_id={base_id}; run 'just example <task>' first")


def inference_params(stac_item: Path) -> dict:
    hyp = json.loads(stac_item.read_text())["properties"].get("mlm:hyperparameters", {})
    return {k.removeprefix("inference."): v for k, v in hyp.items() if k.startswith("inference.")}


def wait_for_health() -> None:
    for _ in range(60):
        try:
            with urlopen(f"http://localhost:{PORT}/health", timeout=2) as r:
                if r.status == 200:
                    return
        except OSError:
            time.sleep(1)
    sys.exit("container did not become healthy within 60s")


def post_predict(payload: dict) -> tuple[int, str]:
    req = Request(
        f"http://localhost:{PORT}/predict",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(req, timeout=600) as r:
        return r.status, r.read().decode()


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--output", default="", help="Path to write the prediction GeoJSON (optional)")
    args = parser.parse_args()

    model = args.model
    base_id = model.replace("_", "-")
    stac_item = Path(f"models/{model}/stac-item.json")
    image = json.loads(stac_item.read_text())["assets"]["mlm:inference"]["href"]

    model_uri = latest_local_model(base_id)
    print(f"model_uri: {model_uri}")

    subprocess.run(
        ["docker", "build", "-f", f"models/{model}/Dockerfile", "--target", "inference", "-t", image, "."],
        check=True,
    )
    cid = subprocess.check_output(
        [
            "docker",
            "run",
            "-d",
            "--rm",
            "--network",
            "host",
            "-e",
            f"MODEL_MODULE=models.{model}.pipeline",
            image,
            "fair.serve.base:create_app",
            "--factory",
            "--host",
            "0.0.0.0",
            "--port",
            str(PORT),
        ],
        text=True,
    ).strip()

    try:
        wait_for_health()
        payload = {
            "model_uri": model_uri,
            "image_uri": OAM_TMS,
            "bbox": BANEPA_BBOX,
            "zoom": ZOOM,
            "params": inference_params(stac_item),
        }
        print("POSTing /predict ...")
        code, body = post_predict(payload)
        print(f"HTTP {code}")
        print(body[:800])
        if args.output and code == 200:
            Path(args.output).write_text(body)
        return 0 if code == 200 else 1
    finally:
        subprocess.run(["docker", "kill", cid], check=False, capture_output=True)


if __name__ == "__main__":
    sys.exit(main())
