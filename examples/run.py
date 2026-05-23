"""Run a fAIr example pipeline end-to-end for any model in models/.

Usage:
    python examples/run.py <model>                 # use STAC defaults
    python examples/run.py <model> --epochs 1      # override one hyperparameter
    python examples/run.py                         # iterate every model in models/
"""

import argparse
import json
import os
import sys
from pathlib import Path

from fair.client import FairClient
from fair.utils import install_s3_cleanup_handler

OVERRIDES = ("epochs", "batch_size", "learning_rate", "samples_per_epoch", "chip_size")
CI_OVERRIDES: dict[str, object] = {
    "epochs": 5,
    "batch_size": 2,
    "sample_fraction": 0.1,
    "samples_per_epoch": 12,
}


def run(model: str, overrides: dict[str, object]) -> None:
    stac_path = Path(f"models/{model}/stac-item.json")
    stac = json.loads(stac_path.read_text())
    base_id = stac["id"]
    task = stac["properties"]["mlm:tasks"][0]
    dataset_path = f"data/sample/buildings-banepa-{task}/stac-item.json"

    client = FairClient(
        zenml_store_url=os.environ.get("FAIR_ZENML_STORE_URL"),
        stac_api_url=os.environ.get("FAIR_STAC_API_URL"),
        dsn=os.environ.get("FAIR_DSN"),
        user_id=os.environ.get("FAIR_USER_ID", "anonymous"),
        config_dir=f"examples/{model}/config",
        upload_artifacts=os.environ.get("FAIR_UPLOAD_ARTIFACTS", "").lower() == "true",
    )
    install_s3_cleanup_handler()
    client.setup()

    base_model_id = client.register_base_model(str(stac_path))
    dataset_id = client.register_dataset(dataset_path)
    finetuned_model_id = client.finetune(
        base_model_id=base_model_id,
        dataset_id=dataset_id,
        model_name=f"{base_id}-finetuned-banepa",
        overrides=overrides,
    )
    local_model_id = client.promote(
        finetuned_model_id,
        description=f"{base_id} finetuned on buildings-banepa-{task}",
    )
    client.predict(local_model_id, image_path="data/sample/test/oam")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", nargs="?", help="Model directory under models/. Omit to run all.")
    for name in OVERRIDES:
        parser.add_argument(f"--{name.replace('_', '-')}", dest=name, type=str, default=None)
    args = parser.parse_args()

    overrides: dict[str, object] = dict(CI_OVERRIDES) if os.environ.get("FAIR_CI") == "1" else {}
    for name in OVERRIDES:
        raw = getattr(args, name)
        if raw is None:
            continue
        overrides[name] = float(raw) if name == "learning_rate" else int(raw)

    if args.model:
        models = [args.model]
    else:
        models = sorted(p.name for p in Path("models").iterdir() if (p / "stac-item.json").exists())
        if not models:
            sys.exit("no models found under models/")

    for m in models:
        print(f"\n=== {m} ===")
        run(m, overrides)
    return 0


if __name__ == "__main__":
    sys.exit(main())
