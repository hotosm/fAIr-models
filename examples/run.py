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
from fair.stac.inputs import chips_hrefs_by_input
from fair.stac.validators import is_trainable
from fair.utils import install_s3_cleanup_handler

OVERRIDES = ("epochs", "batch_size", "learning_rate", "samples_per_epoch", "chip_size")
CI_OVERRIDES: dict[str, object] = {
    "epochs": 2,
    "batch_size": 2,
    "samples_per_epoch": 6,
}


def run(model: str, overrides: dict[str, object]) -> None:
    stac_path = Path(f"models/{model}/stac-item.json")
    stac = json.loads(stac_path.read_text())
    base_id = stac["id"]
    sample_path = Path(f"models/{model}/sample/stac-item.json")
    if not sample_path.is_file():
        sys.exit(f"{model} has no sample dataset at {sample_path}; see docs/contributing/model.md")
    sample = json.loads(sample_path.read_text())
    sample_chips = {
        name: str(sample_path.parent / href)
        for name, href in chips_hrefs_by_input({key: asset["href"] for key, asset in sample["assets"].items()}).items()
    }

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

    base_model_id = client.register_base_model(str(stac_path), pin_images=os.environ.get("FAIR_CI") != "1")
    dataset_id = client.register_dataset(str(sample_path))
    finetuned_model_id = client.finetune(
        base_model_id=base_model_id,
        dataset_id=dataset_id,
        model_name=f"{base_id}-finetuned-sample",
        overrides=overrides,
    )
    local_model_id = client.promote(
        finetuned_model_id,
        description=f"{base_id} finetuned on {sample['id']}",
    )
    client.predict(local_model_id, sample_chips)


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
        items = {p.name: p / "stac-item.json" for p in Path("models").iterdir() if (p / "stac-item.json").exists()}
        skipped = {name.strip() for name in os.environ.get("FAIR_EXAMPLE_SKIP", "").split(",") if name.strip()}
        if unknown := sorted(skipped - items.keys()):
            sys.exit(f"unknown models in FAIR_EXAMPLE_SKIP: {unknown}")
        models = sorted(
            name
            for name, item in items.items()
            if name not in skipped and is_trainable(json.loads(item.read_text())["assets"])
        )
        if not models:
            sys.exit("no trainable models under models/")

    for m in models:
        print(f"\n=== {m} ===")
        run(m, overrides)
    return 0


if __name__ == "__main__":
    sys.exit(main())
