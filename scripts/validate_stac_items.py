#!/usr/bin/env python
"""Validate every model STAC item and, for trainable models, the sample dataset each one ships."""

import glob
import sys
from pathlib import Path

import pystac

from fair.stac.validators import is_trainable, validate_compatibility, validate_item, validate_model_weight_href

SAMPLE_MAX_BYTES = 5 * 1024 * 1024


def sample_errors(model: pystac.Item, model_dir: Path) -> list[str]:
    sample_path = model_dir / "sample" / "stac-item.json"
    if not sample_path.is_file():
        return [f"trainable model has no sample dataset at {sample_path}"]
    sample = pystac.Item.from_file(str(sample_path))
    errors = [f"sample: {err}" for err in validate_item(sample)]
    errors += [f"sample: {err}" for err in validate_compatibility(model, sample)]
    for key, asset in sample.assets.items():
        target = sample_path.parent / asset.href
        if "data" in (asset.roles or []) and not any(target.glob("*.tif")):
            errors.append(f"sample asset {key} has no .tif chips at {target}")
        if "labels" in (asset.roles or []) and not target.is_file():
            errors.append(f"sample labels file is missing at {target}")
    size = sum(f.stat().st_size for f in sample_path.parent.rglob("*") if f.is_file())
    if size > SAMPLE_MAX_BYTES:
        errors.append(f"sample is {size / 1e6:.1f} MB, over the {SAMPLE_MAX_BYTES / 1e6:.1f} MB limit")
    return errors


def main() -> int:
    paths = sorted(glob.glob("models/*/stac-item.json"))
    if not paths:
        print("No stac-item.json found")
        return 1

    failed = False
    for path in paths:
        item = pystac.Item.from_file(path)
        errors = validate_item(item) + validate_model_weight_href(item)
        if is_trainable(item.assets):
            errors += sample_errors(item, Path(path).parent)

        if errors:
            failed = True
            print(f"FAIL {path}")
            for err in errors:
                print(f"  {err}")
        else:
            print(f"OK   {path}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
