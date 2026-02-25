#!/usr/bin/env python
"""Validate all models/*/stac-item.json against fAIr platform requirements."""

import glob
import sys

import pystac

from fair.stac.validators import validate_base_model_item


def main() -> int:
    paths = sorted(glob.glob("models/*/stac-item.json"))
    if not paths:
        print("No stac-item.json found under models/")
        return 1

    failed = False
    for path in paths:
        item = pystac.Item.from_file(path)
        errors = validate_base_model_item(item)
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
