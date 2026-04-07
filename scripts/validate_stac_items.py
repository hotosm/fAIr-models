#!/usr/bin/env python
"""Validate all STAC items (base models + datasets) against fAIr schemas."""

import glob
import sys

import pystac

from fair.stac.validators import validate_item


def main() -> int:
    paths = sorted(glob.glob("models/*/stac-item.json") + glob.glob("data/sample/*/stac-item.json"))
    if not paths:
        print("No stac-item.json found")
        return 1

    failed = False
    for path in paths:
        item = pystac.Item.from_file(path)
        errors = validate_item(item)
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
