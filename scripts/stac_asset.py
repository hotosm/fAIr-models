"""Print the href of the first of the given assets the STAC item has.

Usage: stac_asset.py <item.json> <asset_key> [<asset_key> ...]
"""

import json
import sys
from pathlib import Path

assets = json.loads(Path(sys.argv[1]).read_text())["assets"]
keys = sys.argv[2:]
present = [key for key in keys if key in assets]
if not present:
    sys.exit(f"{sys.argv[1]} has none of the assets {keys}")
print(assets[present[0]]["href"])
