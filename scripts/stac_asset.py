"""Print a STAC item asset href. Usage: stac_asset.py <item.json> <asset_key>."""

import json
import sys
from pathlib import Path

print(json.loads(Path(sys.argv[1]).read_text())["assets"][sys.argv[2]]["href"])
