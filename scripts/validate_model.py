"""CI entrypoint: validate all model contributions under models/.

Exit code 0 if all models valid, 1 otherwise.
Usage: python scripts/validate_model.py [models/specific_model ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

from fair.utils.model_validator import validate_model

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def main(paths: list[str] | None = None) -> int:
    if paths:
        dirs = [Path(p) for p in paths]
    else:
        dirs = sorted(d for d in MODELS_DIR.iterdir() if d.is_dir() and not d.name.startswith("_"))

    all_errors: list[str] = []
    for model_dir in dirs:
        all_errors.extend(validate_model(model_dir))

    if all_errors:
        for err in all_errors:
            print(f"ERROR: {err}", file=sys.stderr)
        return 1

    print(f"OK: {len(dirs)} model(s) validated")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or None))
