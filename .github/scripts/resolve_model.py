"""Resolve which model to test in CI.

Reads:
    INPUT_MODEL   model dir under models/ (workflow_dispatch input; takes precedence)
    BASE_SHA      PR base sha (pull_request_target)
    HEAD_SHA      PR head sha (pull_request_target)

Writes `name=<model>` to $GITHUB_OUTPUT and prints the name. Exits 1 with a
clear message if no model can be resolved or required files are missing.
"""

import os
import subprocess
import sys
from pathlib import Path

REQUIRED = ("Dockerfile", "pipeline.py", "stac-item.json")


def changed_model_dirs() -> list[str]:
    base, head = os.environ.get("BASE_SHA"), os.environ.get("HEAD_SHA")
    if not (base and head):
        return []
    out = subprocess.run(
        ["git", "diff", "--name-only", base, head],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    return sorted({Path(p).parts[1] for p in out if p.startswith("models/") and len(Path(p).parts) > 1})


def main() -> int:
    name = os.environ.get("INPUT_MODEL", "").strip()
    if not name:
        dirs = changed_model_dirs()
        if not dirs:
            sys.exit("no model dir changed in this PR")
        if len(dirs) > 1:
            sys.exit(
                "multiple model dirs changed; trigger via workflow_dispatch with explicit model:\n  "
                + "\n  ".join(dirs)
            )
        name = dirs[0]

    missing = [f for f in REQUIRED if not Path(f"models/{name}/{f}").exists()]
    if missing:
        sys.exit(f"models/{name} missing: {', '.join(missing)}")

    print(name)
    if out := os.environ.get("GITHUB_OUTPUT"):
        with Path(out).open("a") as f:
            f.write(f"name={name}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
