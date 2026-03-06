"""Upload sample data to MinIO for dev cluster."""

from __future__ import annotations

import sys
from pathlib import Path

from minio import Minio  # ty: ignore[unresolved-import]  # optional dep (k8s infra)


def main() -> None:
    root = Path(__file__).resolve().parents[3] / "data" / "sample"
    if not root.exists():
        sys.exit(f"Sample data not found at {root}")

    client = Minio("localhost:9000", "minioadmin", "minioadmin", secure=False)
    files = [f for f in root.rglob("*") if f.is_file()]
    print(f"Uploading {len(files)} files to fair-data/sample/")
    for f in files:
        client.fput_object("fair-data", f"sample/{f.relative_to(root)}", str(f))
    print("Done")


if __name__ == "__main__":
    main()
