"""Upload sample data to DO Spaces."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import boto3


def main() -> None:
    root = Path(__file__).resolve().parents[3] / "data" / "sample"
    if not root.exists():
        sys.exit(f"Sample data not found at {root}")

    endpoint = os.environ["AWS_ENDPOINT_URL"]
    bucket = os.environ.get("SPACES_BUCKET", "hotosm-fair")

    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    )

    files = [f for f in root.rglob("*") if f.is_file()]
    print(f"Uploading {len(files)} files to {bucket}/data/sample/")
    for f in files:
        key = f"data/sample/{f.relative_to(root)}"
        s3.upload_file(str(f), bucket, key)
    print("Done")


if __name__ == "__main__":
    main()
