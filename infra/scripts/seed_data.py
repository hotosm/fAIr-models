"""Upload sample data to the configured S3-compatible bucket.

Reads endpoint and credentials from environment:
  AWS_ENDPOINT_URL, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, S3_BUCKET (default: fair-data)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import boto3
from botocore.exceptions import ClientError


def main() -> None:
    root = Path(__file__).resolve().parents[2] / "data" / "sample"
    if not root.exists():
        sys.exit(f"Sample data not found at {root}")

    endpoint = os.environ["AWS_ENDPOINT_URL"]
    bucket = os.environ.get("S3_BUCKET", "fair-data")
    prefix = os.environ.get("S3_PREFIX", "sample")

    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    )

    try:
        s3.head_bucket(Bucket=bucket)
    except ClientError:
        s3.create_bucket(Bucket=bucket)

    acl = os.environ.get("FAIR_S3_UPLOAD_ACL", "").strip()
    extra_args = {"ACL": acl} if acl else None

    files = [f for f in root.rglob("*") if f.is_file()]
    print(f"Uploading {len(files)} files to s3://{bucket}/{prefix}/")
    for f in files:
        key = f"{prefix}/{f.relative_to(root)}"
        s3.upload_file(str(f), bucket, key, ExtraArgs=extra_args)
    print("Done")


if __name__ == "__main__":
    main()
