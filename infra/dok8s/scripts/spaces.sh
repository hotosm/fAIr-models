#!/usr/bin/env bash
set -euo pipefail

action=${1:?Usage: spaces.sh <create|seed>}

case "$action" in
  create)
    AWS_ACCESS_KEY_ID="$SPACES_ACCESS_KEY" \
    AWS_SECRET_ACCESS_KEY="$SPACES_SECRET_KEY" \
    aws s3api head-bucket --bucket "$SPACES_BUCKET" --endpoint-url "$SPACES_ENDPOINT" 2>/dev/null || \
    AWS_ACCESS_KEY_ID="$SPACES_ACCESS_KEY" \
    AWS_SECRET_ACCESS_KEY="$SPACES_SECRET_KEY" \
    aws s3 mb "s3://$SPACES_BUCKET" --endpoint-url "$SPACES_ENDPOINT"
    ;;
  seed)
    AWS_ENDPOINT_URL="$SPACES_ENDPOINT" \
    AWS_ACCESS_KEY_ID="$SPACES_ACCESS_KEY" \
    AWS_SECRET_ACCESS_KEY="$SPACES_SECRET_KEY" \
    uv run --with boto3 python "$(dirname "$0")/seed_data.py"
    ;;
esac
