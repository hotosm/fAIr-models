#!/usr/bin/env bash
set -euo pipefail

export ZENML_URL="http://localhost:8080"
export ZENML_ADMIN_USER="default"
export ZENML_ADMIN_PASSWORD=""
export ZENML_STORE_URL="$ZENML_URL"

TOKEN=$(infra/scripts/zenml-token.sh --wait)
export ZENML_STORE_API_TOKEN="$TOKEN"

uv run zenml stack set default 2>/dev/null || true
uv run zenml stack delete compose -y -r 2>/dev/null || true
uv run zenml stack import compose -f stacks/compose.yaml --ignore-version-mismatch
uv run zenml stack set compose
