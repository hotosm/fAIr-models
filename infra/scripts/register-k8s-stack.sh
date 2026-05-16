#!/usr/bin/env bash
set -euo pipefail

export ZENML_URL="http://localhost:8080"
export ZENML_ADMIN_USER="default"
export ZENML_ADMIN_PASSWORD=""
export ZENML_STORE_URL="$ZENML_URL"

uv run zenml stack set default 2>/dev/null || true
uv run zenml stack delete k8s -y -r 2>/dev/null || true

TOKEN=$(infra/scripts/zenml-token.sh --wait)
export ZENML_STORE_API_TOKEN="$TOKEN"

uv run zenml stack import k8s -f stacks/k8s.yaml --ignore-version-mismatch
uv run zenml stack set k8s
