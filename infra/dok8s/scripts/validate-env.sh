#!/usr/bin/env bash
set -euo pipefail

required_vars=(
  DO_TOKEN
  FAIR_DOMAIN
  MLFLOW_ADMIN_USER
  MLFLOW_ADMIN_PASSWORD
  ZENML_ADMIN_USER
  ZENML_ADMIN_PASSWORD
  LETSENCRYPT_EMAIL
  SPACES_BUCKET
  SPACES_ACCESS_KEY
  SPACES_SECRET_KEY
)

missing=()
for var in "${required_vars[@]}"; do
  if [[ -z "${!var:-}" ]]; then
    missing+=("$var")
  fi
done

if (( ${#missing[@]} > 0 )); then
  echo "ERROR: Missing required environment variables:"
  printf '  %s\n' "${missing[@]}"
  echo "Copy env.example to .env and fill in the values."
  exit 1
fi
