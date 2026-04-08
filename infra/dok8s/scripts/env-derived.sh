#!/usr/bin/env bash
# Computed from .env values; sourced by justfile recipes and deploy.sh
export FAIR_LABEL_DOMAIN="$FAIR_DOMAIN"
export DOKS_CONTEXT="do-${DO_REGION}-${CLUSTER_NAME}"
export SPACES_ENDPOINT="https://${DO_REGION}.digitaloceanspaces.com"
ENCODED_PG_PASSWORD=$(python3 -c "import urllib.parse,os; print(urllib.parse.quote(os.environ['PG_PASSWORD'], safe=''))")
export PGSTAC_DSN="postgresql://${PG_USER}:${ENCODED_PG_PASSWORD}@${PG_HOST}:${PG_PORT}/fair_models?sslmode=require"
export STAC_API_URL="https://stac.${FAIR_DOMAIN}/stac"
