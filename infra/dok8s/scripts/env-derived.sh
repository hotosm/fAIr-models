#!/usr/bin/env bash
# Computed from .env values; sourced by justfile recipes and deploy.sh
export FAIR_LABEL_DOMAIN="$FAIR_DOMAIN"
export DOKS_CONTEXT="do-${DO_REGION}-${CLUSTER_NAME}"
export SPACES_ENDPOINT="https://${DO_REGION}.digitaloceanspaces.com"
export PGSTAC_DSN="postgresql://${PG_USER}:${PG_PASSWORD}@${PG_HOST}:${PG_PORT}/fair_models?sslmode=require"
export STAC_API_URL="https://stac.${FAIR_DOMAIN}/stac"
