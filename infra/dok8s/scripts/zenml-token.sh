#!/usr/bin/env bash
set -euo pipefail

# Waits for ZenML to become reachable, then prints an access token to stdout.
# Usage: source env-derived.sh first, then:
#   TOKEN=$(scripts/zenml-token.sh)
#   TOKEN=$(scripts/zenml-token.sh --wait)   # with retry loop (for fresh deploys)

ZENML_URL="https://zenml.${FAIR_DOMAIN:?FAIR_DOMAIN required}"
LOGIN_URL="$ZENML_URL/api/v1/login"
INFO_URL="$ZENML_URL/api/v1/info"
MAX_ATTEMPTS=30
WAIT_SECONDS=10

if [[ "${1:-}" == "--wait" ]]; then
  echo "Waiting for ZenML at $ZENML_URL ..." >&2
  READY=false
  for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
    if curl -kfsS -o /dev/null "$INFO_URL" 2>/dev/null; then
      READY=true
      break
    fi
    echo "  attempt $attempt/$MAX_ATTEMPTS - not ready yet" >&2
    sleep "$WAIT_SECONDS"
  done
  if [[ "$READY" != "true" ]]; then
    echo "ERROR: ZenML not reachable at $ZENML_URL after $((MAX_ATTEMPTS * WAIT_SECONDS))s" >&2
    exit 1
  fi
fi

RESPONSE=$(curl -kfsS -X POST "$LOGIN_URL" \
  -d "username=${ZENML_ADMIN_USER:?ZENML_ADMIN_USER required}&password=${ZENML_ADMIN_PASSWORD:?ZENML_ADMIN_PASSWORD required}&grant_type=password" 2>&1) \
  || { echo "ERROR: ZenML login failed at $LOGIN_URL - is the server reachable?" >&2; exit 1; }

TOKEN=$(python3 -c "import sys,json; print(json.loads(sys.argv[1])['access_token'])" "$RESPONSE") \
  || { echo "ERROR: Failed to parse ZenML login response" >&2; exit 1; }

[[ -n "$TOKEN" ]] || { echo "ERROR: Empty ZenML token" >&2; exit 1; }

echo "$TOKEN"
