#!/usr/bin/env bash
set -euo pipefail

# Waits for ZenML to become reachable, then prints an access token to stdout.
# Usage: source env-derived.sh first, then:
#   TOKEN=$(scripts/zenml-token.sh)
#   TOKEN=$(scripts/zenml-token.sh --wait)   # with retry loop (for fresh deploys)

ZENML_URL="https://zenml.${FAIR_DOMAIN:?FAIR_DOMAIN required}"
LOGIN_URL="$ZENML_URL/api/v1/login"
LOGIN_DATA="username=${ZENML_ADMIN_USER:?ZENML_ADMIN_USER required}&password=${ZENML_ADMIN_PASSWORD:?ZENML_ADMIN_PASSWORD required}&grant_type=password"
MAX_ATTEMPTS=30
WAIT_SECONDS=10

_try_login() {
  curl -kfsS --connect-timeout 15 --max-time 30 -X POST "$LOGIN_URL" -d "$LOGIN_DATA" 2>/dev/null
}

if [[ "${1:-}" == "--wait" ]]; then
  echo "Waiting for ZenML at $ZENML_URL ..." >&2
  RESPONSE=""
  for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
    if RESPONSE=$(_try_login); then
      break
    fi
    echo "  attempt $attempt/$MAX_ATTEMPTS - not ready yet" >&2
    sleep "$WAIT_SECONDS"
  done
  [[ -n "$RESPONSE" ]] || { echo "ERROR: ZenML not reachable at $ZENML_URL after $((MAX_ATTEMPTS * WAIT_SECONDS))s" >&2; exit 1; }
else
  RESPONSE=$(_try_login) \
    || { echo "ERROR: ZenML login failed at $LOGIN_URL - is the server reachable?" >&2; exit 1; }
fi

TOKEN=$(python3 -c "import sys,json; print(json.loads(sys.argv[1])['access_token'])" "$RESPONSE") \
  || { echo "ERROR: Failed to parse ZenML login response" >&2; exit 1; }

[[ -n "$TOKEN" ]] || { echo "ERROR: Empty ZenML token" >&2; exit 1; }

echo "$TOKEN"
