#!/usr/bin/env bash
set -euo pipefail

# Fetches a ZenML access token. Reads:
#   ZENML_URL          (e.g. http://localhost:8080 or https://zenml.fair.example.com)
#   ZENML_ADMIN_USER   (default: default)
#   ZENML_ADMIN_PASSWORD (default: empty)
# Optional first arg: --wait  retries for up to 5 min while ZenML comes up.

ZENML_URL="${ZENML_URL:?ZENML_URL required}"
USER="${ZENML_ADMIN_USER:-default}"
PASS="${ZENML_ADMIN_PASSWORD:-}"
LOGIN_URL="$ZENML_URL/api/v1/login"
DATA="username=${USER}&password=${PASS}&grant_type=password"

_try() {
    curl -kfsS --connect-timeout 15 --max-time 30 -X POST "$LOGIN_URL" -d "$DATA" 2>/dev/null
}

if [[ "${1:-}" == "--wait" ]]; then
    for attempt in $(seq 1 30); do
        if RESP=$(_try); then break; fi
        echo "  zenml not ready, retrying ($attempt/30)..." >&2
        sleep 10
    done
    [[ -n "${RESP:-}" ]] || { echo "ERROR: zenml unreachable at $ZENML_URL" >&2; exit 1; }
else
    RESP=$(_try) || { echo "ERROR: zenml login failed at $LOGIN_URL" >&2; exit 1; }
fi

python3 -c "import sys, json; print(json.loads(sys.argv[1])['access_token'])" "$RESP"
