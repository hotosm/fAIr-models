#!/usr/bin/env bash
set -euo pipefail

action=${1:?Usage: database.sh <create|init|write-env|firewall|delete>}
DB_NAME="${CLUSTER_NAME}-pg"

_db_id() {
  local id
  id=$(doctl databases list --format ID,Name --no-header | awk "/$DB_NAME/ {print \$1}")
  [[ -n "$id" ]] || { echo "ERROR: Database '$DB_NAME' not found"; exit 1; }
  echo "$id"
}
_db_uri() { doctl databases connection "$(_db_id)" --format URI --no-header; }

_wait_ready() {
  local uri
  uri=$(_db_uri)
  echo "Waiting for database to accept connections..."
  for attempt in $(seq 1 30); do
    if psql "$uri" -c "SELECT 1" >/dev/null 2>&1; then
      echo "Database is ready."
      return 0
    fi
    echo "  attempt $attempt/30"
    sleep 10
  done
  echo "ERROR: Database not reachable after 300s"
  exit 1
}

case "$action" in
  create)
    if doctl databases list --format Name --no-header 2>/dev/null | grep -qx "$DB_NAME"; then
      echo "Database $DB_NAME exists."
    else
      doctl databases create "$DB_NAME" \
        --engine pg --version 17 \
        --region "$DO_REGION" \
        --size db-s-2vcpu-4gb --num-nodes 1 \
        --wait
    fi
    _wait_ready
    ;;
  init)
    SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
    BASE_URI=$(_db_uri)
    psql "$BASE_URI" -f "$SCRIPT_DIR/init.sql" 2>/dev/null
    FAIR_URI=$(echo "$BASE_URI" | sed 's|/defaultdb|/fair_models|')
    psql "$FAIR_URI" -c "CREATE EXTENSION IF NOT EXISTS postgis; CREATE EXTENSION IF NOT EXISTS btree_gist;"
    echo "Databases initialized."
    echo "Running pgstac migration..."
    uv run --extra k8s pypgstac migrate --dsn "$FAIR_URI"
    echo "pgstac migration complete."
    ;;
  write-env)
    SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
    DB_ID=$(_db_id)
    PG_HOST=$(doctl databases connection "$DB_ID" --format Host --no-header)
    PG_PORT=$(doctl databases connection "$DB_ID" --format Port --no-header)
    PG_USER=$(doctl databases connection "$DB_ID" --format User --no-header)
    PG_PASSWORD=$(doctl databases connection "$DB_ID" --format Password --no-header)
    ENV_FILE="$SCRIPT_DIR/.env"
    [[ -f "$ENV_FILE" ]] || cp "$SCRIPT_DIR/env.example" "$ENV_FILE"
    sed -i.bak \
      -e "s|^PG_HOST=.*|PG_HOST=$PG_HOST|" \
      -e "s|^PG_PORT=.*|PG_PORT=$PG_PORT|" \
      -e "s|^PG_USER=.*|PG_USER=$PG_USER|" \
      -e "s|^PG_PASSWORD=.*|PG_PASSWORD=$PG_PASSWORD|" "$ENV_FILE"
    rm -f "$ENV_FILE.bak"
    echo "PG credentials written to .env"
    ;;
  firewall)
    DB_ID=$(_db_id)
    DOKS_UUID=$(doctl kubernetes cluster get "$CLUSTER_NAME" --format ID --no-header)
    MY_IP=$(curl -fsS https://api.ipify.org)
    EXISTING=$(doctl databases firewalls list "$DB_ID" --format Type,Value --no-header 2>/dev/null || true)
    if echo "$EXISTING" | grep -q "$DOKS_UUID"; then
      echo "Firewall: DOKS cluster already trusted."
    else
      doctl databases firewalls append "$DB_ID" --rule "k8s:$DOKS_UUID"
      echo "Firewall: added DOKS cluster $DOKS_UUID."
    fi
    if echo "$EXISTING" | grep -q "$MY_IP"; then
      echo "Firewall: local IP $MY_IP already trusted."
    else
      doctl databases firewalls append "$DB_ID" --rule "ip_addr:$MY_IP"
      echo "Firewall: added local IP $MY_IP."
    fi
    ;;
  delete)
    DB_ID=$(_db_id)
    [[ -n "$DB_ID" ]] && doctl databases delete "$DB_ID" --force
    ;;
esac
