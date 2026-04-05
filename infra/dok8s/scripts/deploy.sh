#!/usr/bin/env bash
set -euo pipefail

action=${1:?Usage: deploy.sh <helm|stac-ingress|dns|cert-issuer|register-stack>}
NS="${NS:-fair}"
STACK_NAME="dok8s"

case "$action" in
  helm)
    kubectl create ns "$NS" --dry-run=client -o yaml | kubectl apply -f -
    helmfile -l tier=ingress apply
    echo "Waiting for load balancer IP..."
    for _ in $(seq 1 60); do
      LB_IP=$(kubectl get svc -n ingress-nginx ingress-nginx-controller \
        -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null)
      [[ -n "$LB_IP" ]] && { echo "LB IP: $LB_IP"; break; }
      sleep 5
    done
    [[ -n "$LB_IP" ]] || { echo "ERROR: LB IP not assigned after 300s"; exit 1; }
    helmfile -l tier=app apply
    ;;
  stac-ingress)
    SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
    sed "s/{{STAC_HOST}}/stac.${FAIR_DOMAIN}/" "$SCRIPT_DIR/stac/ingress.yaml" | kubectl apply -f -
    ;;
  dns)
    LB_IP=$(kubectl get svc -n ingress-nginx ingress-nginx-controller \
      -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
    ZONE=$(echo "$FAIR_DOMAIN" | awk -F. '{print $(NF-1)"."$NF}')
    PREFIX=$(echo "$FAIR_DOMAIN" | sed 's/\.[^.]*\.[^.]*$//')
    REC_ID=$(doctl compute domain records list "$ZONE" --format ID,Type,Name --no-header 2>/dev/null | \
      awk -v p="*.$PREFIX" '$2=="A" && $3==p {print $1; exit}')
    if [[ -n "$REC_ID" ]]; then
      doctl compute domain records update "$ZONE" \
        --record-id "$REC_ID" --record-type A --record-name "*.$PREFIX" --record-data "$LB_IP" --record-ttl 300
      echo "DNS updated: *.$FAIR_DOMAIN -> $LB_IP"
    else
      doctl compute domain records create "$ZONE" \
        --record-type A --record-name "*.$PREFIX" --record-data "$LB_IP" --record-ttl 300
      echo "DNS created: *.$FAIR_DOMAIN -> $LB_IP"
    fi
    ;;
  cert-issuer)
    kubectl apply -f - <<EOF
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: ${LETSENCRYPT_EMAIL:?LETSENCRYPT_EMAIL required}
    privateKeySecretRef:
      name: letsencrypt-account-key
    solvers:
      - http01:
          ingress:
            ingressClassName: nginx
EOF
    echo "ClusterIssuer 'letsencrypt' created"
    ;;
  register-stack)
    SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
    source "$SCRIPT_DIR/scripts/env-derived.sh"
    echo "Waiting for ZenML to become reachable..."
    for attempt in $(seq 1 30); do
      if curl -kfsS -o /dev/null "https://zenml.${FAIR_DOMAIN}/api/v1/info" 2>/dev/null; then
        break
      fi
      echo "  attempt $attempt/30 - not ready yet"
      sleep 10
    done
    TOKEN=$(curl -kfsS -X POST "https://zenml.${FAIR_DOMAIN}/api/v1/login" \
      -d "username=${ZENML_ADMIN_USER}&password=${ZENML_ADMIN_PASSWORD}&grant_type=password" | \
      python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])")
    [[ -n "$TOKEN" ]] || { echo "ERROR: Failed to obtain ZenML token"; exit 1; }
    export ZENML_STORE_URL="https://zenml.${FAIR_DOMAIN}"
    export ZENML_STORE_API_TOKEN="$TOKEN"
    export ZENML_STORE_VERIFY_SSL=false
    RESOLVED="$SCRIPT_DIR/.${STACK_NAME}-stack-resolved.yaml"
    envsubst < "$SCRIPT_DIR/../../stacks/${STACK_NAME}.yaml" > "$RESOLVED"
    uv run zenml stack import "$STACK_NAME" -f "$RESOLVED" --ignore-version-mismatch 2>/dev/null || true
    uv run zenml stack set "$STACK_NAME"
    rm -f "$RESOLVED"
    ;;
esac
