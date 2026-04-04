#!/usr/bin/env bash
set -euo pipefail

action=${1:?Usage: cluster.sh <create|delete|kubeconfig>}

case "$action" in
  create)
    if doctl kubernetes cluster get "$CLUSTER_NAME" >/dev/null 2>&1; then
      echo "Cluster $CLUSTER_NAME exists."
    else
      doctl kubernetes cluster create "$CLUSTER_NAME" \
        --region "$DO_REGION" \
        --node-pool "name=infra;size=${INFRA_NODE_SIZE};count=1;label=${FAIR_LABEL_DOMAIN}/role=infra" \
        --wait
      doctl kubernetes cluster node-pool create "$CLUSTER_NAME" \
        --name ml \
        --size "${ML_NODE_SIZE:-s-4vcpu-8gb}" \
        --count 0 \
        --auto-scale \
        --min-nodes 0 \
        --max-nodes 1 \
        --label "${FAIR_LABEL_DOMAIN}/training=true" \
        --label "${FAIR_LABEL_DOMAIN}/inference=true"
    fi
    doctl kubernetes cluster kubeconfig save "$CLUSTER_NAME"
    ;;
  delete)
    doctl kubernetes cluster delete "$CLUSTER_NAME" --force --dangerous
    ;;
  kubeconfig)
    doctl kubernetes cluster kubeconfig save "$CLUSTER_NAME"
    ;;
esac
