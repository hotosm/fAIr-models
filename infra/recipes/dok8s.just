_up-dok8s: _tofu-apply _save-kubeconfig _apply-postgres _helmfile-dok8s _post-deploy-dok8s _seed-dok8s _stack-register-dok8s _write-env-dok8s
    @just _urls-dok8s

_tofu-apply:
    cd dok8s && tofu init -upgrade && tofu apply -auto-approve

_save-kubeconfig:
    #!/usr/bin/env bash
    set -euo pipefail
    cd dok8s
    CTX="$(tofu output -raw doks_context)"
    doctl kubernetes cluster kubeconfig save "${CTX#do-*-}"

_helmfile-dok8s:
    #!/usr/bin/env bash
    set -euo pipefail
    set -a; source dok8s/.env.helmfile; set +a
    helmfile -e dok8s apply

_post-deploy-dok8s:
    #!/usr/bin/env bash
    set -euo pipefail
    set -a; source dok8s/.env.helmfile; set +a
    kubectl apply -f manifests/knative-serving.yaml
    kubectl apply -f manifests/s3-credentials.yaml
    just _wait-lb
    just _set-dns
    kubectl wait --for=condition=Available deployment/cert-manager-webhook -n cert-manager --timeout=180s
    envsubst < manifests/cluster-issuer.yaml.gotmpl | kubectl apply -f -
    envsubst < manifests/ingress.yaml.gotmpl | kubectl apply -f -

_wait-lb:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Waiting for ingress LoadBalancer IP..."
    for i in $(seq 1 60); do
        IP=$(kubectl get svc -n ingress-nginx ingress-nginx-controller \
            -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || true)
        [[ -n "$IP" ]] && { echo "$IP" > dok8s/.lb-ip; echo "  $IP"; exit 0; }
        sleep 5
    done
    echo "ERROR: LoadBalancer IP not assigned after 5 minutes" >&2; exit 1

_set-dns:
    #!/usr/bin/env bash
    set -euo pipefail
    set -a; source dok8s/.env.helmfile; set +a
    IP=$(cat dok8s/.lb-ip)
    ZONE=$(echo "$FAIR_DOMAIN" | awk -F. '{print $(NF-1)"."$NF}')
    NAME=$(echo "$FAIR_DOMAIN" | sed "s/\.$ZONE\$//")
    EXISTING=$(doctl compute domain records list "$ZONE" --format ID,Type,Name --no-header 2>/dev/null \
        | awk -v t=A -v n="*.$NAME" '$2==t && $3==n {print $1}')
    if [[ -n "$EXISTING" ]]; then
        for id in $EXISTING; do
            doctl compute domain records update "$ZONE" --record-id "$id" --record-data "$IP" >/dev/null
        done
    else
        doctl compute domain records create "$ZONE" --record-type A --record-name "*.$NAME" --record-data "$IP" --record-ttl 300 >/dev/null
    fi
    echo "DNS *.$FAIR_DOMAIN -> $IP"

_seed-dok8s:
    #!/usr/bin/env bash
    set -euo pipefail
    kubectl port-forward -n {{ ns }} svc/minio 9001:9000 >/dev/null 2>&1 &
    PF_PID=$!
    trap "kill $PF_PID 2>/dev/null || true" EXIT
    for i in $(seq 1 20); do nc -z localhost 9001 2>/dev/null && break; sleep 1; done
    AWS_ENDPOINT_URL=http://localhost:9001 \
    AWS_ACCESS_KEY_ID=minioadmin \
    AWS_SECRET_ACCESS_KEY=minioadmin \
    S3_BUCKET=fair-data \
        uv run --with boto3 python scripts/seed_data.py

_stack-register-dok8s:
    #!/usr/bin/env bash
    set -euo pipefail
    set -a; source dok8s/.env.helmfile; set +a
    export ZENML_URL="https://zenml.${FAIR_DOMAIN:?}"
    TOKEN=$(scripts/zenml-token.sh --wait) || { echo "ERROR: failed to fetch ZenML token" >&2; exit 1; }
    export ZENML_STORE_URL="$ZENML_URL"
    export ZENML_STORE_API_TOKEN="$TOKEN"
    export ZENML_STORE_VERIFY_SSL="true"
    export DOKS_CONTEXT="$(cd dok8s && tofu output -raw doks_context)"
    envsubst < ../stacks/dok8s.yaml > dok8s/.dok8s-stack-resolved.yaml
    uv run zenml stack set default 2>/dev/null || true
    echo y | uv run zenml stack delete dok8s 2>/dev/null || true
    echo y | uv run zenml orchestrator delete doks_orchestrator 2>/dev/null || true
    echo y | uv run zenml artifact-store delete s3_artifacts 2>/dev/null || true
    echo y | uv run zenml experiment-tracker delete mlflow_tracker 2>/dev/null || true
    echo y | uv run zenml container-registry delete ghcr 2>/dev/null || true
    uv run zenml stack import dok8s -f dok8s/.dok8s-stack-resolved.yaml --ignore-version-mismatch
    uv run zenml stack set dok8s
    rm -f dok8s/.dok8s-stack-resolved.yaml

_write-env-dok8s:
    #!/usr/bin/env bash
    set -euo pipefail
    set -a; source dok8s/.env.helmfile; set +a
    DOKS_CONTEXT="$(cd dok8s && tofu output -raw doks_context)"
    {
        echo "export ZENML_STORE_URL=\"https://zenml.${FAIR_DOMAIN}\""
        echo "export ZENML_STORE_VERIFY_SSL=\"true\""
        echo "export FAIR_DSN=\"postgresql://postgres:postgres@localhost:15432/fair_models\""
        echo "export FAIR_STAC_API_URL=\"https://stac.${FAIR_DOMAIN}/stac\""
        echo "export FAIR_LABEL_DOMAIN=\"${FAIR_DOMAIN}\""
        echo "export FAIR_DOMAIN=\"${FAIR_DOMAIN}\""
        echo "export FAIR_USER_ID=\"krschap\""
        echo "export FAIR_UPLOAD_ARTIFACTS=\"true\""
        echo "export FAIR_FORCE_CPU=\"1\""
        echo "export AWS_ENDPOINT_URL=\"https://s3.${FAIR_DOMAIN}\""
        echo "export FAIR_S3_PUBLIC_URL=\"https://s3.${FAIR_DOMAIN}\""
        echo "export AWS_ACCESS_KEY_ID=\"minioadmin\""
        echo "export AWS_SECRET_ACCESS_KEY=\"minioadmin\""
        echo "export AWS_REQUEST_CHECKSUM_CALCULATION=\"when_required\""
        echo "export AWS_RESPONSE_CHECKSUM_VALIDATION=\"when_required\""
        echo "export DOKS_CONTEXT=\"${DOKS_CONTEXT}\""
    } > .env.dok8s
    echo "Wrote .env.dok8s. Use: source infra/.env.dok8s before running examples locally."

_down-dok8s:
    -helmfile -e dok8s destroy

_tear-dok8s:
    #!/usr/bin/env bash
    set -euo pipefail
    -helmfile -e dok8s destroy 2>/dev/null || true
    cd dok8s && tofu destroy -auto-approve

_urls-dok8s:
    #!/usr/bin/env bash
    set -euo pipefail
    cd dok8s
    set -a; source .env.helmfile 2>/dev/null || true; set +a
    DOMAIN="${FAIR_DOMAIN:-$(tofu output -raw fair_domain 2>/dev/null || echo unknown)}"
    echo "ZenML:   https://zenml.${DOMAIN}"
    echo "MLflow:  https://mlflow.${DOMAIN}"
    echo "STAC:    https://stac.${DOMAIN}/stac"
    echo "Predict: https://predict.${DOMAIN}"

_example-dok8s:
    #!/usr/bin/env bash
    set -euo pipefail
    [[ -f .env.dok8s ]] || { echo "ERROR: .env.dok8s missing - run 'just up dok8s' first"; exit 1; }
    source .env.dok8s
    set -a; source dok8s/.env.helmfile; set +a
    PG_PORT=15432
    kubectl port-forward -n {{ ns }} svc/postgres ${PG_PORT}:5432 >/tmp/pf-pg.log 2>&1 &
    PG_PID=$!
    trap "kill $PG_PID 2>/dev/null || true" EXIT
    for i in $(seq 1 20); do nc -z localhost $PG_PORT 2>/dev/null && break; sleep 1; done
    export FAIR_DSN="postgresql://postgres:postgres@localhost:${PG_PORT}/fair_models"
    export ZENML_URL="$ZENML_STORE_URL"
    TOKEN=$(scripts/zenml-token.sh --wait) || { echo "ERROR: failed to fetch ZenML token" >&2; exit 1; }
    export ZENML_STORE_API_TOKEN="$TOKEN"
    cd ..
    for ex in segmentation classification detection; do
        echo "=== $ex ==="
        uv run --group example python "examples/$ex/run.py"
    done
