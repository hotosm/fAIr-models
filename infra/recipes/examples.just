# Recipes for running the fair-models example pipelines from a laptop against a
# deployed cluster. None of this is needed for the deployed fair-backend.
# Set FAIR_DOMAIN before use; everything else is derived from it.

# Render the env vars needed for examples and pipe them to stdout. Consumers do:
#   eval "$(just _env)"
_env:
    #!/usr/bin/env bash
    set -euo pipefail
    DOMAIN="${FAIR_DOMAIN:?FAIR_DOMAIN must be set to the deployed apex domain}"
    cat <<EOF
    export ZENML_STORE_URL="https://zenml.${DOMAIN}"
    export ZENML_STORE_VERIFY_SSL="true"
    export FAIR_STAC_API_URL="https://stac.${DOMAIN}/stac"
    export FAIR_LABEL_DOMAIN="${DOMAIN}"
    export FAIR_PREDICT_DOMAIN="${DOMAIN}"
    export FAIR_DOMAIN="${DOMAIN}"
    export FAIR_USER_ID="${FAIR_USER_ID:-anonymous}"
    export FAIR_UPLOAD_ARTIFACTS="true"
    export FAIR_FORCE_CPU="1"
    export AWS_ENDPOINT_URL="https://s3.${DOMAIN}"
    export FAIR_S3_PUBLIC_URL="https://s3.${DOMAIN}"
    export AWS_ACCESS_KEY_ID="minioadmin"
    export AWS_SECRET_ACCESS_KEY="minioadmin"
    export AWS_REQUEST_CHECKSUM_CALCULATION="when_required"
    export AWS_RESPONSE_CHECKSUM_VALIDATION="when_required"
    export KUBE_CONTEXT="${KUBE_CONTEXT:-$(kubectl config current-context)}"
    EOF

[doc('Run all 3 example pipelines (segmentation, classification, detection) against the cluster')]
example: _example

[doc('Smoke-test the live KNative predict services')]
predict:
    #!/usr/bin/env bash
    set -euo pipefail
    DOMAIN="${FAIR_DOMAIN:?FAIR_DOMAIN must be set to the deployed apex domain}"
    for svc in unet-segmentation resnet18-classification yolo11n-detection; do
        url="https://${svc}.predict.${DOMAIN}/health"
        printf "%-30s " "$svc"
        if curl -sfk --max-time 30 "$url" >/dev/null; then echo "ok"; else echo "fail ($url)"; fi
    done

[doc('Seed sample data into MinIO (the fair-data bucket the example pipelines read from)')]
seed:
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

[doc('Register the cluster stack with the local ZenML CLI (so pipelines run against the cluster)')]
stack-register:
    #!/usr/bin/env bash
    set -euo pipefail
    eval "$(just _env)"
    TOKEN=$(scripts/zenml-token.sh --wait) || { echo "ERROR: failed to fetch ZenML token" >&2; exit 1; }
    export ZENML_STORE_API_TOKEN="$TOKEN"
    RESOLVED=".cluster-stack-resolved.yaml"
    trap "rm -f $RESOLVED" EXIT
    envsubst < ../stacks/cluster.yaml > "$RESOLVED"
    uv run zenml stack set default 2>/dev/null || true
    echo y | uv run zenml stack delete cluster 2>/dev/null || true
    echo y | uv run zenml orchestrator delete cluster_orchestrator 2>/dev/null || true
    echo y | uv run zenml artifact-store delete s3_artifacts 2>/dev/null || true
    echo y | uv run zenml experiment-tracker delete mlflow_tracker 2>/dev/null || true
    echo y | uv run zenml container-registry delete ghcr 2>/dev/null || true
    uv run zenml stack import cluster -f "$RESOLVED" --ignore-version-mismatch
    uv run zenml stack set cluster

_example:
    #!/usr/bin/env bash
    set -euo pipefail
    eval "$(just _env)"
    PG_PORT=15432
    kubectl port-forward -n {{ ns }} svc/postgres ${PG_PORT}:5432 >/tmp/pf-pg.log 2>&1 &
    PG_PID=$!
    trap "kill $PG_PID 2>/dev/null || true" EXIT
    for i in $(seq 1 20); do nc -z localhost $PG_PORT 2>/dev/null && break; sleep 1; done
    export FAIR_DSN="postgresql://postgres:postgres@localhost:${PG_PORT}/fair_models"
    TOKEN=$(scripts/zenml-token.sh --wait) || { echo "ERROR: failed to fetch ZenML token" >&2; exit 1; }
    export ZENML_STORE_API_TOKEN="$TOKEN"
    cd ..
    for ex in segmentation classification detection; do
        echo "=== $ex ==="
        uv run --group example python "examples/$ex/run.py"
    done
