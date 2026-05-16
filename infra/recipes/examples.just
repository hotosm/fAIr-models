# Recipes for running the fair-models example pipelines locally against the
# deployed dok8s cluster. None of this is needed for the deployed fair-backend.
# Values are pulled from `tofu output` directly 

# Render the env vars needed for examples and pipe them to stdout. Consumers do:
#   eval "$(just _env-dok8s)"
_env-dok8s:
    #!/usr/bin/env bash
    set -euo pipefail
    DOMAIN="$(cd dok8s && tofu output -raw fair_domain)"
    cat <<EOF
    export ZENML_STORE_URL="https://zenml.${DOMAIN}"
    export ZENML_STORE_VERIFY_SSL="true"
    export FAIR_STAC_API_URL="https://stac.${DOMAIN}/stac"
    export FAIR_LABEL_DOMAIN="${DOMAIN}"
    export FAIR_DOMAIN="${DOMAIN}"
    export FAIR_USER_ID="krschap"
    export FAIR_UPLOAD_ARTIFACTS="true"
    export FAIR_FORCE_CPU="1"
    export AWS_ENDPOINT_URL="https://s3.${DOMAIN}"
    export FAIR_S3_PUBLIC_URL="https://s3.${DOMAIN}"
    export AWS_ACCESS_KEY_ID="minioadmin"
    export AWS_SECRET_ACCESS_KEY="minioadmin"
    export AWS_REQUEST_CHECKSUM_CALCULATION="when_required"
    export AWS_RESPONSE_CHECKSUM_VALIDATION="when_required"
    export DOKS_CONTEXT="$(cd dok8s && tofu output -raw doks_context)"
    EOF

[doc('Run all 3 example pipelines (segmentation, classification, detection) against dok8s')]
example: _example-dok8s

[doc('Smoke-test live KNative predict services on dok8s')]
predict:
    #!/usr/bin/env bash
    set -euo pipefail
    DOMAIN="$(cd dok8s && tofu output -raw fair_domain)"
    for svc in unet-segmentation resnet18-classification yolo11n-detection; do
        url="https://predict.${DOMAIN}/${svc}/health"
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

[doc('Register the dok8s stack with the local ZenML CLI (so pipelines run against the cluster)')]
stack-register:
    #!/usr/bin/env bash
    set -euo pipefail
    eval "$(just _env-dok8s)"
    TOKEN=$(scripts/zenml-token.sh --wait) || { echo "ERROR: failed to fetch ZenML token" >&2; exit 1; }
    export ZENML_STORE_API_TOKEN="$TOKEN"
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

_example-dok8s:
    #!/usr/bin/env bash
    set -euo pipefail
    eval "$(just _env-dok8s)"
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
