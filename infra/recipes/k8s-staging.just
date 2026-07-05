# Recipes for running fair-models example pipelines against the
# AWS EKS staging cluster (hotosm-production-cluster).
# Auth: AWS SSO (run `aws sso login` first).

eks_context := "arn:aws:eks:us-east-1:670261699094:cluster/hotosm-production-cluster"

# Port-forward postgres (fair-stac-db) and MLflow to localhost.
# Run this in a dedicated terminal, it blocks.
[doc('Port-forward postgres (5433) and MLflow (5001) from EKS staging cluster')]
port-forward-staging:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "port-forwarding postgres (5433) and mlflow (5001)..."
    kubectl port-forward svc/fair-stac-db-rw -n postgres 5433:5432 \
        --context="{{ eks_context }}" &
    PF_PG=$!
    kubectl port-forward svc/mlflow -n mlflow 5001:80 \
        --context="{{ eks_context }}" &
    PF_ML=$!
    trap "kill $PF_PG $PF_ML 2>/dev/null || true" EXIT INT TERM
    echo "postgres -> localhost:5433"
    echo "mlflow   -> localhost:5001"
    echo "ctrl-c to stop"
    wait

[doc('Register the k8s-staging ZenML stack (requires source .env.staging first)')]
stack-register-staging:
    #!/usr/bin/env bash
    set -euo pipefail
    envsubst < ../stacks/k8s-staging.yaml > /tmp/k8s-staging-resolved.yaml
    uv run zenml stack set default 2>/dev/null || true
    echo y | uv run zenml stack delete k8s-staging 2>/dev/null || true
    echo y | uv run zenml orchestrator delete k8s_staging_orchestrator 2>/dev/null || true
    echo y | uv run zenml artifact-store delete s3_staging 2>/dev/null || true
    echo y | uv run zenml experiment-tracker delete mlflow_staging 2>/dev/null || true
    echo y | uv run zenml container-registry delete ghcr_staging 2>/dev/null || true
    uv run zenml stack import k8s-staging -f /tmp/k8s-staging-resolved.yaml --ignore-version-mismatch
    uv run zenml stack set k8s-staging
    rm -f /tmp/k8s-staging-resolved.yaml
    echo "k8s-staging stack registered and set as active"

[doc('Run example pipeline(s) against the staging cluster')]
_example-staging *model:
    #!/usr/bin/env bash
    set -euo pipefail
    cd ..
    uv run python examples/run.py $@
