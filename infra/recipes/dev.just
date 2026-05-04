_up-dev: _ensure-kind _apply-postgres _helmfile-dev _port-forward _seed-dev _stack-register-dev
    @just _urls-dev

_ensure-kind:
    #!/usr/bin/env bash
    set -euo pipefail
    if ! kind get clusters 2>/dev/null | grep -qx {{ kind_cluster }}; then
        kind create cluster --name {{ kind_cluster }} --config kind-config.yaml
    fi

_helmfile-dev:
    helmfile -e dev apply

_port-forward:
    #!/usr/bin/env bash
    set -euo pipefail
    if [[ -f {{ pid_file }} ]] && kill -0 "$(head -1 {{ pid_file }})" 2>/dev/null; then exit 0; fi
    just _kill-port-forward
    mkdir -p .pf-logs
    while read -r svc ports; do
        ( while true; do
            kubectl port-forward -n {{ ns }} "svc/$svc" "$ports" >>.pf-logs/"$svc".log 2>&1 || true
            sleep 2
        done ) & echo $! >> {{ pid_file }}
    done < ports.conf
    while read -r svc ports; do
        for i in $(seq 1 30); do nc -z localhost "${ports%%:*}" 2>/dev/null && break; sleep 1; done
    done < ports.conf

_kill-port-forward:
    #!/usr/bin/env bash
    set -uo pipefail
    if [[ -f {{ pid_file }} ]]; then
        while read -r pid; do
            kill -- "-$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true
        done < {{ pid_file }}
        rm -f {{ pid_file }}
    fi
    rm -rf .pf-logs

_seed-dev:
    #!/usr/bin/env bash
    set -euo pipefail
    AWS_ENDPOINT_URL=http://localhost:9000 \
    AWS_ACCESS_KEY_ID=minioadmin \
    AWS_SECRET_ACCESS_KEY=minioadmin \
    S3_BUCKET=fair-data \
        uv run --with boto3 python scripts/seed_data.py

_stack-register-dev:
    #!/usr/bin/env bash
    set -euo pipefail
    export ZENML_URL="http://localhost:8080"
    export ZENML_ADMIN_USER="default"
    export ZENML_ADMIN_PASSWORD=""
    export ZENML_STORE_URL="$ZENML_URL"
    TOKEN=$(scripts/zenml-token.sh --wait) || { echo "ERROR: failed to fetch ZenML token" >&2; exit 1; }
    export ZENML_STORE_API_TOKEN="$TOKEN"
    uv run zenml stack import dev -f ../stacks/local.yaml --ignore-version-mismatch || true
    uv run zenml stack import k8s -f ../stacks/k8s.yaml --ignore-version-mismatch || true
    uv run zenml stack set k8s

_down-dev: _kill-port-forward

_tear-dev: _kill-port-forward
    -helmfile -e dev destroy
    -kubectl delete -f manifests/postgres.yaml --ignore-not-found
    -kind delete cluster --name {{ kind_cluster }}

_urls-dev:
    @echo "ZenML:  http://localhost:8080"
    @echo "MLflow: http://localhost:5000"
    @echo "STAC:   http://localhost:8082"
    @echo "MinIO:  http://localhost:9001  (console)"

_example-dev:
    #!/usr/bin/env bash
    set -euo pipefail
    cd ..
    uv run zenml stack set k8s
    for ex in segmentation classification detection; do
        AWS_ENDPOINT_URL=http://localhost:9000 \
        FAIR_STAC_API_URL=http://localhost:8082 \
        FAIR_DSN=postgresql://postgres:postgres@localhost:5432/fair_models \
            uv run --group example python "examples/$ex/run.py"
    done
