_up-dev: _ensure-kind _apply-postgres _helmfile-dev _port-forward-daemon _wait-services _seed-minio
    @just _urls-dev

_ensure-kind:
    #!/usr/bin/env bash
    set -euo pipefail
    if ! kind get clusters 2>/dev/null | grep -qx {{ kind_cluster }}; then
        kind create cluster --name {{ kind_cluster }} --config kind-config.yaml
    fi

_helmfile-dev:
    helmfile -e dev apply

_port-forward-daemon:
    #!/usr/bin/env bash
    set -euo pipefail
    docker rm -f {{ pf_container }} 2>/dev/null || true
    PORTS_CONFIG=$(cat ports.conf)
    HOST_KUBE="${HOST_HOME:-$HOME}/.kube"
    docker run -d --name {{ pf_container }} \
        --network host \
        -v "$HOST_KUBE":/root/.kube \
        -e PORTS_CONFIG="$PORTS_CONFIG" \
        --restart unless-stopped \
        {{ cli_image }} \
        bash -c '
            trap "kill 0" EXIT
            while read -r svc ports; do
                [ -z "$svc" ] && continue
                ( while true; do
                    kubectl port-forward -n {{ ns }} "svc/$svc" "$ports" >/dev/null 2>&1 || true
                    sleep 2
                done ) &
            done <<< "$PORTS_CONFIG"
            wait
        '

_wait-services:
    #!/usr/bin/env bash
    set -euo pipefail
    while read -r svc ports; do
        port="${ports%%:*}"
        for i in $(seq 1 60); do
            nc -z localhost "$port" 2>/dev/null && break
            sleep 1
        done
    done < ports.conf

_seed-minio:
    #!/usr/bin/env bash
    set -euo pipefail
    docker run --rm --network host -v "$PWD/..:/workspace:ro" --entrypoint sh \
        minio/mc:RELEASE.2025-08-13T08-35-41Z -c '
            mc alias set --quiet fair http://localhost:9000 minioadmin minioadmin
            mc mb --ignore-existing fair/fair-data
            mc cp --quiet --recursive /workspace/data/sample/ fair/fair-data/sample/
        '

_down-dev:
    #!/usr/bin/env bash
    set -uo pipefail
    docker rm -f {{ pf_container }} 2>/dev/null || true

_tear-dev: _down-dev
    -helmfile -e dev destroy
    -kubectl delete -f manifests/postgres.yaml --ignore-not-found
    -kind delete cluster --name {{ kind_cluster }}

_urls-dev:
    @echo "ZenML:  http://localhost:8080"
    @echo "MLflow: http://localhost:5000"
    @echo "STAC:   http://localhost:8082"
    @echo "MinIO:  http://localhost:9001  (console)"
