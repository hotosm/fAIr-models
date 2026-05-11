_up-dok8s: _tofu-apply _save-kubeconfig _apply-postgres _helmfile-dok8s _post-deploy-dok8s
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
    cd dok8s
    export FAIR_DOMAIN="$(tofu output -raw fair_domain)"
    export LETSENCRYPT_EMAIL="$(tofu output -raw letsencrypt_email)"
    export MLFLOW_ADMIN_USER="$(tofu output -raw mlflow_admin_user)"
    export MLFLOW_ADMIN_PASSWORD="$(tofu output -raw mlflow_admin_password)"
    export ZENML_ADMIN_USER="$(tofu output -raw zenml_admin_user)"
    export ZENML_ADMIN_PASSWORD="$(tofu output -raw zenml_admin_password)"
    export ZENML_STORE_API_KEY="$(tofu output -raw zenml_store_api_key)"
    cd ..
    helmfile -e dok8s apply

_post-deploy-dok8s:
    #!/usr/bin/env bash
    set -euo pipefail
    export FAIR_DOMAIN="$(cd dok8s && tofu output -raw fair_domain)"
    export LETSENCRYPT_EMAIL="$(cd dok8s && tofu output -raw letsencrypt_email)"
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
    FAIR_DOMAIN="$(cd dok8s && tofu output -raw fair_domain)"
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
    DOMAIN="$(cd dok8s && tofu output -raw fair_domain 2>/dev/null || echo unknown)"
    echo "ZenML:   https://zenml.${DOMAIN}"
    echo "MLflow:  https://mlflow.${DOMAIN}"
    echo "STAC:    https://stac.${DOMAIN}/stac"
    echo "Predict: https://predict.${DOMAIN}"
    echo "API:     https://api.${DOMAIN}/api/v1/health/"

[doc('Deploy the fAIr Django backend to dok8s. Secrets are managed by terraform — run `tofu apply` in dok8s/ first.')]
up-fair-api: _bootstrap-fair-db
    #!/usr/bin/env bash
    set -euo pipefail
    FAIR_DOMAIN="$(cd dok8s && tofu output -raw fair_domain)"
    kubectl -n {{ ns }} get secret fair-backend-secrets >/dev/null 2>&1 \
        || { echo "ERROR: fair-backend-secrets missing — run 'cd dok8s && tofu apply' first" >&2; exit 1; }
    kubectl apply -f manifests/fair-backend.yaml
    kubectl rollout status -n {{ ns }} deployment/fair-backend --timeout=300s
    echo
    echo "  https://api.${FAIR_DOMAIN}/api/v1/health/"

[doc('Force a redeploy of fair-backend (after pushing a new :test image)')]
rollout-fair-api:
    kubectl rollout restart -n {{ ns }} deployment/fair-backend
    kubectl rollout status -n {{ ns }} deployment/fair-backend --timeout=300s

_bootstrap-fair-db:
    #!/usr/bin/env bash
    set -euo pipefail
    kubectl apply -f manifests/fair-db-bootstrap.yaml
    kubectl wait --for=condition=complete -n {{ ns }} job/fair-db-bootstrap --timeout=120s

[doc('Mint a scoped kubeconfig: cluster-wide read + postgres port-forward in fair. Usage: just kubeconfig [OUT] [DURATION]')]
kubeconfig out="viewer.kubeconfig" duration="720h":
    #!/usr/bin/env bash
    set -euo pipefail
    kubectl apply -f manifests/viewer-rbac.yaml >/dev/null
    CLUSTER_NAME="$(kubectl config view --minify -o jsonpath='{.clusters[0].name}')"
    SERVER="$(kubectl config view --minify -o jsonpath='{.clusters[0].cluster.server}')"
    CA_DATA="$(kubectl config view --minify --raw -o jsonpath='{.clusters[0].cluster.certificate-authority-data}')"
    TOKEN="$(kubectl create token viewer -n {{ ns }} --duration={{ duration }})"
    cat > "{{ out }}" <<EOF
    apiVersion: v1
    kind: Config
    clusters:
      - name: ${CLUSTER_NAME}
        cluster:
          server: ${SERVER}
          certificate-authority-data: ${CA_DATA}
    users:
      - name: viewer
        user:
          token: ${TOKEN}
    contexts:
      - name: fair
        context:
          cluster: ${CLUSTER_NAME}
          namespace: {{ ns }}
          user: viewer
    current-context: fair
    EOF
    chmod 600 "{{ out }}"
    echo "Wrote {{ out }} (valid {{ duration }})"
