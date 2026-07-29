# Recipes for a real (non-kind) cluster. Cluster provisioning, kubeconfig and
# DNS are managed outside this repo; every recipe reads its configuration from
# the environment and fails if a required value is missing.
#
#   FAIR_DOMAIN          apex domain serving the stack (e.g. fair.example.com)
#   LETSENCRYPT_EMAIL    contact address on the cluster issuer
#   MLFLOW_ADMIN_USER    MLflow admin credentials
#   MLFLOW_ADMIN_PASSWORD
#   ZENML_ADMIN_USER     ZenML admin credentials
#   ZENML_ADMIN_PASSWORD
#   ZENML_STORE_API_KEY

_up-cluster: _apply-postgres _helmfile-cluster _post-deploy-cluster
    @just _urls-cluster

_require-env:
    #!/usr/bin/env bash
    set -euo pipefail
    missing=()
    for var in FAIR_DOMAIN LETSENCRYPT_EMAIL MLFLOW_ADMIN_USER MLFLOW_ADMIN_PASSWORD \
               ZENML_ADMIN_USER ZENML_ADMIN_PASSWORD ZENML_STORE_API_KEY; do
        [[ -n "${!var:-}" ]] || missing+=("$var")
    done
    if (( ${#missing[@]} )); then
        echo "ERROR: unset required env vars: ${missing[*]}" >&2
        exit 1
    fi

_helmfile-cluster: _require-env
    helmfile -e cluster apply

_post-deploy-cluster: _require-env
    #!/usr/bin/env bash
    set -euo pipefail
    envsubst '${FAIR_DOMAIN} ${LETSENCRYPT_EMAIL}' < manifests/knative-serving.yaml.gotmpl | kubectl apply -f -
    kubectl apply -f manifests/s3-credentials.yaml
    just _wait-lb
    kubectl wait --for=condition=Available deployment/cert-manager-webhook -n cert-manager --timeout=180s
    envsubst '${FAIR_DOMAIN} ${LETSENCRYPT_EMAIL}' < manifests/cluster-issuer.yaml.gotmpl | kubectl apply -f -
    envsubst '${FAIR_DOMAIN} ${LETSENCRYPT_EMAIL}' < manifests/ingress.yaml.gotmpl | kubectl apply -f -

_wait-lb:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Waiting for ingress LoadBalancer IP..."
    for i in $(seq 1 60); do
        IP=$(kubectl get svc -n ingress-nginx ingress-nginx-controller \
            -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || true)
        if [[ -n "$IP" ]]; then
            echo "  $IP"
            # `<svc>.predict.$FAIR_DOMAIN` is two labels deep and DNS wildcards
            # do not span dots, so both records are required.
            echo "  point *.${FAIR_DOMAIN} and *.predict.${FAIR_DOMAIN} at this address"
            exit 0
        fi
        sleep 5
    done
    echo "ERROR: LoadBalancer IP not assigned after 5 minutes" >&2; exit 1

_down-cluster:
    -helmfile -e cluster destroy

_tear-cluster:
    #!/usr/bin/env bash
    set -euo pipefail
    helmfile -e cluster destroy 2>/dev/null || true
    kubectl delete -f manifests/postgres.yaml --ignore-not-found

_urls-cluster:
    #!/usr/bin/env bash
    set -euo pipefail
    DOMAIN="${FAIR_DOMAIN:-unset}"
    echo "ZenML:   https://zenml.${DOMAIN}"
    echo "MLflow:  https://mlflow.${DOMAIN}"
    echo "STAC:    https://stac.${DOMAIN}/stac"
    echo "Predict: https://predict.${DOMAIN}"
    echo "API:     https://api.${DOMAIN}/api/v1/health/"

[doc('Deploy the fAIr Django backend. Needs the fair-backend-secrets secret in the cluster.')]
up-fair-api: _bootstrap-fair-db
    #!/usr/bin/env bash
    set -euo pipefail
    kubectl -n {{ ns }} get secret fair-backend-secrets >/dev/null 2>&1 \
        || { echo "ERROR: fair-backend-secrets missing, create it before deploying" >&2; exit 1; }
    kubectl apply -f manifests/fair-backend.yaml
    kubectl rollout status -n {{ ns }} deployment/fair-backend --timeout=300s
    echo
    echo "  https://api.${FAIR_DOMAIN:-unset}/api/v1/health/"

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
