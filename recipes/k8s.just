cli_image := "fair-dev-cli:latest"
cli_remote := "ghcr.io/hotosm/fair-models/fair-dev-cli:latest"
pf_container := "fair-dev-cli-pf"

[doc('Drop into a shell inside the fair-dev-cli container')]
cli:
    #!/usr/bin/env bash
    set -euo pipefail
    just _cli-build
    mkdir -p "$HOME/.kube"
    docker run --rm -it \
        --network host \
        -v /var/run/docker.sock:/var/run/docker.sock \
        -v "$HOME/.kube":/root/.kube \
        -v "$PWD":/workspace -w /workspace \
        {{ cli_image }} bash

[doc('Bring up services without re-installing host deps')]
up:
    #!/usr/bin/env bash
    set -euo pipefail
    just _cli-build
    mkdir -p "$HOME/.kube"
    just _cli-run just _bootstrap-stack
    just _stack-register

[doc('Stop port-forwards (kind cluster preserved)')]
down:
    #!/usr/bin/env bash
    set -uo pipefail
    docker rm -f {{ pf_container }} 2>/dev/null || true

_setup-k8s:
    #!/usr/bin/env bash
    set -euo pipefail
    uv sync --group dev --group local --group docs --group example --extra k8s
    uv run pre-commit install --hook-type commit-msg --hook-type pre-commit
    uv run zenml init >/dev/null
    just _cli-build
    mkdir -p "$HOME/.kube"
    just _cli-run just _bootstrap-stack
    just _stack-register
    just _print-alias

_clean-k8s:
    #!/usr/bin/env bash
    set -uo pipefail
    docker rm -f {{ pf_container }} 2>/dev/null || true
    if docker image inspect {{ cli_image }} >/dev/null 2>&1; then
        just _cli-run just _teardown-stack || true
    fi
    uv run zenml clean -y || true
    rm -rf .zen artifacts dist *.egg-info

_example-k8s:
    #!/usr/bin/env bash
    set -euo pipefail
    uv run zenml stack set k8s
    for ex in segmentation classification detection; do
        AWS_ENDPOINT_URL=http://localhost:9000 \
        FAIR_STAC_API_URL=http://localhost:8082 \
        FAIR_DSN=postgresql://postgres:postgres@localhost:5432/fair_models \
            uv run --group example python "examples/$ex/run.py"
    done

_cli-build:
    #!/usr/bin/env bash
    set -euo pipefail
    if docker image inspect {{ cli_image }} >/dev/null 2>&1; then
        exit 0
    fi
    if docker pull {{ cli_remote }} >/dev/null 2>&1; then
        docker tag {{ cli_remote }} {{ cli_image }}
    else
        docker build -t {{ cli_image }} infra/cli
    fi

_cli-run +args:
    docker run --rm \
        --network host \
        -v /var/run/docker.sock:/var/run/docker.sock \
        -v "$HOME/.kube":/root/.kube \
        -v "$PWD":/workspace -w /workspace \
        -e HOST_HOME="$HOME" \
        -e HOST_WORKSPACE="$PWD" \
        {{ cli_image }} {{ args }}

_bootstrap-stack:
    @cd infra && just up dev

_teardown-stack:
    @cd infra && just tear dev

_stack-register:
    #!/usr/bin/env bash
    set -euo pipefail
    export ZENML_URL="http://localhost:8080"
    export ZENML_ADMIN_USER="default"
    export ZENML_ADMIN_PASSWORD=""
    export ZENML_STORE_URL="$ZENML_URL"
    TOKEN=$(infra/scripts/zenml-token.sh --wait) || { echo "ERROR: failed to fetch ZenML token" >&2; exit 1; }
    export ZENML_STORE_API_TOKEN="$TOKEN"
    uv run zenml stack import dev -f stacks/local.yaml --ignore-version-mismatch || true
    uv run zenml stack import k8s -f stacks/k8s.yaml --ignore-version-mismatch || true
    uv run zenml stack set k8s

_print-alias:
    #!/usr/bin/env bash
    cat <<'EOF'

    Stack is up. URLs:
      ZenML:  http://localhost:8080
      MLflow: http://localhost:5000
      STAC:   http://localhost:8082
      MinIO:  http://localhost:9001  (console)

    For ad-hoc infra commands, add to your shell rc:
      alias fair-dev-cli='docker run --rm -it --network host \
          -v /var/run/docker.sock:/var/run/docker.sock \
          -v "$HOME/.kube":/root/.kube \
          -v "$PWD":/workspace -w /workspace fair-dev-cli:latest'

    Or run `just cli` to drop into the container.
    EOF
