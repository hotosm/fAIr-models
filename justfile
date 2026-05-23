set shell := ["bash", "-euo", "pipefail", "-c"]

compose := "docker compose -f infra/compose/docker-compose.yml"
stac := "python3 scripts/stac_asset.py"

[doc('Install deps, bring up the stack, register the ZenML stack')]
setup:
    uv sync --group dev --group docs --extra k8s
    uv run pre-commit install --hook-type commit-msg --hook-type pre-commit
    {{ compose }} up -d --wait
    uv run zenml init >/dev/null
    infra/compose/register-stack.sh
    @echo
    @echo "Stack up. ZenML :8080  MLflow :5000  STAC :8082  MinIO :9001"
    @echo "Next: 'just build' to build model images, then 'just example'."

[doc('Build model image(s). No arg = all (e.g. `just build unet_segmentation`)')]
build model="":
    #!/usr/bin/env bash
    set -euo pipefail
    for d in $([ -n "{{ model }}" ] && echo "models/{{ model }}" || echo models/*); do
        [[ -f "$d/Dockerfile" ]] || continue
        href=$({{ stac }} "$d/stac-item.json" mlm:training)
        echo "==> building $(basename "$d") -> $href"
        docker build -f "$d/Dockerfile" --target runtime -t "$href" .
    done

[doc('Run example pipeline(s). No arg = all (e.g. `just example unet_segmentation`)')]
example model="":
    #!/usr/bin/env bash
    set -euo pipefail
    export AWS_ENDPOINT_URL=http://localhost:9000
    export AWS_ACCESS_KEY_ID=minioadmin
    export AWS_SECRET_ACCESS_KEY=minioadmin
    export FAIR_STAC_API_URL=http://localhost:8082
    export FAIR_DSN=postgresql://postgres:postgres@localhost:5432/fair_models
    export FAIR_UPLOAD_ARTIFACTS=true
    uv run python examples/run.py {{ model }}

[doc('Serve a model inference container on http://localhost:8090 (Ctrl-C to stop)')]
serve model:
    #!/usr/bin/env bash
    set -euo pipefail
    href=$({{ stac }} "models/{{ model }}/stac-item.json" mlm:inference)
    docker build -f "models/{{ model }}/Dockerfile" --target inference -t "$href" .
    docker run --rm -it --network host \
        -e MODEL_MODULE=models.{{ model }}.pipeline \
        "$href" \
        fair.serve.base:create_app --factory --host 0.0.0.0 --port 8090

[doc('End-to-end test: trained ONNX + OAM TMS -> POST /predict (needs prior `just example`)')]
test-serve model:
    uv run python scripts/test_serve.py {{ model }}

[doc('Stop the stack (containers stopped, state preserved)')]
down:
    {{ compose }} stop

[doc('Bring the stack back up after `just down`')]
up:
    {{ compose }} start

[doc('Destroy the stack: containers + volumes + local ZenML state + artifacts')]
tear:
    -{{ compose }} down -v
    -uv run zenml clean -y
    rm -rf .zen artifacts dist *.egg-info

[doc('Lint and format')]
lint:
    uv run ruff check --fix . && uv run ruff format . && uv run ty check

[doc('Run tests')]
test:
    uv run pytest -v

[doc('Validate STAC items and model pipelines')]
validate:
    uv run python scripts/validate_stac_items.py && uv run python scripts/validate_model.py

[doc('End-to-end smoke test against a deployed fAIr API (~30-40 min)')]
smoke *args:
    #!/usr/bin/env bash
    set -euo pipefail
    uv run python scripts/smoke_e2e.py {{ args }}

[doc('Serve documentation locally')]
docs:
    uv sync --group docs && uv run zensical serve

[doc('Run pre-commit hooks and commitizen')]
commit:
    uv run pre-commit run --all-files && uv run cz commit
