set shell := ["bash", "-euo", "pipefail", "-c"]

mode_file := ".fair-mode"
mode := `cat .fair-mode 2>/dev/null || echo local`
[private]
_sync := if mode == "k8s" { "--group dev --group local --group docs --group example --extra k8s" } else { "--group dev --group local --group docs --group example" }

[doc('Show current mode and available recipes')]
default:
    @echo "mode: {{ mode }}"
    @echo ""
    @just --list --unsorted

[doc('Switch to local mode')]
local:
    @echo local > {{ mode_file }} && echo "mode: local"

[doc('Switch to k8s mode')]
k8s:
    @echo k8s > {{ mode_file }} && echo "mode: k8s"

[doc('Install dependencies and configure tools')]
setup:
    #!/usr/bin/env bash
    set -euo pipefail
    uv sync {{ _sync }}
    if [[ "$(cat {{ mode_file }} 2>/dev/null || echo local)" == "k8s" ]]; then
        missing=""
        for cmd in kind kubectl helm helmfile mc; do
            command -v "$cmd" >/dev/null 2>&1 || missing="$missing $cmd"
        done
        [[ -z "$missing" ]] || { echo "Missing:$missing"; exit 1; }
    fi
    uv run pre-commit install --hook-type commit-msg --hook-type pre-commit
    uv run zenml init
    OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES uv run zenml login --local

[doc('Lint and format')]
lint:
    uv run ruff check --fix . && uv run ruff format . && uv run ty check

[doc('Remove ZenML state and artifacts')]
clean:
    #!/usr/bin/env bash
    set -euo pipefail
    if [[ "$(cat {{ mode_file }} 2>/dev/null || echo local)" == "k8s" ]]; then
        just --justfile infra/dev/justfile tear
    fi
    uv run zenml clean -y
    rm -rf .zen artifacts dist *.egg-info

[doc('Run tests')]
test:
    uv run pytest tests/ -v

[doc('Run model tests inside Docker (requires built images)')]
test-models model="":
    #!/usr/bin/env bash
    set -euo pipefail
    if [[ -n "{{ model }}" ]]; then
        dirs=("models/{{ model }}")
    else
        dirs=(models/*/tests)
        dirs=("${dirs[@]%/tests}")
    fi
    for model_dir in "${dirs[@]}"; do
        name=$(basename "$model_dir")
        echo "=== Testing $name ==="
        docker build -f "$model_dir/Dockerfile" -t "fair-models/$name:test" .
        echo "Step tests :"
        docker run --rm --entrypoint "" \
            -e FAIR_FORCE_CPU=1 \
            "fair-models/$name:test" \
            bash -c "pip install pytest && python -m pytest models/$name/tests/ -v --tb=short"
        echo "Integration test :"
        docker run --rm --entrypoint "" \
            -e FAIR_FORCE_CPU=1 \
            "fair-models/$name:test" \
            bash -c "pip install pytest 'zenml[server]' && python -m pytest models/test_integration.py -v --tb=short -m slow --model-dir=models/$name"
    done

[doc('Validate STAC items and model pipelines')]
validate:
    uv run python scripts/validate_stac_items.py && uv run python scripts/validate_model.py

[doc('Serve documentation locally')]
docs:
    uv sync --group docs && uv run zensical serve

[doc('Run all example pipelines')]
example:
    #!/usr/bin/env bash
    set -euo pipefail
    if [[ "$(cat {{ mode_file }} 2>/dev/null || echo local)" == "k8s" ]]; then
        just --justfile infra/dev/justfile run-example
    else
        for ex in segmentation classification detection; do
            uv run python "examples/$ex/run.py" clean
            uv run python "examples/$ex/run.py" all
        done
    fi

[doc('Run pre-commit hooks and commitizen')]
commit:
    uv run pre-commit run --all-files && uv run cz commit
