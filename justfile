set shell := ["bash", "-euo", "pipefail", "-c"]

mode_file := ".fair-mode"
mode := `cat .fair-mode 2>/dev/null || echo local`

_sync := if mode == "k8s" { "--group dev --group local --group docs --group example --extra k8s" } else { "--group dev --group local --group docs --group example" }

[doc('Show current mode and available recipes')]
default:
    @echo "mode: {{mode}}"
    @echo ""
    @just --list --unsorted

[doc('Switch to local mode')]
local:
    @echo local > {{mode_file}} && echo "mode: local"

[doc('Switch to k8s mode')]
k8s:
    @echo k8s > {{mode_file}} && echo "mode: k8s"

[doc('Install dependencies and configure tools')]
setup:
    #!/usr/bin/env bash
    set -euo pipefail
    uv sync {{_sync}}
    if [[ "$(cat {{mode_file}} 2>/dev/null || echo local)" == "k8s" ]]; then
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
    if [[ "$(cat {{mode_file}} 2>/dev/null || echo local)" == "k8s" ]]; then
        just --justfile infra/dev/justfile tear
    fi
    uv run zenml clean -y
    rm -rf .zen artifacts dist *.egg-info

[doc('Run tests')]
test:
    uv run pytest tests/ -v

[doc('Validate STAC items and model pipelines')]
validate:
    uv run python scripts/validate_stac_items.py && uv run python scripts/validate_model.py

[doc('Serve documentation locally')]
docs:
    uv sync --group docs && uv run zensical serve

[doc('Run example pipeline')]
example:
    #!/usr/bin/env bash
    set -euo pipefail
    if [[ "$(cat {{mode_file}} 2>/dev/null || echo local)" == "k8s" ]]; then
        just --justfile infra/dev/justfile run-example
    else
        uv run python examples/unet/run.py clean
        uv run python examples/unet/run.py all
    fi

[doc('Run pre-commit hooks and commitizen')]
commit:
    uv run pre-commit run --all-files && uv run cz commit
