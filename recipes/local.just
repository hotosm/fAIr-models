_setup-local:
    #!/usr/bin/env bash
    set -euo pipefail
    uv sync --group dev --group local --group docs --group example
    uv run pre-commit install --hook-type commit-msg --hook-type pre-commit
    uv run zenml init >/dev/null
    OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES uv run zenml login --local
    echo
    echo "Local stack ready. ZenML dashboard: http://localhost:8237"

_clean-local:
    #!/usr/bin/env bash
    set -uo pipefail
    uv run zenml clean -y || true
    rm -rf .zen artifacts dist *.egg-info

_example-local:
    #!/usr/bin/env bash
    set -euo pipefail
    for ex in segmentation classification detection; do
        uv run --group example python "examples/$ex/run.py"
    done
