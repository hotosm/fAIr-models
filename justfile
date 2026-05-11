set shell := ["bash", "-euo", "pipefail", "-c"]

mode_file := ".fair-mode"
mode := `cat .fair-mode 2>/dev/null || echo k8s`

import 'recipes/local.just'
import 'recipes/k8s.just'

[doc('Show current mode and available recipes')]
default:
    @echo "mode: {{ mode }}"
    @echo ""
    @just --list --unsorted

[doc('Switch to local mode')]
local:
    @echo local > {{ mode_file }} && echo "mode: local"

[doc('Switch to k8s mode (default)')]
k8s:
    @echo k8s > {{ mode_file }} && echo "mode: k8s"

[doc('One-shot: install deps, build cli image, bring up the full stack')]
setup:
    @just _setup-{{ mode }}

[doc('Full teardown: cluster, port-forwards, zenml state, artifacts')]
clean:
    @just _clean-{{ mode }}

[doc('Run all 3 example pipelines (segmentation, classification, detection)')]
example:
    @just _example-{{ mode }}

[doc('Lint and format')]
lint:
    uv run ruff check --fix . && uv run ruff format . && uv run ty check

[doc('Run tests')]
test:
    uv run pytest -v

[doc('Validate STAC items and model pipelines')]
validate:
    uv run python scripts/validate_stac_items.py && uv run python scripts/validate_model.py

[doc('Serve documentation locally')]
docs:
    uv sync --group docs && uv run zensical serve

[doc('Run pre-commit hooks and commitizen')]
commit:
    uv run pre-commit run --all-files && uv run cz commit
