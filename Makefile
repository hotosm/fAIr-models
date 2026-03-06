.DEFAULT_GOAL := help
.PHONY: help local k8s setup lint clean test validate docs example commit

MODE_FILE := .fair-mode
MODE := $(shell cat $(MODE_FILE) 2>/dev/null || echo local)
SYNC := --group dev --group local --group docs --group example
ifeq ($(MODE),k8s)
SYNC += --extra k8s
endif

help:
	@echo "mode: $(MODE)\n"
	@echo "targets: local k8s setup lint clean test validate docs example commit"

local:
	@echo local > $(MODE_FILE) && echo "mode: local"

k8s:
	@echo k8s > $(MODE_FILE) && echo "mode: k8s"

setup:
	uv sync $(SYNC)
ifeq ($(MODE),k8s)
	@missing=""; for cmd in kind kubectl helm helmfile mc envsubst; do \
		command -v $$cmd >/dev/null 2>&1 || missing="$$missing $$cmd"; done; \
	[ -z "$$missing" ] || { echo "Missing:$$missing"; exit 1; }
endif
	uv run pre-commit install --hook-type commit-msg --hook-type pre-commit
	uv run zenml init
	OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES uv run zenml login --local

lint:
	uv run ruff check --fix . && uv run ruff format . && uv run ty check

clean:
ifeq ($(MODE),k8s)
	$(MAKE) -C infra/dev tear
endif
	uv run zenml clean -y && rm -rf .zen artifacts dist *.egg-info

test:
	uv run pytest tests/ -v

validate:
	uv run python scripts/validate_stac_items.py && uv run python scripts/validate_model.py

docs:
	uv sync --group docs && uv run zensical serve

example:
ifeq ($(MODE),k8s)
	$(MAKE) -C infra/dev run-example
else
	uv run python examples/unet/run.py clean && uv run python examples/unet/run.py all
endif

commit:
	uv run pre-commit run --all-files && uv run cz commit

