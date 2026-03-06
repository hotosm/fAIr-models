.DEFAULT_GOAL := help
.PHONY: help setup lint clean test validate docs example

help: 
	@awk -F ':[^#]*## ' '/^[a-zA-Z_-]+:.*##/{printf "\033[36m%-12s\033[0m %s\n",$$1,$$2}' $(MAKEFILE_LIST)

setup: 
	uv sync --group local --group docs
	uv run pre-commit install --hook-type commit-msg --hook-type pre-commit
	uv run zenml init && uv run zenml login --local

lint: 
	uv run ruff check --fix . && uv run ruff format . && uv run ty check

clean: 
	uv run zenml clean -y && rm -rf .zen artifacts dist *.egg-info

test: 
	uv run pytest tests/ -v

validate: 
	uv run python scripts/validate_stac_items.py && uv run python scripts/validate_model.py

docs: 
	uv sync --group docs && uv run zensical serve

example: 
	uv run python examples/unet/run.py clean && uv run python examples/unet/run.py all

