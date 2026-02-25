.PHONY: init setup clean example lint format typecheck test build bump pre-commit validate-stac

init:
	uv sync --group local
	uv run zenml init
	uv run zenml login --local

setup:
	mkdir -p .zen artifacts
	uv sync --group local
	uv run zenml integration install wandb -y
	uv run zenml stack import local -f stacks/local.yaml
	uv run zenml stack set local

run:
	uv run zenml login --local

clean:
	uv run zenml clean -y
	rm -rf .zen artifacts dist *.egg-info

example:
	uv run python examples/unet/run.py clean
	uv run python examples/unet/run.py all

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

typecheck:
	uv run ty check

test:
	uv run pytest tests/ -v

build:
	uv build

bump:
	uv run cz bump --changelog

pre-commit:
	uv run pre-commit install --hook-type commit-msg --hook-type pre-commit

validate-stac:
	uv run python scripts/validate_stac_items.py

