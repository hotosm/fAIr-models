.PHONY: init setup clean example

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
	rm -rf .zen artifacts

example:
	uv run python examples/building_segmentation/download.py
	uv run python examples/building_segmentation/register_dataset.py
	uv run python examples/building_segmentation/register_basemodel.py
	uv run python examples/building_segmentation/finetune.py
	uv run python examples/building_segmentation/promote.py
	uv run python examples/building_segmentation/predict.py

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

