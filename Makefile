# PROJECT_NAME := fair-models # zenml doesn't support capitals in project name

GITHUB_OWNER := hotosm
GITHUB_REPO := fAIr-models

.PHONY: run setup-local setup-stage clean

init:
	uv sync --group local
	uv run zenml init
	uv run zenml login --local

setup:
	mkdir -p .zen artifacts
	uv sync --group local
	uv run zenml integration install wandb github -y
	uv run zenml model-registry flavor register fair_integrations.registry.flavor.STACModelRegistryFlavor
# 	uv run zenml project register $(PROJECT_NAME)
# 	uv run zenml project set $(PROJECT_NAME)
	uv run zenml stack import local -f fair_integrations/stacks/local.yaml
	uv run zenml stack set local
	uv run zenml code-repository register github-repo --type=github --owner=$(GITHUB_OWNER) --repository=$(GITHUB_REPO)

run:
# 	uv run zenml login --local --project $(PROJECT_NAME) 
	uv run zenml login --local 

clean:
	uv run zenml clean -y
	rm -rf .zen artifacts

example: 
	uv run examples/iris.py --config examples/iris_train.yaml 
	uv run examples/iris.py --config examples/iris_inference.yaml inference

