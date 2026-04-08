SHELL := /bin/bash
.DEFAULT_GOAL := status

-include .env
export

NS                := fair
SPACES_ENDPOINT   := https://$(DO_REGION).digitaloceanspaces.com
FAIR_LABEL_DOMAIN := $(FAIR_DOMAIN)
DOKS_CONTEXT      := do-$(DO_REGION)-$(CLUSTER_NAME)
STAC_API_URL       = https://stac.$(FAIR_DOMAIN)/stac
PGSTAC_DSN         = postgresql://$(PG_USER):$(PG_PASSWORD)@$(PG_HOST):$(PG_PORT)/fair_models?sslmode=require
INFRA_NODE_SIZE   ?= s-2vcpu-4gb
S                  = scripts

.PHONY: up infra deploy tear status urls run-example

up: infra
	@$(MAKE) --no-print-directory deploy

infra:
	@$(S)/cluster.sh create
	@$(S)/database.sh create
	@$(S)/database.sh write-env
	@$(S)/spaces.sh create
	@$(S)/database.sh init

deploy:
	@$(S)/validate-env.sh
	@$(S)/deploy.sh helm
	@$(S)/deploy.sh cert-issuer
	@$(S)/deploy.sh stac-ingress
	@$(S)/deploy.sh dns
	@$(S)/spaces.sh seed
	@$(S)/deploy.sh register-stack
	@$(MAKE) --no-print-directory urls

tear:
	-helmfile destroy 2>/dev/null
	-$(S)/database.sh delete
	-$(S)/cluster.sh delete
	@echo "Spaces bucket $(SPACES_BUCKET) preserved. Delete manually if needed."

status:
	@echo "Cluster:  $$(doctl kubernetes cluster get $(CLUSTER_NAME) --format Status --no-header 2>/dev/null || echo 'not found')"
	@echo "Database: $$(doctl databases list --format Status --no-header 2>/dev/null | head -1 || echo 'not found')"
	@echo "Pods:"
	@kubectl get pods -n $(NS) --no-headers 2>/dev/null | sed 's/^/  /' || echo "  no context"
	@echo ""
	@$(MAKE) --no-print-directory urls 2>/dev/null || true

urls:
	@echo "ZenML:  https://zenml.$(FAIR_DOMAIN)"
	@echo "MLflow: https://mlflow.$(FAIR_DOMAIN)"
	@echo "STAC:   https://stac.$(FAIR_DOMAIN)/stac"

run-example:
	@TOKEN=$$(curl -fsS -X POST https://zenml.$(FAIR_DOMAIN)/api/v1/login \
		-d 'username=$(ZENML_ADMIN_USER)&password=$(ZENML_ADMIN_PASSWORD)&grant_type=password' | \
		python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])") && \
	cd ../.. && \
	ZENML_STORE_URL=https://zenml.$(FAIR_DOMAIN) \
	ZENML_STORE_API_TOKEN=$$TOKEN \
	ZENML_STORE_VERIFY_SSL=false \
	FAIR_FORCE_CPU=1 \
	FAIR_LABEL_DOMAIN=$(FAIR_LABEL_DOMAIN) \
	AWS_ENDPOINT_URL=$(SPACES_ENDPOINT) \
	AWS_ACCESS_KEY_ID=$(SPACES_ACCESS_KEY) \
	AWS_SECRET_ACCESS_KEY=$(SPACES_SECRET_KEY) \
	uv run python examples/segmentation/run.py all \
		--stac-api-url $(STAC_API_URL) \
		--dsn "$(PGSTAC_DSN)" \
		--data-prefix s3://$(SPACES_BUCKET)/data/sample
