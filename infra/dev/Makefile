SHELL := /bin/bash
.DEFAULT_GOAL := help
export DOCKER_HOST ?= unix://$(HOME)/.colima/default/docker.sock

NS      := fair
CLUSTER := fair-dev
PATCH   := https://raw.githubusercontent.com/hotosm/fAIr/develop/infra/zenml/patch_zenml.py
PGSTAC_DSN ?= postgresql://postgres:postgres@localhost:5432/fair_models

.PHONY: helm-repos
helm-repos:
	@helm repo add eoapi https://devseed.com/eoapi-k8s/ || true
	@helm repo add community-charts https://community-charts.github.io/helm-charts || true
	@helm repo add minio https://charts.min.io/ || true
	@helm repo update

.PHONY: cluster-up
cluster-up:
	kind create cluster --name $(CLUSTER) --config kind-config.yaml

.PHONY: cluster-down
cluster-down:
	kind delete cluster --name $(CLUSTER)

.PHONY: infra-up
infra-up: helm-repos
	@kubectl create ns $(NS) --dry-run=client -o yaml | kubectl apply -f -
	kubectl apply -n $(NS) -f postgres/
	kubectl rollout status -n $(NS) statefulset/postgres --timeout=120s
	helm upgrade --install stac eoapi/eoapi -n $(NS) -f stac/values.yaml --wait --timeout 3m
	helm upgrade --install minio minio/minio -n $(NS) -f minio/values.yaml --wait --timeout 3m
	helm upgrade --install mlflow community-charts/mlflow -n $(NS) -f mlflow/values.yaml --wait --timeout 3m
	helm upgrade --install zenml oci://public.ecr.aws/zenml/zenml --version 0.93.3 -n $(NS) -f zenml/values.yaml --wait --timeout 5m

.PHONY: infra-down
infra-down:
	-helm uninstall zenml mlflow minio stac -n $(NS)
	-kubectl delete -n $(NS) -f postgres/

.PHONY: port-forward
port-forward:
	@kubectl port-forward -n $(NS) svc/zenml 8080:80 &
	@kubectl port-forward -n $(NS) svc/stac-stac 8082:8080 &
	@kubectl port-forward -n $(NS) svc/minio 9000:9000 &
	@kubectl port-forward -n $(NS) svc/mlflow 5000:80 &
	@kubectl port-forward -n $(NS) svc/postgres 5432:5432 &

.PHONY: seed-data
seed-data:
	mc alias set fair http://localhost:9000 minioadmin minioadmin
	mc cp --recursive ../../data/sample/ fair/fair-data/sample/

.PHONY: client-patch
client-patch:
	@curl -sO $(PATCH) && uv run python patch_zenml.py && rm patch_zenml.py

.PHONY: stack-register
stack-register:
	@uv run python -c "from zenml.config.global_config import GlobalConfiguration as GC; \
		from zenml.zen_stores.rest_zen_store import RestZenStoreConfiguration as RSC; \
		gc = GC(); gc.set_store(RSC(url='http://localhost:8080', username='default', password='', verify_ssl=False)); \
		print('connected', gc.store_configuration.url)"
	zenml orchestrator register k8s_orchestrator --flavor kubernetes --kubernetes_namespace=$(NS) || true
	zenml artifact-store register s3_artifacts --flavor s3 --path=s3://zenml || true
	zenml container-registry register ghcr --flavor default --uri=ghcr.io/hotosm || true
	zenml experiment-tracker register mlflow_tracker --flavor mlflow \
		--tracking_uri=http://mlflow.$(NS).svc:80 --tracking_username=admin --tracking_password=admin || true
	zenml stack register k8s -o k8s_orchestrator -a s3_artifacts -c ghcr -e mlflow_tracker --set || true

.PHONY: run-example
run-example:
	cd ../.. && uv run python examples/unet/run.py all --stac-api-url http://localhost:8082 --dsn $(PGSTAC_DSN)

.PHONY: up
up: cluster-up infra-up port-forward seed-data client-patch stack-register

.PHONY: teardown
teardown: infra-down cluster-down

.PHONY: help
help:
	@echo "cluster-up       Create kind cluster"
	@echo "cluster-down     Delete kind cluster"
	@echo "infra-up         Deploy all services"
	@echo "infra-down       Remove all services"
	@echo "port-forward     Forward services to localhost"
	@echo "seed-data        Upload sample data to minio"
	@echo "client-patch     Patch ZenML client for Postgres"
	@echo "stack-register   Register ZenML k8s stack"
	@echo "run-example      Run UNet example E2E"
	@echo "up               Full setup"
	@echo "teardown         Full teardown"
