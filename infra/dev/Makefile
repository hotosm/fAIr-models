SHELL := /bin/bash
.DEFAULT_GOAL := help

# Docker socket auto-detect
DOCKER_HOST ?= $(if $(wildcard /var/run/docker.sock),,\
  $(if $(wildcard $(HOME)/.colima/default/docker.sock),unix://$(HOME)/.colima/default/docker.sock,\
  $(if $(wildcard $(HOME)/.docker/run/docker.sock),unix://$(HOME)/.docker/run/docker.sock,)))
export DOCKER_HOST

NS      := fair
CLUSTER := fair-dev
PATCH   := https://raw.githubusercontent.com/hotosm/fAIr/develop/infra/zenml/patch_zenml.py
PGSTAC_DSN ?= postgresql://postgres:postgres@localhost:5432/fair_models
PID_FILE := .port-forward.pids
MC       ?= mc

.PHONY: helm-repos cluster-up cluster-down infra-up infra-down \
        port-forward kill-port-forward seed-data client-patch \
        stack-register run-example up teardown help

helm-repos:
	@helm repo add eoapi https://devseed.com/eoapi-k8s/ 2>/dev/null || true
	@helm repo add community-charts https://community-charts.github.io/helm-charts 2>/dev/null || true
	@helm repo add minio https://charts.min.io/ 2>/dev/null || true
	@helm repo update

cluster-up:
	kind create cluster --name $(CLUSTER) --config kind-config.yaml

cluster-down:
	kind delete cluster --name $(CLUSTER)

infra-up: helm-repos
	@kubectl create ns $(NS) --dry-run=client -o yaml | kubectl apply -f -
	kubectl apply -n $(NS) -f postgres/
	kubectl rollout status -n $(NS) statefulset/postgres --timeout=120s
	helm upgrade --install stac eoapi/eoapi -n $(NS) -f stac/values.yaml --timeout 10m
	kubectl rollout status -n $(NS) deployment/stac-stac --timeout=300s
	helm upgrade --install minio minio/minio -n $(NS) -f minio/values.yaml --wait --timeout 5m
	helm upgrade --install mlflow community-charts/mlflow -n $(NS) -f mlflow/values.yaml --wait --timeout 5m
	helm upgrade --install zenml oci://public.ecr.aws/zenml/zenml --version 0.93.3 -n $(NS) -f zenml/values.yaml --wait --timeout 8m

infra-down:
	-helm uninstall zenml mlflow minio stac -n $(NS)
	-kubectl delete -n $(NS) -f postgres/

port-forward: kill-port-forward
	@kubectl port-forward -n $(NS) svc/zenml 8080:80 & echo $$! >> $(PID_FILE)
	@kubectl port-forward -n $(NS) svc/stac-stac 8082:8080 & echo $$! >> $(PID_FILE)
	@kubectl port-forward -n $(NS) svc/minio 9000:9000 & echo $$! >> $(PID_FILE)
	@kubectl port-forward -n $(NS) svc/mlflow 5000:80 & echo $$! >> $(PID_FILE)
	@kubectl port-forward -n $(NS) svc/postgres 5432:5432 & echo $$! >> $(PID_FILE)
	@for port in 8080 8082 9000 5000 5432; do \
		for i in $$(seq 1 30); do nc -z localhost $$port 2>/dev/null && break; sleep 1; done; \
	done

kill-port-forward:
	@if [ -f $(PID_FILE) ]; then \
		while read -r pid; do kill "$$pid" 2>/dev/null || true; done < $(PID_FILE); \
		rm -f $(PID_FILE); \
	fi

seed-data:
	$(MC) alias set fair http://localhost:9000 minioadmin minioadmin
	$(MC) cp --recursive ../../data/sample/ fair/fair-data/sample/

client-patch:
	@curl -sO $(PATCH) && uv run python patch_zenml.py && rm patch_zenml.py

stack-register:
	@uv run python -c "from zenml.config.global_config import GlobalConfiguration as GC; \
		from zenml.zen_stores.rest_zen_store import RestZenStoreConfiguration as RSC; \
		gc = GC(); gc.set_store(RSC(url='http://localhost:8080', username='default', password='', verify_ssl=False))"
	-uv run zenml orchestrator register k8s_orchestrator --flavor kubernetes --kubernetes_namespace=$(NS)
	-uv run zenml artifact-store register s3_artifacts --flavor s3 --path=s3://zenml \
		--client_kwargs='{"endpoint_url": "http://minio.$(NS).svc:9000", "aws_access_key_id": "minioadmin", "aws_secret_access_key": "minioadmin"}'
	-uv run zenml artifact-store register s3_local --flavor s3 --path=s3://zenml \
		--client_kwargs='{"endpoint_url": "http://localhost:9000", "aws_access_key_id": "minioadmin", "aws_secret_access_key": "minioadmin"}'
	-uv run zenml container-registry register ghcr --flavor default --uri=ghcr.io/hotosm
	-uv run zenml experiment-tracker register mlflow_tracker --flavor mlflow \
		--tracking_uri=http://mlflow.$(NS).svc:80 --tracking_username=admin --tracking_password=admin
	-uv run zenml experiment-tracker register mlflow_local --flavor mlflow \
		--tracking_uri=http://localhost:5000 --tracking_username=admin --tracking_password=admin
	-uv run zenml stack register k8s -o k8s_orchestrator -a s3_artifacts -c ghcr -e mlflow_tracker
	-uv run zenml stack register dev -o default -a s3_local -e mlflow_local --set

run-example:
	cd ../.. && uv run python examples/unet/run.py all --stac-api-url http://localhost:8082 --dsn $(PGSTAC_DSN)

up: cluster-up infra-up port-forward seed-data client-patch stack-register
teardown: kill-port-forward infra-down cluster-down

help:
	@echo "cluster-up          Create kind cluster"
	@echo "cluster-down        Delete kind cluster"
	@echo "infra-up            Deploy all services"
	@echo "infra-down          Remove all services"
	@echo "port-forward        Forward services to localhost"
	@echo "kill-port-forward   Stop port-forwards"
	@echo "seed-data           Upload sample data to MinIO"
	@echo "client-patch        Patch ZenML client for Postgres"
	@echo "stack-register      Register ZenML stacks"
	@echo "run-example         Run UNet example E2E"
	@echo "up                  Full setup"
	@echo "teardown            Full teardown"
