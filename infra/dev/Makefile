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
NVKIND  := $(shell command -v nvkind 2>/dev/null || echo $(HOME)/go/bin/nvkind)

IMAGE ?= ghcr.io/hotosm/fair-models/example-unet:v1

.PHONY: helm-repos cluster-up cluster-down infra-up infra-down \
        port-forward kill-port-forward seed-data client-patch \
        stack-register run-example build-image run-example-k8s \
        up teardown help

helm-repos:
	@helm repo add eoapi https://devseed.com/eoapi-k8s/ 2>/dev/null || true
	@helm repo add community-charts https://community-charts.github.io/helm-charts 2>/dev/null || true
	@helm repo add minio https://charts.min.io/ 2>/dev/null || true
	@helm repo add nvdp https://nvidia.github.io/k8s-device-plugin 2>/dev/null || true
	@helm repo update

cluster-up:
	-$(NVKIND) cluster create --name $(CLUSTER) --config-template kind-config.yaml
	@# nvkind may fail at PatchProcDriverNvidia (not needed for single-GPU); ensure RuntimeClass exists
	@kubectl apply -f - <<< '{"apiVersion":"node.k8s.io/v1","kind":"RuntimeClass","handler":"nvidia","metadata":{"name":"nvidia"}}' 2>/dev/null || true

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
	@# GPU: nvkind installed toolkit + configured containerd; label node and deploy device plugin
	@kubectl label node $(CLUSTER)-worker2 nvidia.com/gpu.present=true --overwrite >/dev/null
	helm upgrade --install nvidia-device-plugin nvdp/nvidia-device-plugin \
		-n nvidia --create-namespace \
		--set runtimeClassName=nvidia \
		--set deviceDiscoveryStrategy=nvml \
		--wait --timeout 5m

infra-down:
	-helm uninstall nvidia-device-plugin -n nvidia
	-kubectl delete ns nvidia --ignore-not-found
	-helm uninstall zenml mlflow minio stac -n $(NS)
	-kubectl delete -n $(NS) -f postgres/

port-forward: kill-port-forward
	@kubectl port-forward -n $(NS) svc/zenml 8080:80 >/dev/null 2>&1 & echo $$! >> $(PID_FILE)
	@kubectl port-forward -n $(NS) svc/stac-stac 8082:8080 >/dev/null 2>&1 & echo $$! >> $(PID_FILE)
	@kubectl port-forward -n $(NS) svc/minio 9000:9000 >/dev/null 2>&1 & echo $$! >> $(PID_FILE)
	@kubectl port-forward -n $(NS) svc/minio-console 9001:9001 >/dev/null 2>&1 & echo $$! >> $(PID_FILE)
	@kubectl port-forward -n $(NS) svc/mlflow 5000:80 >/dev/null 2>&1 & echo $$! >> $(PID_FILE)
	@kubectl port-forward -n $(NS) svc/postgres 5432:5432 >/dev/null 2>&1 & echo $$! >> $(PID_FILE)
	@for port in 8080 8082 9000 9001 5000 5432; do \
		for i in $$(seq 1 30); do nc -z localhost $$port 2>/dev/null && break; sleep 1; done; \
	done

kill-port-forward:
	@if [ -f $(PID_FILE) ]; then \
		while read -r pid; do kill "$$pid" 2>/dev/null || true; done < $(PID_FILE); \
		rm -f $(PID_FILE); \
	fi

seed-data:
	@uv run --with minio python -c "from pathlib import Path; from minio import Minio; c = Minio('localhost:9000', 'minioadmin', 'minioadmin', secure=False); root = Path('../../data/sample'); files = [f for f in root.rglob('*') if f.is_file()]; print(f'Uploading {len(files)} files to fair-data/sample/'); [c.fput_object('fair-data', f'sample/{f.relative_to(root)}', str(f)) for f in files]; print('Done')"

client-patch:
	@curl -sO $(PATCH) && uv run python patch_zenml.py && rm patch_zenml.py

stack-register:
	@uv run python -c "from zenml.config.global_config import GlobalConfiguration as GC; \
		from zenml.zen_stores.rest_zen_store import RestZenStoreConfiguration as RSC; \
		gc = GC(); gc.set_store(RSC(url='http://localhost:8080', username='default', password='', verify_ssl=False))"
	-uv run zenml orchestrator register k8s_orchestrator --flavor kubernetes \
		--kubernetes_namespace=$(NS) --kubernetes_context=kind-$(CLUSTER) \
		--pod_settings='{"env": [{"name": "ZENML_STORE_URL", "value": "http://zenml.$(NS).svc:80"}, {"name": "AWS_ENDPOINT_URL", "value": "http://minio.$(NS).svc:9000"}, {"name": "AWS_ACCESS_KEY_ID", "value": "minioadmin"}, {"name": "AWS_SECRET_ACCESS_KEY", "value": "minioadmin"}]}' \
		--orchestrator_pod_settings='{"env": [{"name": "ZENML_STORE_URL", "value": "http://zenml.$(NS).svc:80"}, {"name": "AWS_ENDPOINT_URL", "value": "http://minio.$(NS).svc:9000"}, {"name": "AWS_ACCESS_KEY_ID", "value": "minioadmin"}, {"name": "AWS_SECRET_ACCESS_KEY", "value": "minioadmin"}]}'
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

build-image:
	cd ../.. && docker buildx build -f models/example_unet/Dockerfile -t $(IMAGE) --platform linux/amd64 --load --no-cache .
	kind load docker-image $(IMAGE) --name $(CLUSTER)

run-example-k8s:
	cd ../.. && uv run zenml stack set k8s
	cd ../.. && uv run python examples/unet/run.py all \
		--stac-api-url http://localhost:8082 --dsn $(PGSTAC_DSN)

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
	@echo "build-image         Build & load Docker image into kind workers"
	@echo "run-example         Run UNet example E2E (local stack)"
	@echo "run-example-k8s     Run UNet example E2E (k8s stack)"
	@echo "up                  Full setup"
	@echo "teardown            Full teardown"
