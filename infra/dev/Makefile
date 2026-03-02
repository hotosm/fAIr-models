SHELL := /bin/bash
.DEFAULT_GOAL := help

# Docker socket auto-detect
DOCKER_HOST ?= $(if $(wildcard /var/run/docker.sock),,\
  $(if $(wildcard $(HOME)/.colima/default/docker.sock),unix://$(HOME)/.colima/default/docker.sock,\
  $(if $(wildcard $(HOME)/.docker/run/docker.sock),unix://$(HOME)/.docker/run/docker.sock,)))
export DOCKER_HOST

NS      := fair
CLUSTER := fair-dev
PGSTAC_DSN ?= postgresql://postgres:postgres@localhost:5432/fair_models
PID_FILE := .port-forward.pids
NVKIND  := $(shell command -v nvkind 2>/dev/null || echo $(HOME)/go/bin/nvkind)

FAIR_LABEL_DOMAIN ?= fair-dev.hotosm.org
export FAIR_LABEL_DOMAIN

IMAGE ?= ghcr.io/hotosm/fair-models/example-unet:v1

.PHONY: cluster-up cluster-down infra-up infra-down \
        port-forward kill-port-forward seed-data \
        stack-register run-example build-image run-example-k8s \
        up teardown help

cluster-up:
	-envsubst '$$FAIR_LABEL_DOMAIN' < kind-config.yaml > .kind-config-resolved.yaml
	-$(NVKIND) cluster create --name $(CLUSTER) --config-template .kind-config-resolved.yaml
	@rm -f .kind-config-resolved.yaml
	@kubectl apply -f - <<< '{"apiVersion":"node.k8s.io/v1","kind":"RuntimeClass","handler":"nvidia","metadata":{"name":"nvidia"}}' 2>/dev/null || true

cluster-down:
	kind delete cluster --name $(CLUSTER)

infra-up:
	@kubectl create ns $(NS) --dry-run=client -o yaml | kubectl apply -f -
	@kubectl label node -l $(FAIR_LABEL_DOMAIN)/role=train nvidia.com/gpu.present=true --overwrite >/dev/null 2>/dev/null || true
	envsubst '$$FAIR_LABEL_DOMAIN' < postgres/statefulset.yaml | kubectl apply -n $(NS) -f postgres/service.yaml -f -
	kubectl rollout status -n $(NS) statefulset/postgres --timeout=120s
	helmfile apply
	kubectl rollout status -n $(NS) deployment/stac-stac --timeout=300s

infra-down:
	-helmfile destroy
	-kubectl delete ns nvidia --ignore-not-found
	-kubectl delete -n $(NS) -f postgres/

port-forward: kill-port-forward
	@mkdir -p .pf-logs
	@while read svc ports; do \
		( while true; do \
			kubectl port-forward -n $(NS) svc/$$svc $$ports >>.pf-logs/$$svc.log 2>&1 || true; \
			sleep 2; \
		done ) & echo $$! >> $(PID_FILE); \
	done < ports.conf
	@while read -r svc ports; do \
		for i in $$(seq 1 30); do nc -z localhost $${ports%%:*} 2>/dev/null && break; sleep 1; done; \
	done < ports.conf

kill-port-forward:
	@if test -f $(PID_FILE); then \
		while read pid; do \
			kill -- -$$pid 2>/dev/null || kill $$pid 2>/dev/null || true; \
		done < $(PID_FILE); \
		rm -f $(PID_FILE); \
	fi
	@rm -rf .pf-logs

seed-data:
	@uv run --with minio python -c "from pathlib import Path; from minio import Minio; c = Minio('localhost:9000', 'minioadmin', 'minioadmin', secure=False); root = Path('../../data/sample'); files = [f for f in root.rglob('*') if f.is_file()]; print(f'Uploading {len(files)} files to fair-data/sample/'); [c.fput_object('fair-data', f'sample/{f.relative_to(root)}', str(f)) for f in files]; print('Done')"

stack-register:
	@for i in 1 2 3 4 5; do \
		uv run zenml connect --url http://localhost:8080 --username default --password '' --no-verify-ssl && break; \
		echo "zenml connect: attempt $$i failed, retrying in 5s..." >&2; sleep 5; \
	done
	-uv run zenml stack import dev -f ../../stacks/local.yaml --ignore-version-mismatch
	-envsubst '$$FAIR_LABEL_DOMAIN' < ../../stacks/k8s.yaml > .k8s-stack-resolved.yaml
	-uv run zenml stack import k8s -f .k8s-stack-resolved.yaml --ignore-version-mismatch
	@rm -f .k8s-stack-resolved.yaml
	-uv run zenml stack set dev

run-example:
	cd ../.. && uv run python examples/unet/run.py all --stac-api-url http://localhost:8082 --dsn $(PGSTAC_DSN)

build-image:
	cd ../.. && docker buildx build -f models/example_unet/Dockerfile -t $(IMAGE) --platform linux/amd64 --load --no-cache .
	kind load docker-image $(IMAGE) --name $(CLUSTER) --nodes $(CLUSTER)-worker,$(CLUSTER)-worker2

run-example-k8s:
	cd ../.. && uv run zenml stack set k8s
	cd ../.. && AWS_ENDPOINT_URL=http://localhost:9000 uv run python examples/unet/run.py all \
		--stac-api-url http://localhost:8082 --dsn $(PGSTAC_DSN)

up: cluster-up infra-up port-forward seed-data stack-register
teardown: kill-port-forward infra-down cluster-down

help:
	@echo "cluster-up          Create kind cluster"
	@echo "cluster-down        Delete kind cluster"
	@echo "infra-up            Deploy postgres + helmfile apply"
	@echo "infra-down          Helmfile destroy + remove postgres"
	@echo "port-forward        Forward services to localhost (auto-reconnect)"
	@echo "kill-port-forward   Stop port-forwards"
	@echo "seed-data           Upload sample data to MinIO"
	@echo "stack-register      Register ZenML stacks"
	@echo "build-image         Build & load Docker image into kind workers"
	@echo "run-example         Run UNet example E2E (local stack)"
	@echo "run-example-k8s     Run UNet example E2E (k8s stack)"
	@echo "up                  Full setup"
	@echo "teardown            Full teardown"
