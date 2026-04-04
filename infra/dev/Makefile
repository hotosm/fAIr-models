SHELL := /bin/bash
.DEFAULT_GOAL := status

DOCKER_HOST ?= $(if $(wildcard /var/run/docker.sock),,\
  $(if $(wildcard $(HOME)/.colima/default/docker.sock),unix://$(HOME)/.colima/default/docker.sock,\
  $(if $(wildcard $(HOME)/.docker/run/docker.sock),unix://$(HOME)/.docker/run/docker.sock,)))
export DOCKER_HOST

NS       := fair
CLUSTER  := local-dev-fair
PGSTAC_DSN := postgresql://postgres:postgres@localhost:5432/fair_models
PID_FILE := .port-forward.pids
NVKIND   := $(shell command -v nvkind 2>/dev/null || echo $(HOME)/go/bin/nvkind)

CLUSTER_EXISTS := $(shell kind get clusters 2>/dev/null | grep -qx $(CLUSTER) && echo 1)
INFRA_RUNNING  := $(shell kubectl get statefulset/postgres -n $(NS) -o jsonpath='{.status.readyReplicas}' 2>/dev/null)
PF_RUNNING     := $(shell test -f $(PID_FILE) && kill -0 $$(head -1 $(PID_FILE)) 2>/dev/null && echo 1)

.PHONY: up down tear status run-example run-example-k8s

up: _ensure-cluster _ensure-infra _ensure-port-forward _seed-data _stack-register
	@echo "Ready."

down: _kill-port-forward
	@echo "Port-forwards stopped. 'make up' to resume."

tear: _kill-port-forward
	-helmfile destroy
	-kubectl delete ns nvidia --ignore-not-found
	-kubectl delete -n $(NS) -f postgres/
	kind delete cluster --name $(CLUSTER)

status:
	@echo "Cluster:       $(if $(CLUSTER_EXISTS),running,not found)"
	@echo "Infra:         $(if $(INFRA_RUNNING),running,not ready)"
	@echo "Port-forwards: $(if $(PF_RUNNING),active,inactive)"
	@echo ""
	@if [ "$(CLUSTER_EXISTS)" = "1" ]; then \
		echo "Nodes:"; kubectl get nodes -o wide --no-headers 2>/dev/null | sed 's/^/  /'; echo ""; \
		echo "Pods ($(NS)):"; kubectl get pods -n $(NS) --no-headers 2>/dev/null | sed 's/^/  /'; echo ""; \
		echo "Ports:"; while read svc ports; do \
			local_port=$${ports%%:*}; \
			if nc -z localhost $$local_port 2>/dev/null; then \
				echo "  $$svc localhost:$$local_port ok"; \
			else \
				echo "  $$svc localhost:$$local_port down"; \
			fi; \
		done < ports.conf; \
	fi

run-example:
	cd ../.. && uv run python examples/unet/run.py all --stac-api-url http://localhost:8082 --dsn $(PGSTAC_DSN)

run-example-k8s:
	cd ../.. && uv run zenml stack set k8s
	cd ../.. && AWS_ENDPOINT_URL=http://localhost:9000 uv run python examples/unet/run.py all \
		--stac-api-url http://localhost:8082 --dsn $(PGSTAC_DSN)

_ensure-cluster:
ifeq ($(CLUSTER_EXISTS),)
	-$(NVKIND) cluster create --name $(CLUSTER) --config-template kind-config.yaml
	@kubectl apply -f - <<< '{"apiVersion":"node.k8s.io/v1","kind":"RuntimeClass","handler":"nvidia","metadata":{"name":"nvidia"}}' 2>/dev/null || true
endif

_ensure-infra:
ifneq ($(INFRA_RUNNING),1)
	@kubectl create ns $(NS) --dry-run=client -o yaml | kubectl apply -f -
	@kubectl apply -n $(NS) -f postgres/service.yaml -f postgres/statefulset.yaml
	@kubectl rollout status -n $(NS) statefulset/postgres --timeout=120s
	@helmfile apply
	@kubectl rollout status -n $(NS) deployment/stac-stac --timeout=300s
endif

_ensure-port-forward:
ifneq ($(PF_RUNNING),1)
	@$(MAKE) --no-print-directory _port-forward
endif

_port-forward: _kill-port-forward
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

_kill-port-forward:
	@if test -f $(PID_FILE); then \
		while read pid; do \
			kill -- -$$pid 2>/dev/null || kill $$pid 2>/dev/null || true; \
		done < $(PID_FILE); \
		rm -f $(PID_FILE); \
	fi
	@rm -rf .pf-logs

_seed-data:
	@uv run --with minio python scripts/seed_data.py

_stack-register:
	@for i in 1 2 3 4 5; do \
		uv run zenml connect --url http://localhost:8080 --username default --password '' --no-verify-ssl && break; \
		echo "zenml connect: attempt $$i failed, retrying in 5s..." >&2; sleep 5; \
	done
	-uv run zenml stack import dev -f ../../stacks/local.yaml --ignore-version-mismatch
	-uv run zenml stack import k8s -f ../../stacks/k8s.yaml --ignore-version-mismatch
	-uv run zenml stack set dev
