#!/usr/bin/env bash
# Run YOLO v8 v2 in-container smoke tests. Default: /workspace/data/sample.
# Base images come from GHCR (no local build of fAIr-utilities needed):
#   CPU: ghcr.io/hotosm/fair-utilities-yolo:cpu-latest  (default)
#   GPU: ghcr.io/hotosm/fair-utilities-yolo:gpu-latest
# Windows PowerShell: use run_docker_tests.ps1 (Bash-style VAR=1 ./script.sh does not work in PowerShell).
# Optional overrides (export before running):
#   - SMOKE_DATASET_ROOT (default: /workspace/data/sample)
#   - SMOKE_EPOCHS (default: 10)
#   - SMOKE_BATCH_SIZE (default: 16)
#   - SMOKE_CONFIDENCE (default: 0.5)
#   - SMOKE_PC (default: 2.0)
set -euo pipefail

REPO_ROOT="${1:-$(pwd)}"
IMAGE="${2:-yolo-v8-v2:gpu}"
BUILD_IMAGE="${BUILD_IMAGE:-0}"
CPU_ONLY="${CPU_ONLY:-0}"
SMOKE_DATASET_ROOT="${SMOKE_DATASET_ROOT:-/workspace/data/sample}"
SMOKE_EPOCHS="${SMOKE_EPOCHS:-10}"
SMOKE_BATCH_SIZE="${SMOKE_BATCH_SIZE:-16}"
SMOKE_CONFIDENCE="${SMOKE_CONFIDENCE:-0.5}"
SMOKE_PC="${SMOKE_PC:-2.0}"

if [[ "$BUILD_IMAGE" == "1" ]]; then
  if [[ "$CPU_ONLY" == "1" ]]; then
    IMAGE="yolo-v8-v2:cpu"
    # No --build-arg needed; Dockerfile default is ghcr.io/hotosm/fair-utilities-yolo:cpu-latest
    echo "Building image $IMAGE (base: ghcr.io/hotosm/fair-utilities-yolo:cpu-latest) ..."
    docker build -t "$IMAGE" -f "$REPO_ROOT/models/yolo_v8_v2/Dockerfile" "$REPO_ROOT"
  else
    echo "Building image $IMAGE (base: ghcr.io/hotosm/fair-utilities-yolo:gpu-latest) ..."
    docker build --build-arg BASE_IMAGE=ghcr.io/hotosm/fair-utilities-yolo:gpu-latest \
      -t "$IMAGE" -f "$REPO_ROOT/models/yolo_v8_v2/Dockerfile" "$REPO_ROOT"
  fi
fi

GPU_ARGS=()
if [[ "$CPU_ONLY" != "1" ]]; then
  GPU_ARGS=(--gpus all)
fi

echo "Running container smoke tests..."
docker run --rm --shm-size 1g "${GPU_ARGS[@]}" \
  -v "$REPO_ROOT:/workspace" \
  "$IMAGE" \
  python /workspace/models/yolo_v8_v2/tests/inside_container_smoke_test.py \
    --dataset-root "$SMOKE_DATASET_ROOT" \
    --epochs "$SMOKE_EPOCHS" \
    --batch-size "$SMOKE_BATCH_SIZE" \
    --confidence "$SMOKE_CONFIDENCE" \
    --pc "$SMOKE_PC"

echo "Done."
