#!/usr/bin/env bash
# Run YOLO v8 v2 in-container smoke tests. Default: /workspace/data/sample.
# Base images come from GHCR (no local build of fAIr-utilities needed):
#   CPU: ghcr.io/hotosm/fair-utilities-yolo:cpu-latest  (default)
#   GPU: ghcr.io/hotosm/fair-utilities-yolo:gpu-latest
# Windows PowerShell: use run_docker_tests.ps1 (Bash-style VAR=1 ./script.sh does not work in PowerShell).
# To pass --epochs / --batch-size / --confidence / --pc / a custom --dataset-root, use docker run manually.
set -euo pipefail

REPO_ROOT="${1:-$(pwd)}"
IMAGE="${2:-yolo-v8-v2:gpu}"
BUILD_IMAGE="${BUILD_IMAGE:-0}"
CPU_ONLY="${CPU_ONLY:-0}"

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
    --dataset-root /workspace/data/sample

echo "Done."
