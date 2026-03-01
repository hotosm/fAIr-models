#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${1:-$(pwd)}"
IMAGE="${2:-yolo-v8-v1:latest}"
BUILD_IMAGE="${BUILD_IMAGE:-0}"
CPU_ONLY="${CPU_ONLY:-0}"

if [[ "$BUILD_IMAGE" == "1" ]]; then
  echo "Building image $IMAGE ..."
  docker build -t "$IMAGE" -f "$REPO_ROOT/models/yolo_v8_v1/Dockerfile" "$REPO_ROOT"
fi

GPU_ARGS=()
if [[ "$CPU_ONLY" != "1" ]]; then
  GPU_ARGS=(--gpus all)
fi

echo "Running container smoke tests..."
docker run --rm --shm-size 1g "${GPU_ARGS[@]}" \
  -v "$REPO_ROOT:/workspace" \
  "$IMAGE" \
  python /workspace/models/yolo_v8_v1/tests/inside_container_smoke_test.py \
    --dataset-root /workspace/data/sample

echo "Done."
