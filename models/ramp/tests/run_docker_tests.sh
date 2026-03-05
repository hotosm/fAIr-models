#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${1:-$(pwd)}"
IMAGE="${2:-ramp-v1:gpu}"
BUILD_IMAGE="${BUILD_IMAGE:-0}"
CPU_ONLY="${CPU_ONLY:-0}"

if [[ "$BUILD_IMAGE" == "1" ]]; then
  BUILD_ARGS=""
  if [[ "$CPU_ONLY" == "1" ]]; then
    BUILD_ARGS="--build-arg BUILD_TYPE=cpu"
    IMAGE="ramp-v1:cpu"
  else
    BUILD_ARGS="--build-arg BUILD_TYPE=gpu"
  fi
  echo "Building image $IMAGE ..."
  docker build $BUILD_ARGS -t "$IMAGE" -f "$REPO_ROOT/models/ramp/Dockerfile" "$REPO_ROOT"
fi

GPU_ARGS=()
if [[ "$CPU_ONLY" != "1" ]]; then
  GPU_ARGS=(--gpus all)
fi

echo "Running container smoke tests..."
docker run --rm "${GPU_ARGS[@]}" \
  -v "$REPO_ROOT:/workspace" \
  "$IMAGE" \
  python /workspace/models/ramp/tests/inside_container_smoke_test.py \
    --dataset-root /workspace/data/sample

echo "Done."
