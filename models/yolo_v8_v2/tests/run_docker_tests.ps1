# YOLO v8 v2 Docker smoke test — PowerShell (Windows).
# Same behavior as run_docker_tests.sh.
#
# Base images are pulled from GHCR (no local build needed):
#   CPU: ghcr.io/hotosm/fair-utilities-yolo:cpu-latest  (default)
#   GPU: ghcr.io/hotosm/fair-utilities-yolo:gpu-latest
#
# Bash syntax "VAR=1 ./script.sh" does not work in PowerShell. Use either:
#   - This script, or
#   - $env:BUILD_IMAGE = "1"; $env:CPU_ONLY = "1"; .\models\yolo_v8_v2\tests\run_docker_tests.ps1
# Optional smoke args:
#   - $env:SMOKE_DATASET_ROOT (default: /workspace/data/sample)
#   - $env:SMOKE_EPOCHS (default: 10)
#   - $env:SMOKE_BATCH_SIZE (default: 16)
#   - $env:SMOKE_CONFIDENCE (default: 0.5)
#   - $env:SMOKE_PC (default: 2.0)
#
# Examples (run from fAIr-models repo root):
#   .\models\yolo_v8_v2\tests\run_docker_tests.ps1
#   $env:BUILD_IMAGE = "1"; $env:CPU_ONLY = "1"; .\models\yolo_v8_v2\tests\run_docker_tests.ps1
#   .\models\yolo_v8_v2\tests\run_docker_tests.ps1 -RepoRoot "E:\path\to\fAIr-models" -Image "yolo-v8-v2:cpu"

#Requires -Version 5.1
param(
    [string] $RepoRoot = (Get-Location).Path,
    [string] $Image = ""
)

$ErrorActionPreference = "Stop"

function Invoke-NativeOrThrow {
    param([scriptblock] $Command)
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE."
    }
}

$buildImage = if ($null -ne $env:BUILD_IMAGE) { $env:BUILD_IMAGE } else { "0" }
$cpuOnly = if ($null -ne $env:CPU_ONLY) { $env:CPU_ONLY } else { "0" }
$smokeDatasetRoot = if ($null -ne $env:SMOKE_DATASET_ROOT) { $env:SMOKE_DATASET_ROOT } else { "/workspace/data/sample" }
$smokeEpochs = if ($null -ne $env:SMOKE_EPOCHS) { $env:SMOKE_EPOCHS } else { "10" }
$smokeBatchSize = if ($null -ne $env:SMOKE_BATCH_SIZE) { $env:SMOKE_BATCH_SIZE } else { "16" }
$smokeConfidence = if ($null -ne $env:SMOKE_CONFIDENCE) { $env:SMOKE_CONFIDENCE } else { "0.5" }
$smokePc = if ($null -ne $env:SMOKE_PC) { $env:SMOKE_PC } else { "2.0" }

if (-not $Image) {
    if ($cpuOnly -eq "1") {
        $Image = "yolo-v8-v2:cpu"
    }
    else {
        $Image = "yolo-v8-v2:gpu"
    }
}

if ($buildImage -eq "1") {
    if ($cpuOnly -eq "1") {
        $Image = "yolo-v8-v2:cpu"
        Write-Host "Building image $Image (base: ghcr.io/hotosm/fair-utilities-yolo:cpu-latest) ..."
        Invoke-NativeOrThrow { docker build -t $Image -f "$RepoRoot/models/yolo_v8_v2/Dockerfile" $RepoRoot }
    }
    else {
        $Image = "yolo-v8-v2:gpu"
        Write-Host "Building image $Image (base: ghcr.io/hotosm/fair-utilities-yolo:gpu-latest) ..."
        Invoke-NativeOrThrow { docker build --build-arg BASE_IMAGE=ghcr.io/hotosm/fair-utilities-yolo:gpu-latest -t $Image -f "$RepoRoot/models/yolo_v8_v2/Dockerfile" $RepoRoot }
    }
}

if ($cpuOnly -ne "1") {
    Write-Host "Running container smoke tests..."
    Invoke-NativeOrThrow { docker run --rm --gpus all --shm-size 1g -v "${RepoRoot}:/workspace" $Image python /workspace/models/yolo_v8_v2/tests/inside_container_smoke_test.py --dataset-root $smokeDatasetRoot --epochs $smokeEpochs --batch-size $smokeBatchSize --confidence $smokeConfidence --pc $smokePc }
}
else {
    Write-Host "Running container smoke tests..."
    Invoke-NativeOrThrow { docker run --rm --shm-size 1g -v "${RepoRoot}:/workspace" $Image python /workspace/models/yolo_v8_v2/tests/inside_container_smoke_test.py --dataset-root $smokeDatasetRoot --epochs $smokeEpochs --batch-size $smokeBatchSize --confidence $smokeConfidence --pc $smokePc }
}

Write-Host "Done."
