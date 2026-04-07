# RAMP Docker smoke test — PowerShell (Windows).
# Same behavior as run_docker_tests.sh.
#
# Base images are pulled from GHCR (no local build needed):
#   CPU: ghcr.io/hotosm/fair-utilities-ramp:cpu-latest  (default)
#   GPU: ghcr.io/hotosm/fair-utilities-ramp:gpu-latest
#
# Bash syntax "VAR=1 ./script.sh" does not work in PowerShell. Use either:
#   - This script, or
#   - $env:BUILD_IMAGE = "1"; $env:CPU_ONLY = "1"; .\models\ramp\tests\run_docker_tests.ps1
# Optional smoke args:
#   - $env:SMOKE_DATASET_ROOT (default: /workspace/data/sample)
#   - $env:SMOKE_EPOCHS (default: 2)
#   - $env:SMOKE_BATCH_SIZE (default: 4)
#   - $env:SMOKE_BACKBONE (default: efficientnetb0)
#
# Examples (run from fAIr-models repo root):
#   .\models\ramp\tests\run_docker_tests.ps1
#   $env:BUILD_IMAGE = "1"; $env:CPU_ONLY = "1"; .\models\ramp\tests\run_docker_tests.ps1
#   .\models\ramp\tests\run_docker_tests.ps1 -RepoRoot "E:\path\to\fAIr-models" -Image "ramp-v1:cpu"

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
$smokeEpochs = if ($null -ne $env:SMOKE_EPOCHS) { $env:SMOKE_EPOCHS } else { "2" }
$smokeBatchSize = if ($null -ne $env:SMOKE_BATCH_SIZE) { $env:SMOKE_BATCH_SIZE } else { "4" }
$smokeBackbone = if ($null -ne $env:SMOKE_BACKBONE) { $env:SMOKE_BACKBONE } else { "efficientnetb0" }

if (-not $Image) {
    if ($cpuOnly -eq "1") {
        $Image = "ramp-v1:cpu"
    }
    else {
        $Image = "ramp-v1:gpu"
    }
}

if ($buildImage -eq "1") {
    if ($cpuOnly -eq "1") {
        $Image = "ramp-v1:cpu"
        Write-Host "Building image $Image (base: ghcr.io/hotosm/fair-utilities-ramp:cpu-latest) ..."
        # No --build-arg needed — the Dockerfile default already points at the GHCR CPU image.
        Invoke-NativeOrThrow { docker build -t $Image -f "$RepoRoot/models/ramp/Dockerfile" $RepoRoot }
    }
    else {
        $Image = "ramp-v1:gpu"
        Write-Host "Building image $Image (base: ghcr.io/hotosm/fair-utilities-ramp:gpu-latest) ..."
        Invoke-NativeOrThrow { docker build --build-arg BASE_IMAGE=ghcr.io/hotosm/fair-utilities-ramp:gpu-latest -t $Image -f "$RepoRoot/models/ramp/Dockerfile" $RepoRoot }
    }
}

if ($cpuOnly -ne "1") {
    Write-Host "Running container smoke tests..."
    Invoke-NativeOrThrow { docker run --rm --gpus all -v "${RepoRoot}:/workspace" $Image python /workspace/models/ramp/tests/inside_container_smoke_test.py --dataset-root $smokeDatasetRoot --epochs $smokeEpochs --batch-size $smokeBatchSize --backbone $smokeBackbone }
}
else {
    Write-Host "Running container smoke tests..."
    Invoke-NativeOrThrow { docker run --rm -v "${RepoRoot}:/workspace" $Image python /workspace/models/ramp/tests/inside_container_smoke_test.py --dataset-root $smokeDatasetRoot --epochs $smokeEpochs --batch-size $smokeBatchSize --backbone $smokeBackbone }
}

Write-Host "Done."
