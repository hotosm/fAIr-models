param(
    [string]$RepoRoot  = "E:\On Going Projects\Work Growth\HOTOSM\fAIr_Repos\fAIr-models",
    [string]$Image     = "ramp-v1:gpu",
    [switch]$BuildImage,
    [switch]$CpuOnly
)

$ErrorActionPreference = "Stop"

if ($BuildImage) {
    $buildArg = ""
    if ($CpuOnly) {
        $buildArg = "--build-arg BUILD_TYPE=cpu"
        $Image = "ramp-v1:cpu"
    } else {
        $buildArg = "--build-arg BUILD_TYPE=gpu"
    }
    Write-Host "Building image $Image ..."
    Invoke-Expression "docker build $buildArg -t $Image -f `"$RepoRoot\models\ramp\Dockerfile`" `"$RepoRoot`""
}

$gpuArgs = @()
if (-not $CpuOnly) {
    $gpuArgs = @("--gpus", "all")
}

Write-Host "Running container smoke tests..."
docker run --rm @gpuArgs `
  -v "${RepoRoot}:/workspace" `
  $Image `
  python /workspace/models/ramp/tests/inside_container_smoke_test.py `
    --dataset-root /workspace/data/sample

Write-Host "Done."
