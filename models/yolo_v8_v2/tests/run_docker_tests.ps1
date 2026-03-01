param(
    [string]$RepoRoot = "E:\On Going Projects\Work Growth\HOTOSM\fAIr_Repos\fAIr-models",
    [string]$Image = "yolo-v8-v2:latest",
    [switch]$BuildImage,
    [switch]$CpuOnly
)

$ErrorActionPreference = "Stop"

if ($BuildImage) {
    Write-Host "Building image $Image ..."
    docker build -t $Image -f "$RepoRoot\models\yolo_v8_v2\Dockerfile" $RepoRoot
}

$gpuArgs = @()
if (-not $CpuOnly) {
    $gpuArgs = @("--gpus", "all")
}

Write-Host "Running container smoke tests..."
docker run --rm @gpuArgs `
  --shm-size 1g `
  -v "${RepoRoot}:/workspace" `
  $Image `
  python /workspace/models/yolo_v8_v2/tests/inside_container_smoke_test.py `
    --dataset-root /workspace/data/sample

Write-Host "Done."
