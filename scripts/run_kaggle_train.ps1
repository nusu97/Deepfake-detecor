param(
  [string]$Dataset = 'sanikatiwarekar/deep-fake-detection-dfd-entire-original-dataset',
  [int]$MaxVideosPerClass = 200,
  [int]$FramesPerVideo = 3,
  [int]$Epochs = 10,
  [string]$OutModel = 'models\\cnn_deepfake.keras'
)

$ErrorActionPreference = 'Stop'

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $repoRoot

$py = Join-Path $repoRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $py)) {
  throw "Missing venv python at $py. Create venv + install requirements first."
}

# Prefer repo-local credentials
$repoKaggleDir = Join-Path $repoRoot '.kaggle'
$repoKaggleJson = Join-Path $repoKaggleDir 'kaggle.json'
if (Test-Path $repoKaggleJson) {
  $env:KAGGLE_CONFIG_DIR = $repoKaggleDir
}

$kaggleExe = Join-Path $repoRoot '.venv\Scripts\kaggle.exe'
if (-not (Test-Path $kaggleExe)) {
  & $py -m pip install -q kaggle
}
if (-not (Test-Path $kaggleExe)) {
  throw "kaggle CLI not found in venv. Try: $py -m pip install kaggle"
}

$rawDir = Join-Path $repoRoot 'data\kaggle_raw'
New-Item -ItemType Directory -Force -Path $rawDir | Out-Null

$dfdOriginal = Join-Path $rawDir 'DFD_original sequences'
$dfdManip = Join-Path $rawDir 'DFD_manipulated_sequences'
$alreadyExtracted = (Test-Path $dfdOriginal) -and (Test-Path $dfdManip)

if (-not $alreadyExtracted) {
  Write-Host "Downloading Kaggle dataset: $Dataset" -ForegroundColor Cyan
  & $kaggleExe datasets download -d $Dataset -p $rawDir --unzip
}

if (-not ((Test-Path $dfdOriginal) -and (Test-Path $dfdManip))) {
  throw "Expected DFD folders not found under $rawDir. Inspect extraction output."
}

# Ensure OpenCV is available, but do not upgrade NumPy (TensorFlow requires numpy<2)
& $py -c "import cv2" 2>$null
if ($LASTEXITCODE -ne 0) {
  & $py -m pip install -q --no-deps opencv-python
}

Write-Host "Extracting frames -> train/test" -ForegroundColor Cyan
& $py .\scripts\prepare_dfd_videos.py --input-root $rawDir --output . --clean --max-videos-per-class $MaxVideosPerClass --frames-per-video $FramesPerVideo

Write-Host "Training model (epochs=$Epochs) -> $OutModel" -ForegroundColor Cyan
& $py .\scripts\train.py --train-dir train --test-dir test --epochs $Epochs --out $OutModel

Write-Host "Done. Model saved to $OutModel" -ForegroundColor Green
