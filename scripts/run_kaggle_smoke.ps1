param(
  [Parameter(Mandatory=$true)]
  [string]$Dataset,

  [int]$MaxPerClass = 200,
  [int]$Epochs = 1
)

$ErrorActionPreference = 'Stop'

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $repoRoot

$py = Join-Path $repoRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $py)) {
  throw "Missing venv python at $py. Create venv + install requirements first."
}

$kaggleExe = Join-Path $repoRoot '.venv\Scripts\kaggle.exe'
if (-not (Test-Path $kaggleExe)) {
  & $py -m pip install -q kaggle
}
if (-not (Test-Path $kaggleExe)) {
  throw "kaggle CLI not found in venv. Try: $py -m pip install kaggle"
}

# Prefer repo-local credentials: <repo>\.kaggle\kaggle.json
$repoKaggleDir = Join-Path $repoRoot '.kaggle'
$repoKaggleJson = Join-Path $repoKaggleDir 'kaggle.json'
if (Test-Path $repoKaggleJson) {
  $env:KAGGLE_CONFIG_DIR = $repoKaggleDir
}

$cfg = Join-Path $env:USERPROFILE '.kaggle\kaggle.json'
$hasCfg = (Test-Path $cfg) -or (Test-Path $repoKaggleJson)
$hasEnv = ($env:KAGGLE_USERNAME -and $env:KAGGLE_KEY)
if (-not ($hasCfg -or $hasEnv)) {
  Write-Host "Kaggle credentials not found." -ForegroundColor Yellow
  Write-Host "Create them in Kaggle -> Account -> API -> Create New API Token" -ForegroundColor Yellow
  Write-Host "Then place kaggle.json at either:" -ForegroundColor Yellow
  Write-Host "  $cfg" -ForegroundColor Yellow
  Write-Host "  $repoKaggleJson" -ForegroundColor Yellow
  throw "Missing Kaggle credentials"
}

$rawDir = Join-Path $repoRoot 'data\kaggle_raw'
New-Item -ItemType Directory -Force -Path $rawDir | Out-Null

# Skip re-download if it looks like data is already extracted.
$alreadyExtracted = (Test-Path (Join-Path $rawDir 'DFD_original sequences')) -or (Test-Path (Join-Path $rawDir 'DFD_manipulated_sequences'))
if (-not $alreadyExtracted) {
  Write-Host "Downloading Kaggle dataset (this may take a while): $Dataset" -ForegroundColor Cyan
  & $kaggleExe datasets download -d $Dataset -p $rawDir --unzip
} else {
  Write-Host "Kaggle data already present under $rawDir; skipping download." -ForegroundColor Green
}

# Special handling: this dataset contains videos, not images.
$dfdOriginal = Join-Path $rawDir 'DFD_original sequences'
$dfdManip = Join-Path $rawDir 'DFD_manipulated_sequences'
if ((Test-Path $dfdOriginal) -and (Test-Path $dfdManip)) {
  Write-Host "Detected DFD video dataset. Extracting frames to build train/test images..." -ForegroundColor Cyan
  # Ensure OpenCV is available, but do not let pip upgrade NumPy (TensorFlow requires numpy<2).
  & $py -c "import cv2" 2>$null
  if ($LASTEXITCODE -ne 0) {
    & $py -m pip install -q --no-deps opencv-python
  }
  & $py .\scripts\prepare_dfd_videos.py --input-root $rawDir --output . --clean --max-videos-per-class $MaxPerClass --frames-per-video 2

  Write-Host "Training (epochs=$Epochs)..." -ForegroundColor Cyan
  & $py .\scripts\train.py --train-dir train --test-dir test --epochs $Epochs --out models\kaggle_smoke.keras

  $sample = Get-ChildItem -Path (Join-Path $repoRoot 'test') -Recurse -File -Include $imageExts -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($null -eq $sample) {
    throw "No test images found after preparation."
  }

  Write-Host "Predicting on: $($sample.FullName)" -ForegroundColor Cyan
  & $py .\scripts\predict.py --model models\kaggle_smoke.keras $sample.FullName
  exit 0
}

# Heuristic: pick a directory under rawDir that has >=2 immediate subdirs and contains images.
$imageExts = @('*.jpg','*.jpeg','*.png','*.bmp','*.webp')
$candidates = @()

$dirs = Get-ChildItem -Path $rawDir -Directory -Recurse -ErrorAction SilentlyContinue
foreach ($d in $dirs) {
  $children = Get-ChildItem -Path $d.FullName -Directory -ErrorAction SilentlyContinue
  if ($children.Count -lt 2) { continue }

  $imgCount = 0
  foreach ($ext in $imageExts) {
    $imgCount += (Get-ChildItem -Path $d.FullName -Recurse -File -Filter $ext -ErrorAction SilentlyContinue).Count
  }
  if ($imgCount -gt 0) {
    $candidates += [pscustomobject]@{ Path=$d.FullName; Images=$imgCount; Classes=$children.Count }
  }
}

if ($candidates.Count -eq 0) {
  throw "Could not auto-detect a class-folder root under $rawDir. Inspect the extracted files and run prepare_kaggle_dataset.py manually."
}

$best = $candidates | Sort-Object Images -Descending | Select-Object -First 1
Write-Host "Using detected dataset root: $($best.Path) (images=$($best.Images), classFolders=$($best.Classes))" -ForegroundColor Green

# Prepare dataset into train/test. If there are more than 2 classes, try to auto-select a likely real/fake pair.
$classDirs = Get-ChildItem -Path $best.Path -Directory -ErrorAction SilentlyContinue
$includeArg = @()
if ($classDirs.Count -ne 2) {
  $names = $classDirs | ForEach-Object { $_.Name }
  $real = $names | Where-Object { $_ -match '(?i)^(real|original|authentic|pristine|raw)$' } | Select-Object -First 1
  if (-not $real) { $real = $names | Where-Object { $_ -match '(?i)(real|original|authentic)' } | Select-Object -First 1 }
  $fake = $names | Where-Object { $_ -match '(?i)^(fake|deepfake|manipulated|edited|synthetic)$' } | Select-Object -First 1
  if (-not $fake) { $fake = $names | Where-Object { $_ -match '(?i)(fake|deepfake|manipulated|synthetic)' } | Select-Object -First 1 }

  if ($real -and $fake) {
    Write-Host "Auto-selecting classes for binary training: $real, $fake" -ForegroundColor Yellow
    $includeArg = @('--include-classes', "$real,$fake")
  } else {
    Write-Host "Detected class folders:" -ForegroundColor Yellow
    $names | ForEach-Object { Write-Host "  $_" -ForegroundColor Yellow }
    throw "Dataset root contains $($classDirs.Count) class folders. Re-run prepare_kaggle_dataset.py with --include-classes classA,classB."
  }
}

Write-Host "Preparing train/test (subset max-per-class=$MaxPerClass)..." -ForegroundColor Cyan
& $py .\scripts\prepare_kaggle_dataset.py --input $best.Path --output . --clean --max-per-class $MaxPerClass @includeArg

Write-Host "Training (epochs=$Epochs)..." -ForegroundColor Cyan
& $py .\scripts\train.py --train-dir train --test-dir test --epochs $Epochs --out models\kaggle_smoke.keras

$sample = Get-ChildItem -Path (Join-Path $repoRoot 'test') -Recurse -File -Include $imageExts -ErrorAction SilentlyContinue | Select-Object -First 1
if ($null -eq $sample) {
  throw "No test images found after preparation."
}

Write-Host "Predicting on: $($sample.FullName)" -ForegroundColor Cyan
& $py .\scripts\predict.py --model models\kaggle_smoke.keras $sample.FullName
