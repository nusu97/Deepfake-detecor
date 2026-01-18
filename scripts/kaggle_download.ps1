$ErrorActionPreference = 'Stop'

param(
  [Parameter(Mandatory=$true)]
  [string]$Dataset,

  [string]$OutDir = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
)

Write-Host "Repo root: $OutDir" -ForegroundColor Cyan

# Requires Kaggle API:
# 1) pip install kaggle
# 2) Place kaggle.json at: $env:USERPROFILE\.kaggle\kaggle.json
#    (or set KAGGLE_USERNAME / KAGGLE_KEY)

if (-not (Get-Command kaggle -ErrorAction SilentlyContinue)) {
  $venvKaggle = Join-Path $OutDir '.venv\Scripts\kaggle.exe'
  if (Test-Path $venvKaggle) {
    Set-Alias -Name kaggle -Value $venvKaggle
  } else {
    throw "kaggle CLI not found. Install it with:  pip install kaggle"
  }
}

$rawDir = Join-Path $OutDir 'data\kaggle_raw'
New-Item -ItemType Directory -Force -Path $rawDir | Out-Null

# Support repo-local credentials: <repo>\.kaggle\kaggle.json
$repoKaggleDir = Join-Path $OutDir '.kaggle'
$repoKaggleJson = Join-Path $repoKaggleDir 'kaggle.json'
if (Test-Path $repoKaggleJson) {
  $env:KAGGLE_CONFIG_DIR = $repoKaggleDir
  Write-Host "Using repo-local Kaggle credentials from $repoKaggleJson" -ForegroundColor Green
}

Write-Host "Downloading Kaggle dataset: $Dataset" -ForegroundColor Cyan
kaggle datasets download -d $Dataset -p $rawDir --unzip

Write-Host "Downloaded to: $rawDir" -ForegroundColor Green
Write-Host "Next: run scripts\\prepare_kaggle_dataset.py pointing at the extracted folder containing class subfolders." -ForegroundColor Green
