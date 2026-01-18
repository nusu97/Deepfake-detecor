$ErrorActionPreference = 'Stop'

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$zipPath = Join-Path $repoRoot 'deep-fake-detection.zip'

if (-not (Test-Path $zipPath)) {
  throw "Zip not found at: $zipPath"
}

# Quick zip signature check (PK)
$firstBytes = [System.IO.File]::ReadAllBytes($zipPath)
if ($firstBytes.Length -lt 4 -or $firstBytes[0] -ne 0x50 -or $firstBytes[1] -ne 0x4B) {
  throw "File does not look like a zip (expected PK header): $zipPath"
}

Write-Host "Extracting $zipPath ..." -ForegroundColor Cyan
Expand-Archive -Path $zipPath -DestinationPath $repoRoot -Force

# Try to locate train/test/prediction anywhere under repo root and move them up.
$targets = @('train','test','prediction')
foreach ($t in $targets) {
  $existing = Join-Path $repoRoot $t
  if (Test-Path $existing) { continue }

  $found = Get-ChildItem -Path $repoRoot -Directory -Recurse -ErrorAction SilentlyContinue | Where-Object { $_.Name -eq $t } | Select-Object -First 1
  if ($null -ne $found) {
    Write-Host "Moving $($found.FullName) -> $existing" -ForegroundColor Cyan
    Move-Item -Path $found.FullName -Destination $existing
  }
}

Write-Host "Done. Expected folders at repo root:" -ForegroundColor Green
foreach ($t in $targets) {
  $p = Join-Path $repoRoot $t
  Write-Host ("  {0} : {1}" -f $t, (Test-Path $p))
}
