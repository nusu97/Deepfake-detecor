$ErrorActionPreference = 'Stop'

$zipUrl = 'https://techvidvan.s3.amazonaws.com/machine-learning-projects/deep-fake-detection.zip'
$outDir = Join-Path $PSScriptRoot '..'
$outDir = (Resolve-Path $outDir).Path

$zipPath = Join-Path $outDir 'deep-fake-detection.zip'

Write-Host "Downloading dataset zip..." -ForegroundColor Cyan

# Some networks/regions get S3 AccessDenied unless requests look like a browser.
$headers = @{
	'User-Agent' = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
	'Referer'    = 'https://techvidvan.com/tutorials/deepfake-detection-using-cnn/'
}

try {
	Invoke-WebRequest -Uri $zipUrl -OutFile $zipPath -Headers $headers
} catch {
	Write-Warning "Automatic download failed: $($_.Exception.Message)"
}

if (-not (Test-Path $zipPath)) {
	throw "Download did not produce a file: $zipPath"
}

# Validate that we actually got a zip and not an XML AccessDenied.
$firstBytes = [System.IO.File]::ReadAllBytes($zipPath)
if ($firstBytes.Length -lt 4 -or $firstBytes[0] -ne 0x50 -or $firstBytes[1] -ne 0x4B) {
	Write-Warning "The downloaded file is not a valid zip (likely AccessDenied/403)."
	Write-Host "\nFallback: download in your browser and place the zip at:" -ForegroundColor Yellow
	Write-Host "  $zipPath" -ForegroundColor Yellow
	Write-Host "Then run:  .\\scripts\\extract_dataset.ps1" -ForegroundColor Yellow
	exit 2
}

Write-Host "Extracting..." -ForegroundColor Cyan
Expand-Archive -Path $zipPath -DestinationPath $outDir -Force

Write-Host "Done. You should now have train/, test/, prediction/ (depending on zip structure)." -ForegroundColor Green
