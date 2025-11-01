# Script para resolver conflicto en ggml-cuda.cu durante parche 0024
# Uso: powershell -ExecutionPolicy Bypass -File Z_Iosu\patches\0024a-resolve-cuda-conflict.ps1

Write-Host "Resolviendo conflicto en ggml/src/ggml-cuda/ggml-cuda.cu..." -ForegroundColor Cyan

$conflictFile = "llama\vendor\ggml\src\ggml-cuda\ggml-cuda.cu"

if (-not (Test-Path $conflictFile)) {
    Write-Host "Error: No se encuentra $conflictFile" -ForegroundColor Red
    exit 1
}

$content = Get-Content $conflictFile -Raw

if ($content -notmatch '<<<<<<< HEAD') {
    Write-Host "No se detectó conflicto en $conflictFile" -ForegroundColor Green
    exit 0
}

Copy-Item $conflictFile "$conflictFile.backup" -Force
Write-Host "Backup creado" -ForegroundColor Gray

# Resolver manteniendo el código de HEAD (compute_major, driver info, etc.)
$resolved = $content -replace '<<<<<<< HEAD\r?\n', ''
$resolved = $resolved -replace '=======\r?\n', ''
$resolved = $resolved -replace '>>>>>>> ggml: Enable resetting backend devices\r?\n', ''

Set-Content -Path $conflictFile -Value $resolved -NoNewline

Write-Host "Conflicto resuelto. Se mantuvo la información del dispositivo (compute_major, driver, etc.)" -ForegroundColor Green

Set-Location llama\vendor
git add ggml\src\ggml-cuda\ggml-cuda.cu

Write-Host ""
Write-Host "Ejecuta: cd llama\vendor; git am --continue" -ForegroundColor Yellow
