# Temporary patch for CMakeLists.txt to disable forced GGML_CPU_ALL_VARIANTS
# Usage: 
#   .\patch-cmake-cpu.ps1         # Apply patch
#   .\patch-cmake-cpu.ps1 -Revert # Revert patch

param([switch]$Revert)

$rootDir = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$cmakeFile = "$rootDir\CMakeLists.txt"

if ($Revert) {
    Write-Host "[PATCH] Reverting CMakeLists.txt..." -ForegroundColor Yellow
    Push-Location $rootDir
    try {
        git checkout CMakeLists.txt *>$null
    } catch {
        # Ignore errors from git checkout
    }
    Pop-Location
    Write-Host "[PATCH] Reverted" -ForegroundColor Green
    exit 0
}

$ErrorActionPreference = "Stop"

Write-Host "[PATCH] Patching CMakeLists.txt (disable forced CPU variants)..." -ForegroundColor Cyan

# Backup
if (!(Test-Path "$cmakeFile.bak")) {
    Copy-Item $cmakeFile "$cmakeFile.bak"
}

# Read and patch
$content = Get-Content $cmakeFile
$patched = $false

for ($i = 0; $i -lt $content.Count; $i++) {
    # Find the line that forces GGML_CPU_ALL_VARIANTS=ON
    if ($content[$i] -match '^\s*set\(GGML_CPU_ALL_VARIANTS ON\)\s*$') {
        $content[$i] = "# PATCHED: " + $content[$i]
        $patched = $true
        Write-Host "  Line $($i+1): GGML_CPU_ALL_VARIANTS commented out" -ForegroundColor Gray
    }
    # Find the line that sets GGML_BACKEND_DL=ON
    if ($content[$i] -match '^\s*set\(GGML_BACKEND_DL ON\)\s*$') {
        $content[$i] = "# PATCHED: " + $content[$i] + "`nset(GGML_BACKEND_DL OFF)"
        $patched = $true
        Write-Host "  Line $($i+1): GGML_BACKEND_DL changed to OFF" -ForegroundColor Gray
    }
}

if ($patched) {
    Set-Content -Path $cmakeFile -Value $content
    Write-Host "[PATCH] Applied successfully" -ForegroundColor Green
} else {
    Write-Host "[PATCH] Already patched or pattern not found" -ForegroundColor Yellow
}
