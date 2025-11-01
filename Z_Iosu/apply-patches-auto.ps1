# Script maestro para aplicar parches de llama.cpp resolviendo conflictos automáticamente
# Uso: powershell -ExecutionPolicy Bypass -File Z_Iosu\apply-patches-auto.ps1

$ErrorActionPreference = "Stop"

Write-Host "======================================" -ForegroundColor Cyan
Write-Host "Aplicando parches de llama.cpp" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan
Write-Host ""

$ollamaRoot = Get-Location
$vendorDir = "llama\vendor"

# Función para resolver conflicto en ggml-impl.h
function Resolve-GgmlImplConflict {
    $file = "$vendorDir\ggml\src\ggml-impl.h"
    
    if (Test-Path $file) {
        $content = Get-Content $file -Raw
        if ($content -match '<<<<<<< HEAD') {
            Write-Host "Resolviendo conflicto en ggml-impl.h..." -ForegroundColor Yellow
            
            Copy-Item $file "$file.backup" -Force
            
            $resolved = $content -replace '<<<<<<< HEAD\r?\n', ''
            $resolved = $resolved -replace '=======\r?\n', ''
            $resolved = $resolved -replace '>>>>>>> GPU discovery enhancements\r?\n', ''
            
            Set-Content -Path $file -Value $resolved -NoNewline
            
            Push-Location $vendorDir
            git add ggml\src\ggml-impl.h
            git am --continue
            Pop-Location
            
            Write-Host "✓ Conflicto en ggml-impl.h resuelto" -ForegroundColor Green
        }
    }
}

# Función para resolver conflicto en ggml-cuda.cu
function Resolve-CudaConflict {
    $file = "$vendorDir\ggml\src\ggml-cuda\ggml-cuda.cu"
    
    if (Test-Path $file) {
        $content = Get-Content $file -Raw
        if ($content -match '<<<<<<< HEAD') {
            Write-Host "Resolviendo conflicto en ggml-cuda.cu..." -ForegroundColor Yellow
            
            Copy-Item $file "$file.backup" -Force
            
            $resolved = $content -replace '<<<<<<< HEAD\r?\n', ''
            $resolved = $resolved -replace '=======\r?\n', ''
            $resolved = $resolved -replace '>>>>>>> ggml: Enable resetting backend devices\r?\n', ''
            
            Set-Content -Path $file -Value $resolved -NoNewline
            
            Push-Location $vendorDir
            git add ggml\src\ggml-cuda\ggml-cuda.cu
            git am --skip  # Parche ya aplicado
            Pop-Location
            
            Write-Host "✓ Conflicto en ggml-cuda.cu resuelto (parche skipped)" -ForegroundColor Green
        }
    }
}

# Aplicar parches con manejo de conflictos
function Apply-PatchesWithConflicts {
    $maxAttempts = 10
    $attempt = 0
    
    while ($attempt -lt $maxAttempts) {
        $attempt++
        
        Write-Host ""
        Write-Host "Intento $attempt : Ejecutando make -f Makefile.sync clean apply-patches" -ForegroundColor Cyan
        
        # Ejecutar make y capturar output
        $output = & bash -c "make -f Makefile.sync clean apply-patches 2>&1"
        $exitCode = $LASTEXITCODE
        
        if ($exitCode -eq 0) {
            Write-Host ""
            Write-Host "✓ Todos los parches aplicados exitosamente" -ForegroundColor Green
            return $true
        }
        
        # Verificar qué conflicto ocurrió
        if ($output -match "ggml/src/ggml-impl.h") {
            Resolve-GgmlImplConflict
        }
        elseif ($output -match "ggml/src/ggml-cuda/ggml-cuda.cu") {
            Resolve-CudaConflict
        }
        else {
            Write-Host ""
            Write-Host "✗ Error desconocido aplicando parches" -ForegroundColor Red
            Write-Host $output -ForegroundColor Gray
            return $false
        }
    }
    
    Write-Host "✗ Demasiados intentos, revisa manualmente" -ForegroundColor Red
    return $false
}

# Formatear parches primero
Write-Host "Formateando parches existentes..." -ForegroundColor Cyan
& bash -c "make -f Makefile.sync format-patches"

Write-Host ""
Write-Host "Iniciando aplicación de parches..." -ForegroundColor Cyan

if (Apply-PatchesWithConflicts) {
    Write-Host ""
    Write-Host "======================================" -ForegroundColor Green
    Write-Host "✓ Proceso completado exitosamente" -ForegroundColor Green
    Write-Host "======================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Los parches de llama.cpp han sido aplicados." -ForegroundColor White
    Write-Host "Ahora puedes compilar Ollama con:" -ForegroundColor White
    Write-Host "  powershell -ExecutionPolicy Bypass -File Z_Iosu\scripts\build_windows.ps1 buildCPU buildVulkan gatherDependencies buildOllama" -ForegroundColor Yellow
    exit 0
} else {
    Write-Host ""
    Write-Host "✗ Hubo errores aplicando los parches" -ForegroundColor Red
    exit 1
}
