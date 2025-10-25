# Script para actualizar archivos de llama.cpp con el fork LETS-BEE (qwen3vl)
# Commit: de0e3d3c3ce4b394746ade9263736c8edb40260e

$baseUrl = "https://raw.githubusercontent.com/LETS-BEE/llama.cpp/de0e3d3c3ce4b394746ade9263736c8edb40260e"
$targetDir = "C:\IA\tools\ollama\llama\llama.cpp\src"

$archivos = @(
    "llama-arch.h",
    "llama-hparams.h",
    "llama-model.h",
    "llama-model.cpp"
)

Write-Host "===================================================================" -ForegroundColor Cyan
Write-Host "Actualizando archivos desde fork LETS-BEE (commit de0e3d3)" -ForegroundColor Cyan
Write-Host "===================================================================" -ForegroundColor Cyan
Write-Host ""

foreach ($archivo in $archivos) {
    $url = "$baseUrl/src/$archivo"
    $destino = Join-Path $targetDir $archivo
    
    Write-Host "📥 Descargando $archivo..." -ForegroundColor Yellow
    
    try {
        # Hacer backup del archivo actual
        if (Test-Path $destino) {
            $backup = "$destino.backup"
            Copy-Item $destino $backup -Force
            Write-Host "   ✅ Backup creado: $archivo.backup" -ForegroundColor Green
        }
        
        # Descargar nuevo archivo
        Invoke-WebRequest -Uri $url -OutFile $destino -UseBasicParsing
        
        $size = [math]::Round((Get-Item $destino).Length / 1KB, 2)
        Write-Host "   ✅ Descargado: $size KB" -ForegroundColor Green
        
    } catch {
        Write-Host "   ❌ Error: $($_.Exception.Message)" -ForegroundColor Red
        
        # Restaurar backup si falla
        if (Test-Path "$destino.backup") {
            Copy-Item "$destino.backup" $destino -Force
            Write-Host "   ↩️  Backup restaurado" -ForegroundColor Yellow
        }
    }
    
    Write-Host ""
}

Write-Host "===================================================================" -ForegroundColor Cyan
Write-Host "Resumen de archivos actualizados:" -ForegroundColor Cyan
Write-Host "===================================================================" -ForegroundColor Cyan

foreach ($archivo in $archivos) {
    $destino = Join-Path $targetDir $archivo
    if (Test-Path $destino) {
        $size = [math]::Round((Get-Item $destino).Length / 1KB, 2)
        $lines = (Get-Content $destino | Measure-Object -Line).Lines
        Write-Host "✅ $archivo - $size KB - $lines líneas" -ForegroundColor Green
    } else {
        Write-Host "❌ $archivo - NO ENCONTRADO" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "===================================================================" -ForegroundColor Cyan
Write-Host "Archivos de backup disponibles (por si necesitas revertir):" -ForegroundColor Cyan
Write-Host "===================================================================" -ForegroundColor Cyan

Get-ChildItem "$targetDir\*.backup" | ForEach-Object {
    $size = [math]::Round($_.Length / 1KB, 2)
    Write-Host "📦 $($_.Name) - $size KB" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "===================================================================" -ForegroundColor Green
Write-Host "¡Actualización completada!" -ForegroundColor Green
Write-Host "===================================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Siguiente paso: Recompilar Ollama" -ForegroundColor Cyan
Write-Host "  > cd C:\IA\tools\ollama" -ForegroundColor White
Write-Host '  > $env:VERSION = "0.12.6.99"' -ForegroundColor White
Write-Host "  > powershell -ExecutionPolicy Bypass -File Z_Iosu\scripts\build_windows.ps1 buildCPU buildCUDA13 buildVulkan gatherDependencies buildOllama buildApp buildInstaller" -ForegroundColor White
Write-Host ""
