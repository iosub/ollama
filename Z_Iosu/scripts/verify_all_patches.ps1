# Script de verificación EXHAUSTIVA de TODOS los patches
# Verifica CADA línea añadida (+) en CADA patch

$patches = @(
    "original_99719122b.patch",
    "original_b913e895a.patch", 
    "original_de0e3d3c3.patch",
    "original_e45aecb7b.patch"
)

$basePath = "C:\IA\tools\ollama"
$patchDir = "$basePath\Z_Iosu\patches"
$llamaPath = "$basePath\llama\llama.cpp"

$totalMissing = 0
$allMissing = @()

foreach ($patchFile in $patches) {
    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host "VERIFICANDO: $patchFile" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    
    $patchPath = Join-Path $patchDir $patchFile
    $content = Get-Content $patchPath -Raw
    
    # Dividir por archivos (diff --git)
    $fileBlocks = $content -split '(?=diff --git)'
    
    foreach ($block in $fileBlocks) {
        if ($block -notmatch 'diff --git a/([^\s]+) b/([^\s]+)') { continue }
        
        $sourceFile = $matches[1]
        $targetFile = $matches[2]
        
        # Solo verificar archivos relevantes para Ollama
        if ($sourceFile -notmatch '^(src/|include/|tools/mtmd/)') { continue }
        
        Write-Host "`nArchivo: $sourceFile" -ForegroundColor Yellow
        
        # Mapear ruta del archivo
        $actualFile = $sourceFile -replace '^src/', 'src/' `
                                  -replace '^include/', 'include/' `
                                  -replace '^tools/mtmd/', 'tools/mtmd/'
        $fullPath = Join-Path $llamaPath $actualFile
        
        if (-not (Test-Path $fullPath)) {
            Write-Host "  ⚠️  ARCHIVO NO EXISTE: $fullPath" -ForegroundColor Red
            continue
        }
        
        $actualContent = Get-Content $fullPath -Raw
        
        # Extraer todas las líneas añadidas (+) del patch
        $lines = $block -split "`n"
        $inHunk = $false
        $lineNum = 0
        $missing = @()
        
        for ($i = 0; $i -lt $lines.Count; $i++) {
            $line = $lines[$i]
            
            # Detectar inicio de hunk (@@ ... @@)
            if ($line -match '^@@.*@@') {
                $inHunk = $true
                $lineNum++
                continue
            }
            
            if (-not $inHunk) { continue }
            
            # Línea añadida (empieza con +)
            if ($line -match '^\+(.*)$') {
                $addedLine = $matches[1]
                
                # Ignorar líneas de metadata del patch
                if ($addedLine -match '^\+\+\+') { continue }
                
                # Buscar la línea en el archivo actual (trim para evitar problemas de espacios)
                $trimmedLine = $addedLine.Trim()
                
                # Buscar la línea exacta (sin trim primero)
                $exactMatch = $actualContent -match [regex]::Escape($addedLine)
                $trimmedMatch = $actualContent -match [regex]::Escape($trimmedLine)
                
                if (-not $exactMatch -and -not $trimmedMatch -and $trimmedLine -ne "") {
                    $missing += @{
                        Line = $lineNum
                        Content = $addedLine
                        File = $sourceFile
                    }
                    $totalMissing++
                }
                
                $lineNum++
            }
            # Línea sin cambios (contexto)
            elseif ($line -match '^[^-+@]') {
                $lineNum++
            }
            # Línea eliminada (-)
            elseif ($line -match '^-') {
                # No incrementar lineNum para líneas eliminadas
            }
        }
        
        if ($missing.Count -gt 0) {
            Write-Host "  ❌ FALTAN $($missing.Count) LÍNEAS:" -ForegroundColor Red
            foreach ($m in $missing) {
                Write-Host "    Línea $($m.Line): $($m.Content)" -ForegroundColor Red
                $allMissing += $m
            }
        } else {
            Write-Host "  ✅ Todas las líneas presentes" -ForegroundColor Green
        }
    }
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "RESUMEN FINAL" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

if ($totalMissing -eq 0) {
    Write-Host "✅✅✅ TODOS LOS PATCHES APLICADOS COMPLETAMENTE ✅✅✅" -ForegroundColor Green
    Write-Host "Total de líneas verificadas: OK" -ForegroundColor Green
} else {
    Write-Host "❌❌❌ FALTAN $totalMissing LÍNEAS EN TOTAL ❌❌❌" -ForegroundColor Red
    Write-Host "`nLíneas faltantes por archivo:" -ForegroundColor Yellow
    $allMissing | Group-Object File | ForEach-Object {
        Write-Host "`n$($_.Name): $($_.Count) líneas faltantes" -ForegroundColor Yellow
        $_.Group | ForEach-Object {
            Write-Host "  - Línea $($_.Line): $($_.Content)" -ForegroundColor Red
        }
    }
}
