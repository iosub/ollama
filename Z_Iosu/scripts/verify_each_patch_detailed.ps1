# Script to verify each patch file individually
$patches = @(
    @{ Name = "99719122b"; File = "Z_Iosu\patches\original_99719122b.patch"; Color = "Cyan" },
    @{ Name = "b913e895a"; File = "Z_Iosu\patches\original_b913e895a.patch"; Color = "Yellow" },
    @{ Name = "de0e3d3c3"; File = "Z_Iosu\patches\original_de0e3d3c3.patch"; Color = "Magenta" },
    @{ Name = "e45aecb7b"; File = "Z_Iosu\patches\original_e45aecb7b.patch"; Color = "Green" }
)

$workspaceRoot = "C:\IA\tools\ollama"
Set-Location $workspaceRoot

foreach ($patch in $patches) {
    Write-Host "`n========================================" -ForegroundColor $patch.Color
    Write-Host "VERIFICANDO PATCH: $($patch.Name)" -ForegroundColor $patch.Color
    Write-Host "========================================" -ForegroundColor $patch.Color
    
    if (-not (Test-Path $patch.File)) {
        Write-Host "ERROR: Patch file not found: $($patch.File)" -ForegroundColor Red
        continue
    }
    
    $patchContent = Get-Content $patch.File -Raw
    
    # Parse the patch to find all files being modified
    $fileBlocks = @{}
    $currentFile = $null
    
    foreach ($line in ($patchContent -split "`n")) {
        # Detect file being modified (--- a/path or +++ b/path)
        if ($line -match '^\+\+\+ b/(.+)$') {
            $currentFile = $matches[1]
            $fileBlocks[$currentFile] = @{
                AddedLines = @()
                RemovedLines = @()
            }
        }
        # Collect added lines (starting with +, but not +++)
        elseif ($currentFile -and $line -match '^\+(?!\+\+)(.*)$') {
            $addedLine = $matches[1]
            # Skip empty lines and lines with only whitespace
            if ($addedLine.Trim() -ne '') {
                $fileBlocks[$currentFile].AddedLines += $addedLine
            }
        }
        # Collect removed lines (starting with -, but not ---)
        elseif ($currentFile -and $line -match '^\-(?!\-\-)(.*)$') {
            $removedLine = $matches[1]
            if ($removedLine.Trim() -ne '') {
                $fileBlocks[$currentFile].RemovedLines += $removedLine
            }
        }
    }
    
    # Now check each file
    $totalMissing = 0
    $totalChecked = 0
    
    foreach ($file in $fileBlocks.Keys) {
        $filePath = Join-Path $workspaceRoot $file
        
        Write-Host "`n  Archivo: $file" -ForegroundColor White
        
        if (-not (Test-Path $filePath)) {
            Write-Host "    [SKIP] Archivo no existe en workspace" -ForegroundColor Gray
            continue
        }
        
        $fileContent = Get-Content $filePath -Raw
        
        # Check added lines
        $missingLines = @()
        foreach ($addedLine in $fileBlocks[$file].AddedLines) {
            $totalChecked++
            $trimmedLine = $addedLine.Trim()
            
            # Search for the line (case-insensitive, whitespace-tolerant)
            if ($fileContent -notmatch [regex]::Escape($trimmedLine)) {
                $missingLines += $addedLine
                $totalMissing++
            }
        }
        
        if ($missingLines.Count -gt 0) {
            Write-Host "    [FALTA] $($missingLines.Count) líneas no encontradas:" -ForegroundColor Red
            foreach ($line in ($missingLines | Select-Object -First 5)) {
                $preview = if ($line.Length -gt 80) { $line.Substring(0, 77) + "..." } else { $line }
                Write-Host "      - $preview" -ForegroundColor DarkRed
            }
            if ($missingLines.Count -gt 5) {
                Write-Host "      ... y $($missingLines.Count - 5) más" -ForegroundColor DarkRed
            }
        } else {
            Write-Host "    [OK] Todas las líneas presentes" -ForegroundColor Green
        }
    }
    
    Write-Host "`n  RESUMEN $($patch.Name):" -ForegroundColor $patch.Color
    Write-Host "    Total líneas verificadas: $totalChecked" -ForegroundColor White
    if ($totalMissing -gt 0) {
        Write-Host "    Líneas faltantes: $totalMissing" -ForegroundColor Red
        Write-Host "    ✗ PATCH INCOMPLETO" -ForegroundColor Red
    } else {
        Write-Host "    Líneas faltantes: 0" -ForegroundColor Green
        Write-Host "    ✓ PATCH COMPLETO" -ForegroundColor Green
    }
}

Write-Host "`n`n========================================"
Write-Host "VERIFICACIÓN COMPLETADA" -ForegroundColor Cyan
Write-Host "========================================`n"
