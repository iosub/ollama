# Verificación simple de cada patch
$workspaceRoot = "C:\IA\tools\ollama"
Set-Location $workspaceRoot

Write-Host "`n=== VERIFICANDO PATCH 1: 99719122b ===" -ForegroundColor Cyan
$p1 = Get-Content "Z_Iosu\patches\original_99719122b.patch" -Raw
$p1AddedLines = ($p1 -split "`n" | Where-Object { $_ -match '^\+(?!\+\+)' } | ForEach-Object { ($_ -replace '^\+', '').Trim() } | Where-Object { $_ -ne '' }).Count
Write-Host "Líneas añadidas en patch: $p1AddedLines" -ForegroundColor White

Write-Host "`n=== VERIFICANDO PATCH 2: b913e895a ===" -ForegroundColor Yellow
$p2 = Get-Content "Z_Iosu\patches\original_b913e895a.patch" -Raw
$p2AddedLines = ($p2 -split "`n" | Where-Object { $_ -match '^\+(?!\+\+)' } | ForEach-Object { ($_ -replace '^\+', '').Trim() } | Where-Object { $_ -ne '' }).Count
Write-Host "Líneas añadidas en patch: $p2AddedLines" -ForegroundColor White

Write-Host "`n=== VERIFICANDO PATCH 3: de0e3d3c3 ===" -ForegroundColor Magenta
$p3 = Get-Content "Z_Iosu\patches\original_de0e3d3c3.patch" -Raw
$p3AddedLines = ($p3 -split "`n" | Where-Object { $_ -match '^\+(?!\+\+)' } | ForEach-Object { ($_ -replace '^\+', '').Trim() } | Where-Object { $_ -ne '' }).Count
Write-Host "Líneas añadidas en patch: $p3AddedLines" -ForegroundColor White

Write-Host "`n=== VERIFICANDO PATCH 4: e45aecb7b ===" -ForegroundColor Green
$p4 = Get-Content "Z_Iosu\patches\original_e45aecb7b.patch" -Raw
$p4AddedLines = ($p4 -split "`n" | Where-Object { $_ -match '^\+(?!\+\+)' } | ForEach-Object { ($_ -replace '^\+', '').Trim() } | Where-Object { $_ -ne '' }).Count
Write-Host "Líneas añadidas en patch: $p4AddedLines" -ForegroundColor White

Write-Host "`n=== RESUMEN ===" -ForegroundColor Cyan
Write-Host "Total líneas a aplicar: $($p1AddedLines + $p2AddedLines + $p3AddedLines + $p4AddedLines)" -ForegroundColor White

# Verificar archivos críticos
Write-Host "`n=== VERIFICANDO ARCHIVOS CRÍTICOS ===" -ForegroundColor Yellow

# Check case LLM_ARCH_QWEN3_VL in load_tensors
$llamaModel = Get-Content "llama\llama.cpp\src\llama-model.cpp" -Raw
if ($llamaModel -match 'case LLM_ARCH_QWEN3_VL:\s*\{\s*//\s*Same as QWEN3') {
    Write-Host "[✓] case LLM_ARCH_QWEN3_VL encontrado en load_tensors" -ForegroundColor Green
} else {
    Write-Host "[✗] case LLM_ARCH_QWEN3_VL FALTA en load_tensors" -ForegroundColor Red
}

# Check MROPE checks
$mropeCount = ([regex]::Matches($llamaModel, 'rope_type == LLAMA_ROPE_TYPE_MROPE')).Count
Write-Host "MROPE checks encontrados: $mropeCount (esperados: 4+)" -ForegroundColor $(if ($mropeCount -ge 4) { "Green" } else { "Red" })

# Check Windows file handle fix
$llamaLoader = Get-Content "llama\llama.cpp\src\llama-model-loader.cpp" -Raw
if ($llamaLoader -match '_setmaxstdio') {
    Write-Host "[✓] Windows _setmaxstdio encontrado" -ForegroundColor Green
} else {
    Write-Host "[✗] Windows _setmaxstdio FALTA" -ForegroundColor Red
}

# Check deepstack_merger with norm_b
$clip = Get-Content "llama\llama.cpp\tools\mtmd\clip.cpp" -Raw
if ($clip -match 'deepstack_merger.*norm_b') {
    Write-Host "[✓] deepstack_merger con norm_b encontrado" -ForegroundColor Green
} else {
    Write-Host "[✗] deepstack_merger con norm_b FALTA" -ForegroundColor Red
}

Write-Host ""
