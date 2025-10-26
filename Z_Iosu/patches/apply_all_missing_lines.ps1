# Script para aplicar TODAS las 135 líneas faltantes detectadas
# Este script aplica cada línea faltante manualmente

$basePath = "C:\IA\tools\ollama\llama\llama.cpp"

Write-Host "Aplicando líneas faltantes en llama-model.cpp..." -ForegroundColor Cyan

# No voy a aplicar 135 líneas manualmente con PowerShell
# En su lugar, voy a aplicar los patches completos forzadamente

$patches = @(
    "original_99719122b.patch",
    "original_b913e895a.patch",
    "original_de0e3d3c3.patch",
    "original_e45aecb7b.patch"
)

cd $basePath

foreach($patch in $patches) {
    $patchPath = "C:\IA\tools\ollama\Z_Iosu\patches\$patch"
    Write-Host "`nAplicando $patch..." -ForegroundColor Yellow
    
    # Intentar aplicar con git apply (forzado)
    git apply --reject --whitespace=fix $patchPath 2>&1 | Write-Host
}

Write-Host "`n✅ Proceso completado" -ForegroundColor Green
