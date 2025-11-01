# Script para resolver conflictos de merge en ggml-impl.h durante aplicación de parches
# Uso: powershell -ExecutionPolicy Bypass -File Z_Iosu\patches\0026a-resolve-merge-conflict.ps1

Write-Host "Resolviendo conflicto de merge en ggml/src/ggml-impl.h..." -ForegroundColor Cyan

# Archivo con conflicto
$conflictFile = "llama\vendor\ggml\src\ggml-impl.h"

# Verificar que estamos en el directorio correcto
if (-not (Test-Path $conflictFile)) {
    Write-Host "Error: No se encuentra el archivo $conflictFile" -ForegroundColor Red
    Write-Host "Asegúrate de ejecutar este script desde el directorio raíz de ollama" -ForegroundColor Yellow
    exit 1
}

# Leer el contenido del archivo
$content = Get-Content $conflictFile -Raw

# Verificar que hay un conflicto
if ($content -notmatch '<<<<<<< HEAD') {
    Write-Host "No se detectó conflicto en $conflictFile" -ForegroundColor Green
    exit 0
}

# Crear backup
Copy-Item $conflictFile "$conflictFile.backup" -Force
Write-Host "Backup creado: $conflictFile.backup" -ForegroundColor Gray

# Resolver el conflicto manteniendo ambas partes
# Eliminar marcadores de conflicto y mantener todo el código
$resolved = $content -replace '<<<<<<< HEAD\r?\n', ''
$resolved = $resolved -replace '=======\r?\n', ''
$resolved = $resolved -replace '>>>>>>> GPU discovery enhancements\r?\n', ''

# Guardar el archivo resuelto
Set-Content -Path $conflictFile -Value $resolved -NoNewline

Write-Host "Conflicto resuelto. Se mantuvieron ambas secciones:" -ForegroundColor Green
Write-Host "  - Funciones ggml_can_fuse_subgraph_ext y ggml_can_fuse_subgraph (de HEAD)" -ForegroundColor White
Write-Host "  - Declaraciones NVML y HIP (del parche GPU discovery enhancements)" -ForegroundColor White

# Agregar el archivo resuelto
Set-Location llama\vendor
git add ggml\src\ggml-impl.h

Write-Host ""
Write-Host "Archivo agregado al staging. Ahora ejecuta:" -ForegroundColor Cyan
Write-Host "  cd llama\vendor; git am --continue" -ForegroundColor Yellow
Write-Host ""
Write-Host "Si algo salió mal, restaura el backup con:" -ForegroundColor Gray
Write-Host "  Copy-Item $conflictFile.backup $conflictFile -Force" -ForegroundColor Gray



# iosuc@IAMAX MINGW64 /c/IA/tools/ollama (12-08)
# $ cd llama/vendor; git add ggml/src/ggml-impl.h

# iosuc@IAMAX MINGW64 /c/IA/tools/ollama/llama/vendor ((8d22b1cac...)|AM 1/1)
# $ cd llama/vendor; git am --continue
# bash: cd: llama/vendor: No such file or directory
# Applying: GPU discovery enhancements

# iosuc@IAMAX MINGW64 /c/IA/tools/ollama/llama/vendor ((cfc7f1bb8...))
# $  make -f Makefile.sync clean   apply-patches syn

#  make -f Makefile.sync format-patches

#   make -f Makefile.sync clean apply-patches