# Script para generar llama/build-info.cpp desde llama/build-info.cpp.in
param(
    [string]$InputFile = "llama/build-info.cpp.in",
    [string]$OutputFile = "llama/build-info.cpp",
    [string]$FetchHead = "origin/add_qwen3vl"
)

# Leer el archivo de entrada
$content = Get-Content $InputFile -Raw

# Reemplazar @FETCH_HEAD@ con el valor
$content = $content -replace '@FETCH_HEAD@', $FetchHead

# Escribir el archivo de salida
Set-Content -Path $OutputFile -Value $content -NoNewline

Write-Host "Generated $OutputFile with FETCH_HEAD=$FetchHead"
