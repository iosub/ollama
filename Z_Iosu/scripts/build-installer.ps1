Param(
  [string]$Version = "0.0.0-local",
  [string]$Arch = "amd64", # amd64 o arm64 (para copiar artefactos correctos)
  [string]$IssFile = "app/ollama.iss",
  [string]$OutDir = "Z_Iosu/release/installer",
  [switch]$SkipBuild,
  [switch]$Quiet
)
$ErrorActionPreference='Stop'
function Info($m){ if(-not $Quiet){ Write-Host "[build-installer] $m" -ForegroundColor Cyan } }
function Die($m){ Write-Host "[build-installer][ERROR] $m" -ForegroundColor Red; exit 1 }

if (-not (Test-Path $IssFile)) { Die "No se encuentra $IssFile" }

# 1. Construir binario / dependencias si no se pidió SkipBuild
if (-not $SkipBuild) {
  if ($Arch -eq 'amd64') {
    Info "Compilando binario Windows amd64 (go build)"
    $env:CGO_ENABLED=1
    go build -o dist/windows-amd64/ollama.exe .
    if ($LASTEXITCODE -ne 0) { Die "Fallo go build amd64" }
    if (-not (Test-Path dist/windows-amd64-app.exe)) {
      Info "Generando wrapper app.exe simple"
      Copy-Item dist/windows-amd64/ollama.exe dist/windows-amd64-app.exe -Force
    }
  } elseif ($Arch -eq 'arm64') {
    Die "ARM64 no implementado en este flujo local simplificado"
  } else {
    Die "Arquitectura desconocida: $Arch"
  }
}

# 2. Validar Inno Setup (ISCC)
$ISCC = Get-Command ISCC.exe -ErrorAction SilentlyContinue
if (-not $ISCC) { Die "No se encontró ISCC.exe en PATH (instala Inno Setup)" }

# 3. Set variable de entorno para versión
$env:PKG_VERSION = $Version

# 4. Invocar Inno Setup
Info "Ejecutando ISCC para versión $Version"
& $ISCC.Source $IssFile /Qp | Out-Null
if ($LASTEXITCODE -ne 0) { Die "Fallo ISCC" }

# 5. Mover artefacto a OutDir
if (-not (Test-Path $OutDir)) { New-Item -ItemType Directory -Path $OutDir | Out-Null }
Get-ChildItem dist -Filter 'OllamaSetup.exe' -File -Recurse | ForEach-Object {
  Copy-Item $_.FullName (Join-Path $OutDir ("OllamaSetup-${Version}.exe")) -Force
}
Info "Instalador listo en $OutDir"
