param(
	[string]$ImagePath = "Z:\IMG_20250125_194419.jpg",
	[switch]$SkipBuild,
	[switch]$SkipDownload
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$distDir = Join-Path $repoRoot "dist\windows-amd64"
$modelDir = $PSScriptRoot
$modelfilePath = Join-Path $modelDir "qwen3_vl_4b_instruct.Modelfile"
$pwsh = Get-Command pwsh -ErrorAction SilentlyContinue
if ($pwsh) {
	$pwshCmd = $pwsh.Source
} else {
	$pwshCmd = (Get-Command powershell.exe).Source
}



Write-Host "[3/6] Writing Modelfile..."
$modelfile = @"
FROM $modelDir

TEMPLATE """{{- if .System }}<|system|>{{ .System }}{{ end }}<|user|>{{ .Prompt }}<|assistant|>"""
PARAMETER temperature 0.2
PARAMETER num_ctx 32768
PARAMETER num_gpu 999
PARAMETER stop "<|user|>"
"@
Set-Content -Path $modelfilePath -Value $modelfile -Encoding UTF8

Write-Host "[4/6] Creating Ollama model..."
Push-Location $distDir
try {
	& C:\IA\tools\ollama\dist\windows-amd64\ollama.exe rm qwen3_vl_4b_instruct >$null | Out-Null
	& C:\IA\tools\ollama\dist\windows-amd64\ollama.exe create qwen3_vl_4b_instruct -f $modelfilePath
} finally {
	Pop-Location
}

Write-Host "[5/6] Start the server with:"
Write-Host "    cd $distDir"
Write-Host "    `$env:OLLAMA_LOG = 'debug'"
Write-Host "    .\ollama.exe serve"

Write-Host "[6/6] After the server is up run:"
Write-Host "    cd $distDir"
Write-Host "    .\ollama.exe run qwen3_vl_4b_instruct:latest \"$ImagePath\""