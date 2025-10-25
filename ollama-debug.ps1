# Script helper para usar ollama CLI con el servidor de debug
# Uso: .\ollama-debug.ps1 run modelo:tag
#      .\ollama-debug.ps1 ps
#      .\ollama-debug.ps1 list

$env:OLLAMA_HOST = "http://127.0.0.1:11434"
& ollama @args
