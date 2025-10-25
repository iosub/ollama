# Script helper para usar ollama CLI con el servidor en debug
# Uso: .\ollama-debug.ps1 run qwen3_vl_4b_instruct:latest "Hola"

$env:OLLAMA_HOST = "http://127.0.0.1:11434"

# Usa el ollama instalado como CLIENTE pero conecta al servidor en debug
& ollama @args
