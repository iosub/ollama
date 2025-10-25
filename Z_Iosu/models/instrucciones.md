ollama show --template  qwen3_vl_2b_instruct


# Descargar el modelo de texto (8.71 GB)
wget https://huggingface.co/bonswouar/unsloth-Qwen3-VL-GGUF/resolve/main/Qwen3-VL-8B-Instruct-Q8_0.gguf

# Descargar el projector de visión (1.16 GB) 
wget https://huggingface.co/bonswouar/unsloth-Qwen3-VL-GGUF/resolve/main/mmproj-Qwen3-VL-8B-Instruct.gguf

FROM ./Qwen3-VL-8B-Instruct-Q8_0.gguf
ADAPTER ./mmproj-Qwen3-VL-8B-Instruct.gguf

TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ .Response }}<|im_end|>
"""

PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"
PARAMETER temperature 0.7
PARAMETER top_p 0.8


ollama create qwen3vl-8b -f Modelfile-qwen3vl-8b


template de: template.txt
sto es un template de Jinja2 que define cómo formatear las conversaciones para el modelo Qwen3VL. Es la plantilla de chat que le dice al modelo cómo estructurar los mensajes

haz un  Script Era el directorio que te he puesto, donde me pregunte. ¿Qué modelo quiero descargar? Y ve generes todo automático. Tú no lo puedes ejecutar. Y hablo en inglés