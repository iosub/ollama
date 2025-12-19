# Informe de Corrección: Qwen3-VL Split GGUF

## Estado Actual
Se ha identificado y corregido la causa del cierre inesperado (crash) al cargar modelos Qwen3-VL.

## Diagnóstico del Error
El archivo de log `12.log` reveló un "panic" de Go:
```text
panic: runtime error: invalid memory address or nil pointer dereference
...
github.com/ollama/ollama/model/models/qwen3vl.(*VisionPositionEmbedding).Forward(...)
```
**Causa:** El campo `PositionEmbedding` dentro de la estructura `VisionModel` no estaba siendo inicializado en la función `newVisionModel`. Al intentar acceder a él durante la inferencia (`Forward`), el programa fallaba por ser un puntero nulo (`nil`).

## Solución Aplicada
Se modificó el archivo `model/models/qwen3vl/model_vision.go` para inicializar explícitamente la estructura:

```go
func newVisionModel(c fs.Config) *VisionModel {
    // ...
    model := &VisionModel{
        // ...
        PositionEmbedding: &VisionPositionEmbedding{}, // <-- Línea añadida
        // ...
    }
    return model
}
```

## Situación del Binario
El código ha sido corregido y **compilado exitosamente**.
El nuevo ejecutable `ollama.exe` se encuentra en la carpeta raíz del proyecto:
`C:\IA\tools\ollama\ollama.exe`

## Pasos para Retomar
Si se desea probar la corrección en el futuro:

1. Copiar el ejecutable a la carpeta de distribución:
   ```powershell
   copy C:\IA\tools\ollama\ollama.exe C:\IA\tools\ollama\dist\windows-amd64\
   ```
2. Ejecutar el servidor y el modelo:
   ```powershell
   cd C:\IA\tools\ollama\dist\windows-amd64
   .\ollama.exe serve
   .\ollama.exe run hf.co/unsloth/Qwen3-VL-8B-Instruct-GGUF:Q4_K_M
   ```
