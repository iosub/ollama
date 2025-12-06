# PR Qwen3-VL Split Models Patches

Patches extraídos del PR #13306 (cerrado sin merge) para soporte de modelos multimodales split.

## Patches

### 01-go-mrope-split-support.patch
**Archivos modificados:**
- `llama/llama.go` - Soporte para cargar modelos split y `NEmbdInp()`
- `runner/llamarunner/runner.go` - Uso de pflag para arrays de modelos
- `llm/server.go` - Actualización de llamada a `LoadModelFromFile`

**Cambios clave:**
```go
// llama.go
func LoadModelFromFile(modelPath string, extraModelPaths []string, params ModelParams) (*Model, error)
func (m *Model) NEmbdInp() int  // Retorna n_embd_inp para multimodal

// runner.go  
mpaths := fs.StringArray("model", nil, "path to the model file")  // pflag array
```

### 02-cpp-fix-multimodal-embd-size.patch
**Archivo modificado:**
- `llama/llama.cpp/src/llama-context.cpp`

**Cambios clave:**
- `get_embeddings_ith()`: Usa `n_embd_inp()` para calcular offset de embeddings
- `output_reserve()`: Reserva buffer con tamaño `n_embd_inp()`  
- `output_reorder()`: Usa `n_embd_inp()` para reordenar embeddings
- `state_write_data()`: Serializa embeddings con tamaño `n_embd_inp()`

## Aplicar Patches

```bash
# Desde la raíz del repositorio ollama
cd c:\IA\tools\ollama

# Verificar que se pueden aplicar
git apply --check z_iosu_2/PR_QWEN3VL-SPLIT/applied/01-go-mrope-split-support.patch
git apply --check z_iosu_2/PR_QWEN3VL-SPLIT/applied/02-cpp-fix-multimodal-embd-size.patch

# Aplicar
git apply z_iosu_2/PR_QWEN3VL-SPLIT/applied/01-go-mrope-split-support.patch
git apply z_iosu_2/PR_QWEN3VL-SPLIT/applied/02-cpp-fix-multimodal-embd-size.patch
```

## Revertir Patches

```bash
git apply --reverse z_iosu_2/PR_QWEN3VL-SPLIT/applied/02-cpp-fix-multimodal-embd-size.patch
git apply --reverse z_iosu_2/PR_QWEN3VL-SPLIT/applied/01-go-mrope-split-support.patch
```

## Notas

- Estos patches están adaptados al upstream actual (commit 31b8c6a2)
- El patch original 0032 estaba corrupto, este es una versión regenerada
- Para modelos split Qwen3-VL, mantener `qwen3vl` **COMENTADO** en `fs/ggml/ggml.go`

## Estado: Noviembre 29, 2025

✅ Patches verificados y aplicados exitosamente.
- 4 archivos modificados
- 78 inserciones, 63 eliminaciones

**Pendiente**: Compilar y probar con modelo split Qwen3-VL.
