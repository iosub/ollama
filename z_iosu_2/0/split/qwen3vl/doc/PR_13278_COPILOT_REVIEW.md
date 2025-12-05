# PR #13278 - Copilot Review Comments

**PR:** https://github.com/ollama/ollama/pull/13278  
**Fecha:** 30 Nov 2025  
**Título:** feat: support split multimodal models with M-RoPE (Qwen3-VL)

## Resumen de Comentarios de Copilot AI

Copilot revisó 163 de 287 archivos y generó **6 comentarios**.

---

## ✅ CORREGIDOS (2)

### 1. Variable Shadowing - Memory Leak
**Archivo:** `llama/llama.go` (líneas 308-310)  
**Problema:** `mp` declarado dos veces (línea 304 y 308), causando memory leak del primer path.

**Antes:**
```go
mp := C.CString(extraModelPaths[i])
defer C.free(unsafe.Pointer(mp))
splitPaths = append(splitPaths, mp)
```

**Después (APLICADO):**
```go
extraMP := C.CString(extraModelPaths[i])
defer C.free(unsafe.Pointer(extraMP))
splitPaths = append(splitPaths, extraMP)
```

**Commit:** `fc491104`

---

### 2. Critical Ordering Requirement
**Archivo:** `llama/llama.go` (línea 559)  
**Problema:** Faltaba comentario explicando dependencia crítica de ordenamiento.

**Antes:**
```go
// Update n_tokens FIRST so we know the final count
```

**Después (APLICADO):**
```go
// CRITICAL: n_tokens MUST be updated FIRST before setting M-RoPE positions below.
// The position stride depends on the FINAL token count (nTokensFinal).
// DO NOT move position-setting code before this line.
```

**Commit:** `fc491104`

---

## ⚠️ OPCIONALES / MENOR PRIORIDAD (2)

### 3. Breaking API Change
**Archivo:** `llama/llama.go` (línea 259)  
**Sugerencia:** Crear `LoadModelFromFiles()` para backward compatibility.

**Estado:** NO APLICADO  
**Razón:** `LoadModelFromFile` es API interna de Ollama, no pública. Los mantenedores decidirán.

---

### 4. M-RoPE Stride Comment
**Archivo:** `llama/llama.go` (líneas 464-476)  
**Sugerencia:** Explicar por qué stride usa `allocSz` en lugar de `n_tokens`.

**Estado:** NO APLICADO  
**Razón:** Ya está explicado en el código. Comentario adicional es opcional.

---

## ⏭️ CÓDIGO UPSTREAM (2)

Estos comentarios son sobre código de llama.cpp (PR #12992 de dhiltgen), no sobre nuestros cambios:

### 5. Comment Mismatch en mtmd.cpp
**Archivo:** `llama/llama.cpp/tools/mtmd/mtmd.cpp` (líneas 1066-1067)  
**Sugerencia:** Cambiar comentario de `max(t,h,w)` a `max(nx, ny)`.

**Estado:** NO APLICADO - Es código upstream de llama.cpp

---

### 6. LIGHTONOCR Comment en clip.cpp
**Archivo:** `llama/llama.cpp/tools/mtmd/clip.cpp`  
**Sugerencia:** Añadir comentario explicando si LIGHTONOCR comparte implementación intencionalmente.

**Estado:** NO APLICADO - Es código upstream de llama.cpp

---

## Commits del PR

| Commit | Descripción |
|--------|-------------|
| `773d9c0d` | feat: support split multimodal models with M-RoPE (Qwen3-VL) |
| `fc491104` | fix: address Copilot review comments |

---

## Archivos Modificados (Nuestros)

- `fs/ggml/ggml.go` - MetaGGML, ForeignTensors
- `fs/ggml/gguf.go` - Split detection
- `llama/llama.go` - NEmbdInp, M-RoPE batch functions ✅ FIXED
- `llm/server.go` - Split model loading
- `runner/llamarunner/cache.go` - KV cache clearing
- `runner/llamarunner/image.go` - Image encoding
- `runner/llamarunner/runner.go` - M-RoPE, loop detection
- `server/create.go` - Split model creation
- `server/images.go` - Split detection
- `server/routes.go` - API routes
- `server/sched.go` - Scheduler
- `discover/runner.go` - Runner startup
- `llama/patches/0032-*.patch` - C++ fix

---

## Repositorios Sincronizados

| Repo | Rama | Último Commit |
|------|------|---------------|
| `C:\IA\tools\ollama` | 14-00 | `e43d5b5e` |
| `C:\IA\tools\ollama-pr` | feat/mrope-split-models | `fc491104` |
| GitHub (iosub/ollama) | feat/mrope-split-models | `fc491104` |
