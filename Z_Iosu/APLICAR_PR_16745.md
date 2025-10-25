# 🔧 APLICAR PR #16745: Qwen VL Causal Masking Fix

## 📋 Problema Actual

Tu Ollama tiene:
- ✅ Arquitectura Qwen3VL registrada (enums, builders)
- ✅ Convertidor con `sections[3]=-1` (MRoPE interleaved)
- ✅ Código MRoPE interleaved correcto
- ❌ **FALTA**: PR #16745 (causal masking fix para Qwen VL)

**Error actual**: `unknown model architecture: 'qwen3vl'`  
**Causa**: El modelo carga pero falla porque el código de causal masking NO tiene el campo `kv_position_of_token`.

## 🎯 Qué hace PR #16745

**Problema que resuelve**:
- Qwen VL tiene posiciones de tokens que NO son estrictamente crecientes (pueden decrecer)
- llama.cpp espera posiciones crecientes para causal masking
- Sin este fix, el causal masking usa posiciones incorrectas

**Solución**:
- Añade nuevo campo `kv_position_of_token` a `llama_ubatch`
- Almacena la posición **real** donde se insertó el token en KV cache
- El causal masking usa esta posición real, no la del modelo

## 📦 Archivos que Modificar

### 1. `include/llama.h`
Añadir campo a estructura `llama_batch`:
```cpp
struct llama_batch {
    int32_t n_tokens;
    llama_token  *  token;
    float        *  embd;
    llama_pos    *  pos;
    int32_t      *  n_seq_id;
    llama_seq_id ** seq_id;
    int8_t       *  logits;
    int32_t      *  kv_position_of_token;  // ← NUEVO: Posición real en KV cache
    // ...
};
```

### 2. `src/llama-batch.h`
Añadir campo a estructura interna:
```cpp
struct llama_batch_allocr {
    // ...
    struct data_t {
        std::vector<llama_token>    token;
        std::vector<float>          embd;
        std::vector<llama_pos>      pos;
        std::vector<int32_t>        n_seq_id;
        std::vector<llama_seq_id *> seq_id;
        std::vector<int8_t>         output;
        std::vector<int32_t>        kv_position_of_token;  // ← NUEVO
    };
};
```

### 3. `src/llama-batch.cpp`
Actualizar lógica de batch:
```cpp
// En llama_batch_allocr::ubatch_add():
udata->kv_position_of_token.resize(n_tokens);
for (int i = 0; i < n_tokens; ++i) {
    if (batch.kv_position_of_token) {
        udata->kv_position_of_token[i] = batch.kv_position_of_token[idxs[i]];
    } else {
        udata->kv_position_of_token[i] = -1;  // Señal de "no especificado"
    }
}
```

### 4. `src/llama-kv-cache.cpp`
Usar nueva posición para causal masking:
```cpp
// En llama_kv_cache_update():
for (uint32_t i = 0; i < n_tokens; ++i) {
    if (ubatch->kv_position_of_token[i] != -1) {
        map_kv_to_batch[ubatch->kv_position_of_token[i]] = i;
    }
}
```

### 5. `tools/mtmd/mtmd.cpp`
Ajustar cálculo de posiciones para M-RoPE:
```cpp
llama_pos mtmd_image_tokens_get_n_pos(const mtmd_image_tokens * image_tokens) {
    if (image_tokens->use_mrope_pos) {
        // Para M-RoPE, la imagen ocupa max(nx, ny) posiciones
        return std::max(image_tokens->nx, image_tokens->ny);
    }
    // ... resto del código
}
```

## 🚀 Cómo Aplicar

### Opción A: Cherry-pick desde PR original (RECOMENDADO)

```powershell
cd C:\IA\tools\ollama\llama\llama.cpp

# Añadir remote del PR
git remote add FMayran https://github.com/FMayran/llama.cpp.git
git fetch FMayran

# Cherry-pick los commits del PR
git cherry-pick b4283007a860231a6916dcd8ced089bdf5b037f0  # Fix proposal
git cherry-pick 5f91c4501622ce64d29cb744db72b4512800433c  # Performance vector
git cherry-pick f05fdf9ece92e03bce606528731a2c94fcb2d28d  # Compiler warning
git cherry-pick 4d51baa03241ce65b27f9770c1395286c623913c  # Syntax adaptation

cd ../..
# Recompilar
$env:VERSION = "0.12.6.99"
powershell -ExecutionPolicy Bypass -File Z_Iosu\scripts\build_windows.ps1 buildCPU buildCUDA13 gatherDependencies buildOllama
```

### Opción B: Cherry-pick desde LETS-BEE/qwen3vl (ALTERNATIVA)

```powershell
cd C:\IA\tools\ollama\llama\llama.cpp

# Añadir remote de LETS-BEE
git remote add LETSBEE https://github.com/LETS-BEE/llama.cpp.git
git fetch LETSBEE

# Cherry-pick commits específicos de causal masking
git cherry-pick <commit-hash-causal-fix>

cd ../..
# Recompilar
```

### Opción C: Aplicar patch manualmente

Si hay conflictos, descarga el patch y aplícalo:

```powershell
# Descargar patch
Invoke-WebRequest -Uri "https://github.com/ggml-org/llama.cpp/pull/16745.patch" `
    -OutFile "C:\IA\tools\ollama\pr16745.patch"

# Aplicar al subdirectorio llama.cpp
cd C:\IA\tools\ollama\llama\llama.cpp
git apply --3way ../../pr16745.patch

# Si hay conflictos, resolverlos manualmente
git status
# Editar archivos en conflicto
git add .
git am --continue

cd ../..
```

## ✅ Verificación Post-Aplicación

### 1. Verificar que el campo existe

```powershell
# Buscar kv_position_of_token en archivos
Select-String -Path "llama\llama.cpp\include\llama.h" -Pattern "kv_position_of_token"
Select-String -Path "llama\llama.cpp\src\llama-batch.h" -Pattern "kv_position_of_token"
```

Debe mostrar las líneas donde se añadió el campo.

### 2. Recompilar completamente

```powershell
# Limpiar build anterior
Remove-Item -Recurse -Force build\lib\ollama\* -ErrorAction SilentlyContinue

# Recompilar DLLs y ejecutable
$env:VERSION = "0.12.6.99"
powershell -ExecutionPolicy Bypass -File Z_Iosu\scripts\build_windows.ps1 buildCPU buildCUDA13 gatherDependencies buildOllama buildApp buildInstaller
```

### 3. Probar Qwen3VL

```powershell
# Instalar nueva versión
.\dist\OllamaSetup.exe

# Probar modelo
ollama run qwen3vl:2b "Describe esta imagen" --image test.jpg
```

## 🔍 Debugging

Si sigue fallando:

```powershell
# Ver logs detallados
$env:OLLAMA_DEBUG=1
ollama serve

# En otra terminal
ollama run qwen3vl:2b "test"
```

Buscar en logs:
- ✅ `llama_model_load: arch = qwen3vl` → Arquitectura reconocida
- ✅ `rope.dimension_sections = [24, 20, 20, -1]` → MRoPE configurado
- ✅ `kv_position_of_token initialized` → Causal masking fix aplicado
- ❌ Si falta alguno, aplicar fix correspondiente

## 📚 Referencias

- PR #16745: https://github.com/ggml-org/llama.cpp/pull/16745
- Issue #13694: https://github.com/ggml-org/llama.cpp/issues/13694
- Issue #16207: https://github.com/ggml-org/llama.cpp/issues/16207
- LETS-BEE branch: https://github.com/LETS-BEE/llama.cpp/tree/qwen3vl

## 🎯 Resumen

**ORDEN DE APLICACIÓN**:
1. ✅ MRoPE interleaved fix (YA APLICADO)
2. ⏳ **PR #16745** - Causal masking (FALTA APLICAR)
3. ⏳ LETS-BEE/qwen3vl - Arquitectura completa
4. ⏳ Ollama PR #12665 - Integración Ollama

**Estado actual**: Tienes #1, falta #2, #3, #4.

**Próximo paso**: Aplicar PR #16745 para que el causal masking funcione correctamente.
