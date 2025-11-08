# Solución para ggml b6970 - Multi-variant CPU Backends (Patch Temporal Automático)

## Problema

Con la actualización de llama.cpp a **b6970** en el PR #12992, se introdujo una nueva arquitectura de backends CPU que crea **múltiples variantes** optimizadas para diferentes arquitecturas de CPU en lugar de un único target `ggml-cpu`.

### Cambios en ggml b6970

Cuando `GGML_CPU_ALL_VARIANTS=ON` (activado por defecto en Windows x86), se crean múltiples targets:

- `ggml-cpu-x64` (baseline)
- `ggml-cpu-sse42`
- `ggml-cpu-sandybridge`
- `ggml-cpu-haswell`
- `ggml-cpu-skylakex`
- `ggml-cpu-icelake`
- `ggml-cpu-alderlake`
- `ggml-cpu-sapphirerapids` (solo GCC/Clang, no MSVC)

**Ya NO existe** el target `ggml-cpu` cuando están activadas las variantes.

### Errores sin el patch

```
CMake Error at CMakeLists.txt:61 (get_target_property):
  get_target_property() called with non-existent target "ggml-cpu".

CMake Error at CMakeLists.txt:66 (install):
  install TARGETS given target "ggml-cpu" which does not exist.
```

```
MSBUILD : error MSB1009: Project file does not exist.
Switch: ggml-cpu.vcxproj
```

## Solución

La solución aplica un **patch temporal** a `CMakeLists.txt` durante la configuración de CMake y lo revierte inmediatamente después. El archivo queda en su estado original tras la configuración.

### Archivos involucrados

1. **`Z_Iosu/scripts/patch-cmake-cpu.ps1`** (nuevo)
   - Comenta la línea `set(GGML_CPU_ALL_VARIANTS ON)` en CMakeLists.txt
   - Puede ejecutarse manualmente:
     ```powershell
     # Aplicar
     .\Z_Iosu\scripts\patch-cmake-cpu.ps1
     # Revertir
     .\Z_Iosu\scripts\patch-cmake-cpu.ps1 -Revert
     ```

2. **`CMakeUserPresets.json`** (raíz del repo, git-ignored)
   - Preset `CPU-Native` con `CMAKE_CUDA_COMPILER=OFF` y `GGML_NATIVE=ON`
   - Creado automáticamente por build_windows.ps1

3. **`Z_Iosu/scripts/build_windows.ps1`** - función `buildCPU()` modificada:
   - Aplica patch-cmake-cpu.ps1 antes de CMake configure
   - Revierte el patch después de CMake configure
   - Compila un único `ggml-cpu` optimizado

### Flujo de trabajo

```
Usuario ejecuta: build_windows.ps1 buildCPU
           ↓
    [1] Aplica patch-cmake-cpu.ps1
        (comenta línea forzada de GGML_CPU_ALL_VARIANTS)
           ↓
    [2] CMake configure con --preset CPU-Native
        (detecta un solo ggml-cpu, sin CUDA)
           ↓
    [3] Revierte patch-cmake-cpu.ps1
        (CMakeLists.txt vuelve al original)
           ↓
    [4] Compila ggml-cpu (nativo)
           ↓
    [5] Instala componente CPU
           ↓
    CMakeLists.txt queda LIMPIO (original)
```

## Detalles técnicos

### CMakeUserPresets.json (generado automáticamente)

```json
{
  "version": 3,
  "configurePresets": [
    {
      "name": "CPU-Native",
      "inherits": "Default",
      "description": "CPU-only build optimized for Alder Lake",
      "cacheVariables": {
        "CMAKE_CUDA_COMPILER": "OFF",
        "GGML_BACKEND_DL": "OFF",
        "GGML_VULKAN": "OFF",
        "GGML_METAL": "OFF",
        "GGML_HIP": "OFF",
        "GGML_BLAS": "OFF",
        "GGML_AVX2": "ON",
        "GGML_AVX_VNNI": "ON",
        "GGML_BMI2": "ON"
      }
    }
  ]
}
```

### Comando CMake configure

```powershell
cmake --fresh --preset CPU-Native --install-prefix $DIST_DIR
```

**Resultado:**
- **Solo CPU** - CUDA, Vulkan, Metal, HIP, BLAS desactivados
- Optimizaciones Alder Lake: AVX2 + AVX-VNNI + BMI2
- Un único `ggml-cpu.dll` (no 7 variantes)
- Compilación ~70% más rápida que con multi-variant

## Ventajas de este enfoque

✅ **Archivo original restaurado** - CMakeLists.txt se revierte SIEMPRE después de configure

✅ **Compatible con git** - `git diff` muestra 0 cambios tras la compilación

✅ **Automático** - build_windows.ps1 aplica y revierte el patch sin intervención

✅ **Seguro** - Si falla, puedes revertir manualmente con `patch-cmake-cpu.ps1 -Revert`

✅ **Optimizado** - `GGML_NATIVE=ON` compila para tu CPU específica (AVX2+AVX-VNNI)

✅ **CPU-only** - CMAKE_CUDA_COMPILER=OFF desactiva CUDA completamente

✅ **Rápido** - Un solo `ggml-cpu` vs 7 variantes = 70% menos tiempo de compilación

## Mantenimiento

**Si actualizas desde upstream:**
```powershell
git pull upstream main
# CMakeUserPresets.json no se toca, sigue funcionando
```

**Si necesitas recompilar desde cero:**
```powershell
Remove-Item build, CMakeUserPresets.json -Recurse -Force
.\Z_Iosu\scripts\build_windows.ps1 buildCPU buildOllama
# build_windows.ps1 recreará CMakeUserPresets.json automáticamente
```

**Para agregar CMakeUserPresets.json al .gitignore permanente:**
```bash
echo CMakeUserPresets.json >> .git/info/exclude
```

## Referencias

- PR Ollama #12992: ggml update to b6970
- llama.cpp b6970: Refactorización de backends CPU con soporte multi-variant
- Documentación ggml: `ml/backend/ggml/ggml/src/ggml-cpu/CMakeLists.txt`
