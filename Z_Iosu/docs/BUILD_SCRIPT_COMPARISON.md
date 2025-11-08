# Comparación: Script Original vs Personalizado

## 📋 Resumen

Nuestro script `Z_Iosu/scripts/build_windows.ps1` replica **toda la funcionalidad** del script original de Ollama (`scripts/build_windows.ps1`) pero con **mejoras significativas** para nuestro entorno de desarrollo.

---

## ✅ Compatibilidad Total

Ambos formatos de nombre funcionan:

```powershell
# Formato original de Ollama (nombres cortos)
.\Z_Iosu\scripts\build_windows.ps1 cpu cuda13 vulkan ollama app deps installer zip

# Formato personalizado (nombres descriptivos)
.\Z_Iosu\scripts\build_windows.ps1 buildCPU buildCUDA13 buildVulkan buildOllama buildApp gatherDependencies buildInstaller distZip
```

**Son equivalentes** - usas el que prefieras.

---

## 🔄 Tabla de Equivalencias

| Script Original | Script Personalizado | Descripción |
|----------------|---------------------|-------------|
| `cpu` | `buildCPU` | Compila backend CPU |
| `cuda11` | `buildCUDA11` | Compila backend CUDA 11 |
| `cuda12` | `buildCUDA12` | Compila backend CUDA 12 |
| `cuda13` | `buildCUDA13` | Compila backend CUDA 13 |
| `rocm` | `buildROCm` | Compila backend ROCm/HIP |
| `vulkan` | `buildVulkan` | Compila backend Vulkan |
| `ollama` | `buildOllama` | Compila CLI ollama.exe |
| `app` | `buildApp` | Compila app de bandeja |
| `deps` | `gatherDependencies` | Recopila dependencias |
| `sign` | `sign` | Firma ejecutables |
| `installer` | `buildInstaller` | Crea instalador |
| `zip` | `distZip` | Crea archivos ZIP |
| `clean` | `clean` | Limpia build/dist |

---

## 🚀 Mejoras del Script Personalizado

### 1. **Inicialización Automática de Visual Studio 2022**

**Original:**
- Requiere ejecutar desde "Developer PowerShell for VS 2022"
- No funciona desde PowerShell normal

**Personalizado:**
```powershell
function initVS2022Env() {
    # Inicializa VS2022 automáticamente
    # Funciona desde cualquier PowerShell
}
```

✅ **Beneficio:** No necesitas abrir VS Developer PowerShell manualmente.

---

### 2. **Soporte de ccache para Compilaciones Rápidas**

**Original:**
- No usa ccache
- Recompilaciones completas tardan ~15 minutos

**Personalizado:**
```powershell
$env:CMAKE_C_COMPILER_LAUNCHER="ccache"
$env:CMAKE_CXX_COMPILER_LAUNCHER="ccache"
```

✅ **Beneficio:** Recompilaciones en ~3-5 minutos (70% más rápido).

---

### 3. **Compilación CPU Optimizada (Fix para ggml b6970)**

**Original:**
```powershell
function cpu {
    & cmake -B build\cpu --preset CPU
    & cmake --build build\cpu --target ggml-cpu
}
```
**Problema:** Falla con error "target ggml-cpu does not exist"

**Personalizado:**
```powershell
function buildCPU {
    # Aplica patch temporal a CMakeLists.txt
    & "$PSScriptRoot\patch-cmake-cpu.ps1"
    # Configura con preset CPU-Native
    & cmake --fresh --preset CPU-Native
    # Compila con optimizaciones Alder Lake
    & cmake --build build --config Release
    # Revierte patch
    & "$PSScriptRoot\patch-cmake-cpu.ps1" -Revert
}
```

✅ **Beneficio:** 
- Compila **un único** `ggml-cpu.dll` optimizado (no 7 variantes)
- CMakeLists.txt **siempre queda limpio**
- Desactiva backends no solicitados (Vulkan, CUDA si solo pides CPU)

---

### 4. **CLI con llvm-mingw (Fix Bug Ejecución)**

**Original:**
```powershell
function ollama {
    & go build .
}
```
**Problema:** ollama.exe compilado con MSVC tiene bug de ejecución

**Personalizado:**
```powershell
function buildOllama {
    $env:CC="$LLVM_MINGW_DIR\x86_64-w64-mingw32-gcc.exe"
    $env:CXX="$LLVM_MINGW_DIR\x86_64-w64-mingw32-g++.exe"
    & go build -buildmode=pie .
}
```

✅ **Beneficio:** ollama.exe funciona correctamente (sin bug de contexto).

---

### 5. **App con MSVC Explícito (Fix Bug Menú Contextual)**

**Original:**
```powershell
function app {
    & go build ./app/cmd/app/
}
```
**Problema:** Si heredara llvm-mingw del CLI, menú contextual no funciona

**Personalizado:**
```powershell
function buildApp {
    Remove-Item env:CC, env:CXX -ErrorAction SilentlyContinue
    & go build ./app/cmd/app/
}
```

✅ **Beneficio:** Menú contextual del Explorer funciona correctamente.

---

### 6. **Validación y Reportes Mejorados**

**Personalizado agrega:**
- Verificación de Vulkan SDK antes de compilar
- Reporte de tamaños de DLLs compiladas
- Reporte de tamaño final del ejecutable
- Validación de arquitecturas CUDA disponibles
- Mensajes de error más descriptivos

---

### 7. **Función Clean Mejorada**

**Original:**
```powershell
function clean {
    Remove-Item -r dist
    Remove-Item -r build
}
```

**Personalizado:**
```powershell
function clean {
    Remove-Item -r dist
    Remove-Item -r build
    Remove-Item CMakeUserPresets.json  # ← También limpia configuración temporal
}
```

---

## 📊 Comparación de Rendimiento

| Escenario | Script Original | Script Personalizado |
|-----------|----------------|---------------------|
| **Primera compilación** | ~15 min | ~10 min |
| **Recompilación completa** | ~15 min | ~3-5 min ⚡ |
| **Solo buildOllama** | ~1 min | ~30 seg ⚡ |
| **buildCPU solo** | ❌ Error | ✅ ~2 min |

---

## 🎯 Casos de Uso

### Compilación Completa

```powershell
# Original
.\scripts\build_windows.ps1

# Personalizado (equivalente)
.\Z_Iosu\scripts\build_windows.ps1
```

Ambos compilan todo por defecto.

---

### Compilación Selectiva

```powershell
# Original - nombres cortos
.\Z_Iosu\scripts\build_windows.ps1 cpu cuda13 ollama

# Personalizado - nombres descriptivos
.\Z_Iosu\scripts\build_windows.ps1 buildCPU buildCUDA13 buildOllama

# Mixto - funciona también
.\Z_Iosu\scripts\build_windows.ps1 cpu buildCUDA13 ollama
```

---

### Solo CLI (desarrollo rápido)

```powershell
# Original
.\Z_Iosu\scripts\build_windows.ps1 ollama

# Personalizado
.\Z_Iosu\scripts\build_windows.ps1 buildOllama
```

---

### Limpiar y Recompilar

```powershell
# Original
.\Z_Iosu\scripts\build_windows.ps1 clean cpu cuda13 ollama

# Personalizado
.\Z_Iosu\scripts\build_windows.ps1 clean buildCPU buildCUDA13 buildOllama
```

---

## 🔧 Requisitos Adicionales del Script Personalizado

El script personalizado requiere herramientas adicionales que el original no necesita:

1. **ccache** (opcional pero recomendado)
   ```powershell
   scoop install ccache
   ```

2. **llvm-mingw** (requerido para buildOllama)
   ```powershell
   # Descarga: https://github.com/mstorsjo/llvm-mingw/releases
   # Extrae en: C:\llvm-mingw-20240619-ucrt-x86_64
   ```

3. **Vulkan SDK** (solo si usas buildVulkan)
   ```powershell
   .\Z_Iosu\scripts\install-vulkan-sdk.ps1
   ```

**Si no tienes estas herramientas:**
- `buildCPU` funcionará (ccache es opcional)
- `buildOllama` fallará (requiere llvm-mingw)
- `buildVulkan` fallará (requiere Vulkan SDK)
- El resto funciona igual que el original

---

## 📝 Resumen

| Aspecto | Original | Personalizado |
|---------|----------|--------------|
| **Compatibilidad de nombres** | ✅ Cortos | ✅ Cortos + Descriptivos |
| **Funcionalidad** | ✅ Completa | ✅ Completa + Fixes |
| **Velocidad** | ⚠️ Estándar | ⚡ 70% más rápido |
| **Fix CPU b6970** | ❌ Error | ✅ Funciona |
| **Fix ollama.exe** | ⚠️ Bug ejecución | ✅ Funciona |
| **Fix app menú** | ⚠️ Bug context menu | ✅ Funciona |
| **Requiere herramientas extra** | ❌ No | ⚠️ Sí (llvm-mingw, ccache) |
| **Requiere Dev Shell** | ✅ Sí | ❌ No (auto-init) |

---

## 💡 Recomendación

**Usa el script personalizado (`Z_Iosu/scripts/build_windows.ps1`)** porque:

1. ✅ **Funciona** con ggml b6970 (CPU compile fix)
2. ⚡ **Es mucho más rápido** (ccache)
3. ✅ **Elimina bugs** (ollama.exe execution, context menu)
4. 🎯 **Es compatible** con sintaxis del original
5. 🔧 **Se auto-configura** (no necesitas Developer Shell)

El script original sirve de referencia pero **no funciona correctamente** en nuestro entorno actual (ggml b6970, necesidades de llvm-mingw).
