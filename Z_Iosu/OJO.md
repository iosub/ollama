# Ollama Windows Build - Notas de Compilación

## Branch: test-llamacpp-bump (PR #12552 - Granite/Docling Support)

### Fecha: 10 de octubre de 2025
### Versión: 0.12.41
### llama.cpp: 1deee0f8 (bump con soporte Granite/Docling)

---

## ✅ COMPILACIÓN EXITOSA

### Archivos Generados:
- **OllamaSetup.exe** - 420 MB (instalador completo)
- **ollama.exe** - 36.5 MB (CLI principal)
- **windows-amd64-app.exe** - 6.81 MB (aplicación de bandeja)
- **8 DLLs CPU** (ggml-base, x64, SSE4.2, AVX, AVX2, AVX512 variants)
- **3 DLLs CUDA 13** (ggml-cuda, cublas64_13, cublasLt64_13)

---

## 🔧 Requisitos Previos

### Software Instalado:
1. **Visual Studio 2022 Professional** (v19.44.35217.0)
   - Workload: "Desktop development with C++"
   - Windows SDK 10.0.26100.0
   - MSVC v143 compiler toolset

2. **CUDA Toolkits**:
   - CUDA 12.6 (C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6)
   - CUDA 12.8 (C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8)
   - CUDA 13.0 (C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0)
   - Variable de entorno: `CUDA_PATH_V13 = C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0`

3. **llvm-mingw UCRT** (CRÍTICO - NO usar MSYS2):
   - Versión: llvm-mingw-20240619-ucrt-x86_64
   - Descarga: https://github.com/mstorsjo/llvm-mingw/releases/download/20240619/llvm-mingw-20240619-ucrt-x86_64.zip
   - Instalación: `C:\llvm-mingw-ucrt\llvm-mingw-20240619-ucrt-x86_64\`
   - **Importante**: Debe ser UCRT (Universal C Runtime), NO MSVCRT

4. **Go** (versión 1.24 o compatible según go.mod)

5. **ccache** (opcional, para acelerar recompilaciones):
   - Instalación: `choco install ccache`
   - Configuración: 10 GB cache

6. **ImDisk** (opcional, para RAM disk):
   - Crea disco RAM en R: para TEMP (acelera compilación)

7. **Inno Setup 6** (para crear instalador):
   - Instalación: `C:\Program Files (x86)\Inno Setup 6\`

---

## 🚨 Problemas Resueltos

### 1. Error: `cannot parse _cgo_.o as ELF, Mach-O, PE or XCOFF`
**Causa**: llvm-mingw sin UCRT o MSYS2 MinGW-w64 generan objetos incompatibles con CGO
**Solución**: Usar llvm-mingw-20240619-ucrt-x86_64 (misma versión que GitHub Actions)

### 2. Error: `undefined reference to __stdio_common_vfprintf`
**Causa**: Las DLLs compiladas con MSVC usan UCRT, pero MinGW-w64 usa MSVCRT (incompatible)
**Solución**: llvm-mingw UCRT es compatible con MSVC

### 3. Error: `'stdlib.h' file not found`
**Causa**: CGO no encuentra los headers de llvm-mingw
**Solución**: Configurar `CGO_CFLAGS` y `CGO_CXXFLAGS` con `-I$llvmPath\include`

### 4. Error: VS 2026 Insiders incompatible con CUDA
**Causa**: CUDA 13.0 no soporta MSVC 19.50 (VS 2026 Insiders)
**Solución**: Forzar Visual Studio 2022 Professional con `initVS2022Env()`

### 5. Modificación en ggml.go
**Archivo**: `ml/backend/ggml/ggml/src/ggml.go`
**Cambio**: Removido `-lmsvcrt` de LDFLAGS (línea 7)
```go
// ANTES:
// #cgo windows LDFLAGS: -lmsvcrt -static -static-libgcc -static-libstdc++

// DESPUÉS:
// #cgo windows LDFLAGS: -static -static-libgcc -static-libstdc++
```
**Razón**: Conflicto entre MSVC runtime y MinGW

---

## 📝 Configuración de Compilación

### Variables de Entorno Necesarias:
```powershell
$env:VERSION = "0.12.41"
$env:CGO_ENABLED = "1"
$env:CC = "C:\llvm-mingw-ucrt\llvm-mingw-20240619-ucrt-x86_64\bin\gcc.exe"
$env:CXX = "C:\llvm-mingw-ucrt\llvm-mingw-20240619-ucrt-x86_64\bin\g++.exe"
$env:CGO_CFLAGS = "-IC:\llvm-mingw-ucrt\llvm-mingw-20240619-ucrt-x86_64\include"
$env:CGO_CXXFLAGS = "-IC:\llvm-mingw-ucrt\llvm-mingw-20240619-ucrt-x86_64\include"
$env:CUDA_PATH_V13 = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0"
```

### Variables de Entorno Opcionales (Optimización):
```powershell
# RAM Disk para temporales (acelera compilación)
$env:TEMP = "R:\Temp"
$env:TMP = "R:\Temp"

# ccache para compilación de C/C++ (solo funciona con CMake/MSVC en este caso)
$env:CMAKE_C_COMPILER_LAUNCHER = "ccache"
$env:CMAKE_CXX_COMPILER_LAUNCHER = "ccache"
$env:CMAKE_CUDA_COMPILER_LAUNCHER = "ccache"
```

---

## 🎯 Optimizaciones Aplicadas

### 1. CUDA Architectures Reducidas (CMakePresets.json):
**Archivo modificado**: `CMakePresets.json` (añadido a `.git/info/exclude`)

**CUDA 12**:
```json
"CMAKE_CUDA_ARCHITECTURES": "86;89;90"
```
**Original**: 50;60;61;70;75;80;86;87;89;90;90a;120 (12 arquitecturas)
**Optimizado**: 86;89;90 (3 arquitecturas - RTX 3090 y superiores)

**CUDA 13**:
```json
"CMAKE_CUDA_ARCHITECTURES": "86;89;90"
```
**Original**: 75-virtual a 121-virtual (11 arquitecturas virtuales)
**Optimizado**: 86;89;90 (mismo criterio)

**Resultado**: ~75% más rápido en compilación CUDA

### 2. RAM Disk para Temporales:
```powershell
# Crear disco RAM de 20GB en R:
imdisk -a -s 20G -m R: -p "/fs:ntfs /q /y"

# Configurar TEMP en RAM disk
New-Item -ItemType Directory -Path "R:\Temp" -Force
$env:TEMP = "R:\Temp"
$env:TMP = "R:\Temp"
[System.Environment]::SetEnvironmentVariable('TEMP', 'R:\Temp', 'User')
[System.Environment]::SetEnvironmentVariable('TMP', 'R:\Temp', 'User')
```

### 3. ccache (Configurado pero no activo para CGO):
```powershell
choco install ccache
ccache -M 10G
ccache -o cache_dir="C:\Users\<usuario>\.ccache"
```
**Nota**: ccache funciona con CMake/MSVC pero NO con CGO (Go usa su propia caché)

---

## 🚀 Comando de Compilación Completo

### Script Personalizado (Recomendado):
```powershell
$env:VERSION = "0.12.41"
powershell -ExecutionPolicy Bypass -File Z_Iosu\scripts\build_windows.ps1 buildCPU buildCUDA12 buildCUDA13 buildOllama buildApp buildInstaller
```

### Compilación Manual de ollama.exe (si solo necesitas el CLI):
```powershell
# 1. Inicializar VS 2022
$vcvarsPath = "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvarsall.bat"
$tempFile = [System.IO.Path]::GetTempFileName()
cmd /c "`"$vcvarsPath`" amd64 && set > `"$tempFile`""
Get-Content $tempFile | ForEach-Object { 
    if ($_ -match "^(.*?)=(.*)$") { 
        Set-Content env:\"$($matches[1])" $matches[2] 
    } 
}
Remove-Item $tempFile

# 2. Configurar llvm-mingw UCRT
$llvmPath = (Resolve-Path "C:\llvm-mingw-ucrt\llvm-mingw-*").Path
$env:PATH = "$llvmPath\bin;$env:PATH"
$env:CC = "$llvmPath\bin\gcc.exe"
$env:CXX = "$llvmPath\bin\g++.exe"
$env:CGO_CFLAGS = "-I$llvmPath\include"
$env:CGO_CXXFLAGS = "-I$llvmPath\include"
$env:CGO_ENABLED = "1"
$env:VERSION = "0.12.41"

# 3. Compilar
go build -a -trimpath -ldflags "-s -w -X=github.com/ollama/ollama/version.Version=0.12.41 -X=github.com/ollama/ollama/server.mode=release" .
```

---

## 📊 Tiempos de Compilación

**Hardware**: RTX 3090, 64GB RAM, RAM disk habilitado

- **buildCPU**: ~3 minutos (8 DLLs CPU)
- **buildCUDA12**: ~2 minutos (reciclaje de CUDA13)
- **buildCUDA13**: ~8 minutos (3 arquitecturas optimizadas)
- **buildOllama**: ~5 minutos (Go + CGO)
- **buildApp**: ~30 segundos
- **buildInstaller**: ~2 minutos

**Total**: ~20 minutos (vs ~60 minutos con 12 arquitecturas CUDA)

---

## 🔍 Validación Post-Compilación

### Verificar ejecutables:
```powershell
Get-ChildItem "dist" -Recurse -Include "*.exe","*.dll" | Select-Object FullName, @{N="Size (MB)";E={[math]::Round($_.Length/1MB, 2)}}
```

### Probar ollama.exe:
```powershell
.\dist\windows-amd64\ollama.exe --version
# Debería mostrar: ollama version is 0.12.41
```

### Instalar y probar:
```powershell
# Ejecutar instalador
.\dist\OllamaSetup.exe

# Después de instalar
ollama serve
ollama run llama3.2
```

---

## 📚 Archivos Modificados (Personalizados)

### 1. `Z_Iosu/scripts/build_windows.ps1` (378 líneas)
Script personalizado basado en `scripts/build_windows.ps1` del upstream con:
- Función `initVS2022Env()` para forzar VS 2022
- Función `buildOllama()` modificada para usar llvm-mingw UCRT
- Configuración de `CGO_CFLAGS` y `CGO_CXXFLAGS`
- Variables de entorno para ccache

### 2. `ml/backend/ggml/ggml/src/ggml.go` (línea 7)
Removido `-lmsvcrt` de `LDFLAGS` para evitar conflictos MSVC/MinGW

### 3. `CMakePresets.json` (añadido a `.git/info/exclude`)
Optimizado `CMAKE_CUDA_ARCHITECTURES` a `86;89;90` en presets CUDA 12 y 13

---

## ⚠️ Notas Importantes

### 1. **NUNCA crear PR a upstream**
Todos los cambios personalizados deben permanecer en el fork `iosub/ollama`

### 2. **Branch Strategy**
- `main`: Sincronizado con upstream ollama/ollama
- `test-llamacpp-bump`: Testing PR #12552 (Granite/Docling)
- Personalizaciones solo en directorio `Z_Iosu/`

### 3. **llvm-mingw vs MSYS2**
- ✅ **Usar**: llvm-mingw-20240619-ucrt-x86_64
- ❌ **NO usar**: MSYS2 MinGW-w64 (incompatible con MSVC runtime)
- ❌ **NO usar**: llvm-mingw sin UCRT

### 4. **Visual Studio Versions**
- ✅ **VS 2022 Professional** (19.44.x)
- ❌ **VS 2026 Insiders** (19.50.x - incompatible con CUDA 13)

### 5. **CMakePresets.json**
Añadido a `.git/info/exclude` para evitar commits accidentales:
```bash
echo "CMakePresets.json" >> .git/info/exclude
```

---

## 🧪 Testing Granite/Docling

### Cambios en PR #12552:
- llama.cpp: 364a7a6d → 1deee0f8 (~200 commits)
- Soporte para arquitectura Granite (IBM)
- Soporte para procesamiento de documentos Docling
- Nueva API `MtmdChunk` para tokenización multimodal
- Delimiters para modelos Idefics3 (SmolVLM, GraniteDocling)

### Modelos a Probar (cuando estén disponibles):
- `granite-3.1-2b-instruct`
- `granite-docling` (procesamiento de PDFs/documentos)
- `smolvlm` (modelo de visión pequeño)

---

## 📖 Referencias

### Documentación Oficial:
- GitHub Actions build: `.github/workflows/release.yaml`
- Ollama build docs: https://github.com/ollama/ollama/blob/main/docs/development.md

### Issues Relacionados:
- PR #12552: LlamaCPPBump - Granite/Docling support
- llvm-mingw: https://github.com/mstorsjo/llvm-mingw

### Tools:
- llvm-mingw releases: https://github.com/mstorsjo/llvm-mingw/releases
- ccache: https://ccache.dev/
- ImDisk: https://sourceforge.net/projects/imdisk-toolkit/

---

## 🎉 Resumen Ejecutivo

**Estado**: ✅ Compilación exitosa en Windows con Granite/Docling support  
**Versión**: 0.12.41  
**Fecha**: 10 de octubre de 2025  
**Tiempo total**: ~20 minutos  
**Tamaño instalador**: 420 MB  

**Key Success Factors**:
1. llvm-mingw UCRT (compatible con MSVC libraries)
2. CGO_CFLAGS configurado correctamente
3. Visual Studio 2022 (no 2026 Insiders)
4. CUDA architectures optimizadas (3 vs 12)
5. RAM disk para temporales

**Próximos pasos**:
- Instalar y probar funcionalidad básica
- Validar soporte Granite cuando haya modelos disponibles
- Testing multimodal con Docling
- Performance benchmarks CUDA 12 vs CUDA 13    