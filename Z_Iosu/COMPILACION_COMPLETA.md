# GUÍA COMPLETA: Compilación Ollama 0.12.59 con Interfaz Gráfica Funcional

## 🚀 CONFIGURACIÓN PREVIA: ccache (RECOMENDADO)

### ⚡ ¿Cómo funciona el cache en Ollama?

| Componente | Sistema de Cache | Beneficio |
|------------|------------------|-----------|
| **Bibliotecas C/C++** (DLLs) | **ccache** | 50-80% más rápido en recompilaciones |
| **CLI Go** (ollama.exe) | **Go build cache** | Automático, muy eficiente |
| **App Bandeja** (MSVC) | **Go build cache** | Automático |

### 📋 Configuración ccache (Solo para bibliotecas C/C++)

**Ejecuta ANTES de compilar (solo una vez):**

```powershell
# Verificar que ccache está instalado
ccache --version
# Debe mostrar: ccache version 4.12.1

# Configurar ccache para máximo rendimiento
ccache --set-config compression=true
ccache --set-config compression_level=1
ccache --set-config max_size=10G
ccache --set-config sloppiness=time_macros

# Limpiar estadísticas para seguimiento limpio
ccache -z

# Verificar configuración
ccache --show-config | Select-String "compression|max_size|sloppiness"
```

### ✅ Resultado esperado:
```
(config) compression = true
(config) compression_level = 1  
(config) max_size = 10.0 GB
(config) sloppiness = time_macros
```

### 📊 Monitoreo durante compilación:
```powershell
# Ver cache de ccache (bibliotecas C/C++)
ccache -s

# Ver cache de Go (ollama.exe + app)
go env GOCACHE
```

### 🎯 **Velocidades esperadas:**
- **Primera compilación**: ~10 minutos (llena ambos caches)
- **Recompilaciones completas**: ~3-5 minutos  
- **Solo Go (buildOllama)**: ~30 segundos (Go cache muy eficiente)

**NOTA:** ccache solo acelera las bibliotecas C/C++ (buildCPU, buildCUDA13). Go tiene su propio sistema de cache automático.

---

## ✅ SOLUCIÓN FINAL - COPY & PASTE

### 🎯 COMANDO ÚNICO AUTOMÁTICO (RECOMENDADO) ⭐

**Copia y pega esto en PowerShell desde `C:\IA\tools\ollama`:**

```powershell
$env:VERSION = "0.12.59"
powershell -ExecutionPolicy Bypass -File Z_Iosu\scripts\build_windows.ps1 buildCPU buildCUDA13 gatherDependencies buildOllama buildApp buildInstaller
```

**Eso es TODO.** Espera ~10 minutos y tendrás `dist\OllamaSetup.exe` completo y funcional.

---

### Compilación Paso a Paso (Si prefieres ver el progreso)

```powershell
# ============================================================================
# PASO 1: Bibliotecas CPU y CUDA con MSVC
# ============================================================================
$env:VERSION = "0.12.59"
powershell -ExecutionPolicy Bypass -File Z_Iosu\scripts\build_windows.ps1 buildCPU buildCUDA13 gatherDependencies

# ============================================================================
# PASO 2: CLI (ollama.exe) con llvm-mingw
# ============================================================================
powershell -ExecutionPolicy Bypass -File Z_Iosu\scripts\build_windows.ps1 buildOllama

# ============================================================================
# PASO 3: App de Bandeja con MSVC (automático desde script arreglado)
# ============================================================================
powershell -ExecutionPolicy Bypass -File Z_Iosu\scripts\build_windows.ps1 buildApp

# ============================================================================
# PASO 4: Generar Instalador
# ============================================================================
powershell -ExecutionPolicy Bypass -File Z_Iosu\scripts\build_windows.ps1 buildInstaller
```

**NOTA:** El script `build_windows.ps1` ahora automáticamente limpia el entorno de llvm-mingw antes de compilar la app, garantizando que use MSVC puro para evitar el bug del menú contextual.

---

## 📦 Archivos Generados

```
dist\
├── OllamaSetup.exe (420 MB)           ← Instalador completo
├── windows-amd64-app.exe (6.81 MB)    ← App bandeja (MSVC)
├── windows-amd64\
│   ├── ollama.exe (34.88 MB)          ← CLI (llvm-mingw)
│   └── lib\ollama\
│       ├── ggml-*.dll (8 archivos)    ← CPU backends
│       ├── cuda_v13\*.dll (3 archivos) ← CUDA 13
│       └── *.dll (17 archivos)        ← Runtime MSVC
```

**Total: 28 DLLs + 2 ejecutables + instalador**

---

## 🚀 Instalación

```powershell
# 1. Detener procesos antiguos
Get-Process | Where-Object { $_.Name -like "*ollama*" } | Stop-Process -Force

# 2. Desinstalar versión anterior (si existe)
if (Test-Path "$env:LOCALAPPDATA\Programs\Ollama\unins000.exe") {
    Start-Process "$env:LOCALAPPDATA\Programs\Ollama\unins000.exe" -ArgumentList "/SILENT" -Wait
}

# 3. Instalar nueva versión
.\dist\OllamaSetup.exe

# 4. Verificar
ollama --version
# Output: ollama version is 0.12.59
```

---

## ✅ Verificación de la App de Bandeja

1. **Busca el icono 🦙** en la bandeja del sistema (abajo a la derecha del reloj)
2. **Haz clic DERECHO** en el icono
3. **Debe aparecer el menú:**
   - Open Ollama
   - Update Available (si hay updates)
   - Quit Ollama

Si NO aparece el menú → La app se compiló con llvm-mingw (bug conocido)  
Si SÍ aparece el menú → ✅ Compilación correcta con MSVC

---

## ⚠️ PROBLEMA CRÍTICO: App de Bandeja con llvm-mingw

### Síntoma
- ✅ El proceso `ollama app.exe` se ejecuta
- ✅ El icono 🦙 aparece en la bandeja
- ❌ **Clic derecho NO muestra menú**
- ❌ No se puede abrir la interfaz gráfica

### Causa Raíz
**llvm-mingw** tiene incompatibilidad con Win32 API para menús contextuales (system tray menus).

### Solución
**Compilar `buildApp` con MSVC puro (sin llvm-mingw):**

| Componente | Compilador | Razón |
|------------|-----------|-------|
| `ollama.exe` (CLI) | llvm-mingw | CGO + stdlib.h compatibility |
| `ollama app.exe` (GUI) | MSVC | Win32 API (menús contextuales) |
| Bibliotecas DLL | MSVC | Compatibilidad con CUDA/CPU |

---

## 🔧 Troubleshooting

### 1. "Build Failed" pero ollama.exe existe
**Síntoma:** Script muestra "Build Failed" pero ollama.exe se genera correctamente.
```powershell
# Verificar si realmente falló
if (Test-Path "ollama.exe") { 
    Write-Host "✅ Compilación exitosa: $([math]::Round((Get-Item ollama.exe).Length/1MB, 2)) MB" 
} else { 
    Write-Host "❌ Compilación falló realmente" 
}
```
**Causa:** Advertencias de C++ (codecvt_utf8) que Go interpreta como error.  
**Solución:** Script actualizado ignora warnings y verifica archivo generado.

### 2. buildOllama falla con mezcla de entornos
**Síntoma:** Error de VS2022 + llvm-mingw incompatible.
```powershell
# Ejecutar manualmente (siempre funciona)
$env:VERSION = "0.12.59"; $env:CGO_ENABLED="1"
$llvmPath = (Resolve-Path "C:\llvm-mingw-ucrt\llvm-mingw-*").Path
$env:PATH = "$llvmPath\bin;$env:PATH"
$env:CC = "$llvmPath\bin\gcc.exe"; $env:CXX = "$llvmPath\bin\g++.exe"
go build -v -a -trimpath -ldflags "-s -w -X=github.com/ollama/ollama/version.Version=$env:VERSION -X=github.com/ollama/ollama/server.mode=release" .
```

### 3. App no aparece en bandeja
```powershell
# Verificar proceso
Get-Process | Where-Object { $_.Name -like "*ollama*" }

# Debe mostrar:
# - ollama app (app de bandeja)  
# - ollama (servidor backend)
```

### 4. Menú contextual no funciona
```powershell
# Verificar que la app se compiló con MSVC (no llvm-mingw)
# Recompilar siguiendo PASO 3 arriba
```

### 5. ccache no se usa
**Normal:** ccache solo funciona en bibliotecas C/C++ (buildCPU, buildCUDA13).  
Go tiene su propio cache automático muy eficiente.
```powershell
# Ver estadísticas de ambos sistemas
ccache -s              # Cache C/C++
go env GOCACHE         # Cache Go
```

### 6. Verificar DLLs instaladas
```powershell
Get-ChildItem "$env:LOCALAPPDATA\Programs\Ollama\lib\ollama" -Recurse -Filter "*.dll" | Measure-Object
# Debe mostrar: Count = 28
```

---

## 📝 Resumen de Requisitos

### Software Necesario
- ✅ Visual Studio 2022 Professional
- ✅ CUDA 13.0
- ✅ llvm-mingw-20240619-ucrt-x86_64
- ✅ Go 1.24+
- ✅ Inno Setup 6.5.1
- ✅ windres (de llvm-mingw)

### Variables de Entorno
```powershell
$env:VERSION = "0.12.59"
$env:CUDA_PATH_V13_0 = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0"
```

---

## 🎯 Comandos Rápidos

### Compilación Completa Automática (Un Solo Comando)
```powershell
$env:VERSION = "0.12.59"
powershell -ExecutionPolicy Bypass -File Z_Iosu\scripts\build_windows.ps1 buildCPU buildCUDA13 gatherDependencies buildOllama buildApp buildInstaller
```

### Solo Recompilar y Reinstalar
```powershell
# Si ya tienes las DLLs compiladas y solo cambió el código
powershell -ExecutionPolicy Bypass -File Z_Iosu\scripts\build_windows.ps1 buildOllama buildApp buildInstaller
```

### Compilar ollama.exe Manualmente (Siempre funciona)
```powershell
# Comando manual que NUNCA falla
$env:VERSION = "0.12.59"; $env:CGO_ENABLED="1"
$llvmPath = (Resolve-Path "C:\llvm-mingw-ucrt\llvm-mingw-*").Path; $env:PATH = "$llvmPath\bin;$env:PATH"
$env:CC = "$llvmPath\bin\gcc.exe"; $env:CXX = "$llvmPath\bin\g++.exe"
go build -v -a -trimpath -ldflags "-s -w -X=github.com/ollama/ollama/version.Version=$env:VERSION -X=github.com/ollama/ollama/server.mode=release" .

# Copiar a dist (si usas comando manual)
cp .\ollama.exe .\dist\windows-amd64\
```

### Reinstalación Limpia
```powershell
Get-Process | Where-Object { $_.Name -like "*ollama*" } | Stop-Process -Force
Start-Process "$env:LOCALAPPDATA\Programs\Ollama\unins000.exe" -ArgumentList "/SILENT" -Wait
.\dist\OllamaSetup.exe
```

---

## ✅ Resultado Final

- **Versión:** Ollama 0.12.59 (test-llamacpp-bump)
- **Soporte:** Granite + Docling (llama.cpp 1deee0f8)
- **Backend:** CUDA 13.0 (arquitecturas 86, 89, 90)
- **Interfaz:** App de bandeja 100% funcional con menú contextual
- **CLI:** Compatible con llvm-mingw UCRT

**¡Disfruta tu comida!** 🍽️
