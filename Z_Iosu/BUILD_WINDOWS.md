# Guía de Build en Windows (Fork / Entorno Local)

Esta guía documenta los pasos reproducibles para compilar y ejecutar el servidor en Windows usando Go + cgo con toolchain **llvm-mingw** (recomendado) y el script de conveniencia `Z_Iosu/scripts/dev-run.ps1`.

## 1. Requisitos Previos

1. **Go** >= 1.21 (detectado: probado con 1.24.x)
2. **llvm-mingw** (recomendado) o toolchain MinGW que provea `clang`/`gcc` compatibles:
   - Descargar release: https://github.com/mstorsjo/llvm-mingw/releases
   - Extraer, por ejemplo en: `C:\llvm-mingw`
   - Añadir al PATH (inicio de sesión o sesión actual):
     ```powershell
     $env:PATH = 'C:\llvm-mingw\bin;' + $env:PATH
     ```
3. (Opcional) Visual Studio Build Tools – solo si se quiere probar `clang-cl` / `cl.exe` (NO recomendado para cgo aquí).
4. PowerShell 5.1 o superior.

## 2. Verificación Inicial

```powershell
where go
where clang
where x86_64-w64-mingw32-clang
```
Debes ver rutas dentro de `C:\llvm-mingw\bin`. Si solo aparece `clang.exe` sin el prefijo triple, el script igualmente lo detecta.

## 3. Script de Conveniencia
El archivo: `Z_Iosu/scripts/dev-run.ps1`

Parámetros principales:
- `-ForceClangGnu`  Fuerza uso de clang estilo GNU (llvm-mingw), ignora `cl.exe`.
- `-ResetGoEnv`     Elimina CC/CXX previamente persistidos en `go env`.
- `-Clean`          Limpia caché de compilación (`go clean -cache`) y borra `build/` si existe.
- `-ShowEnv`        Muestra variables críticas (CC, CXX, flags cgo, host, etc.).
- `-GoRelease`      Compila binario optimizado (`-trimpath -ldflags '-s -w'`) y ejecuta `./ollama-dev.exe serve`.
- `-UseCMake` / `-Release` (para flujo CMake opcional, normalmente no requerido para `go run`).
- `-PreferClangCL`  Si se desea priorizar `clang-cl` cuando exista (no recomendado en este flujo cgo).
- `-DryRun`         Muestra acciones sin ejecutarlas.

## 4. Primer Arranque (Modo Desarrollo)
```powershell
powershell -ExecutionPolicy Bypass -File Z_Iosu\scripts\dev-run.ps1 -ForceClangGnu -ResetGoEnv -Clean -ShowEnv
```
Esto:
1. Limpia CC/CXX persistentes.
2. Detecta `x86_64-w64-mingw32-clang.exe` (si existe) y lo asigna a `CC` y su par `clang++.exe` a `CXX`.
3. Traduce flags MSVC si apareciesen (/std:c++17, /EHsc) a formato GCC.
4. Inyecta: `--target=x86_64-w64-windows-gnu -fuse-ld=lld` y asegura `-std=c++17`.
5. Ejecuta `go run . serve`.

## 5. Build Optimizado (GoRelease)
```powershell
powershell -ExecutionPolicy Bypass -File Z_Iosu\scripts\dev-run.ps1 -ForceClangGnu -ResetGoEnv -GoRelease -Clean -ShowEnv
```
Genera `ollama-dev.exe` y lo lanza con `serve`.

## 6. Variables CGO Añadidas Automáticamente
Cuando se usa `-ForceClangGnu` el script ajusta (pre-pend):
- `CGO_CFLAGS`: `--target=x86_64-w64-windows-gnu -fuse-ld=lld -O3 -DNDEBUG ...`
- `CGO_CXXFLAGS`: idem + `-std=c++17` (si faltaba)
- `CGO_LDFLAGS`: `--target=x86_64-w64-windows-gnu -fuse-ld=lld ...`

Esto evita el error de formato `_cgo_.o` y el rechazo de flags MSVC.

## 7. Problemas Frecuentes
| Síntoma | Causa | Solución |
|---------|-------|----------|
| `cl : Command line error D8021` | Se usó `cl.exe` con flags GCC (`-Werror`) | Añadir `-ForceClangGnu` y asegurar llvm-mingw primero en PATH |
| `cgo: cannot parse _cgo_.o` | Mezcla de toolchain 32/64 o falta `--target` | Confirmar uso de `x86_64-w64-mingw32-clang.exe`; usar script actualizado |
| No se encuentra `clang.exe` | PATH sin llvm-mingw | Añadir `C:\llvm-mingw\bin` al PATH | 
| Flags `/std:c++17` inválidos | Se ejecutó clang GNU con flags MSVC | Script ya traduce; re-ejecutar con `-ResetGoEnv` |
| Código salida 1 tras servir | Cierre manual/context canceled | Ver logs y decidir si ignorar exit code (future improvement) |

## 8. Limpieza Manual
Para limpiar completamente (incluyendo módulos Go):
```powershell
go clean -cache -modcache
Remove-Item -Recurse -Force build -ErrorAction SilentlyContinue
```

## 9. Ejecución sin Script (Referencia)
Si necesitas reproducir manualmente:
```powershell
$env:PATH = 'C:\llvm-mingw\bin;' + $env:PATH
$env:CC='C:\llvm-mingw\bin\x86_64-w64-mingw32-clang.exe'
$env:CXX='C:\llvm-mingw\bin\x86_64-w64-mingw32-clang++.exe'
$env:CGO_ENABLED=1
$env:CGO_CFLAGS='--target=x86_64-w64-windows-gnu -fuse-ld=lld -O3 -DNDEBUG'
$env:CGO_CXXFLAGS='--target=x86_64-w64-windows-gnu -fuse-ld=lld -O3 -DNDEBUG -std=c++17'
$env:CGO_LDFLAGS='--target=x86_64-w64-windows-gnu -fuse-ld=lld'
$env:OLLAMA_HOST='127.0.0.1:11434'
# Desarrollo
go run . serve
# Release
go build -trimpath -ldflags '-s -w' -o ollama-dev.exe .
./ollama-dev.exe serve
```

## 10. Próximas Mejoras (Opcionales)
- Parámetro `-IgnoreExitCode` para no propagar código distinto de 0 al cerrar.
- `-LogFile` para redirigir stdout/stderr.
- `-Model <name>` para precarga controlada.

## 11. Checklist Rápido
```
[ ] Go instalado
[ ] llvm-mingw en PATH (x86_64-w64-mingw32-clang.exe responde)
[ ] go env CC/CXX no persistentes (usar -ResetGoEnv si dudas)
[ ] Script ejecutado con -ForceClangGnu
[ ] Puerto accesible: http://127.0.0.1:11434/api/ps
```

---
Si algo falla, ejecuta con `-ShowEnv -DryRun` y comparte la salida para diagnóstico.
