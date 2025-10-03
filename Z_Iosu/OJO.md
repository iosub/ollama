git checkout main
==========================================
Tus archivos personalizados (incluyendo los scripts) estaban guardados en la rama main
Estabas trabajando en la rama mio (donde no están esos archivos)
Al hacer git checkout main pudiste acceder a tus scripts personalizados
El script build-installer.ps1 se ejecutó correctamente usando llvm-mingw
 powershell -ExecutionPolicy Bypass -File .\scripts\build_windows.ps1 buildCPU buildCUDA12 buildCUDA13 

 $env:PATH = "C:\Program Files\Microsoft Visual Studio\18\Insiders\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin;$env:PATH"; powershell -ExecutionPolicy Bypass -File .\scripts\build_windows.ps1 buildCPU buildCUDA12 buildCUDA13   

 powershell -ExecutionPolicy Bypass -File Z_Iosu\scripts\build-installer.ps1 -Version 0.12.21 -ForceClangGnu -InnoPath "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" -AlsoPortable -SkipBuild -Verbose    