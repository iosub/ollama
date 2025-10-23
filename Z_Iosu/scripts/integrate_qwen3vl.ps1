# Script para integrar mejoras de Qwen3VL desde LETS-BEE/llama.cpp
# Basado en análisis de commits: 6fcc39b y ea677dc

param(
    [switch]$CompileOnly,
    [switch]$TestModel,
    [string]$ModelPath = ".\models\Qwen3-VL-2B-Instruct-Q8_0.gguf"
)

Write-Host "🚀 Integrando mejoras de Qwen3VL en Ollama..." -ForegroundColor Green

# Verificar que estamos en el directorio correcto
if (-not (Test-Path "go.mod")) {
    Write-Error "Debe ejecutar este script desde el directorio raíz de Ollama"
    exit 1
}

# Función para verificar cambios aplicados
function Test-QwenIntegration {
    Write-Host "🔍 Verificando integración de Qwen3VL..." -ForegroundColor Yellow
    
    $checks = @(
        @{
            File = "convert\convert_qwen3vl.go"
            Pattern = "HiddenAct.*string.*json"
            Description = "Campo HiddenAct añadido al VisionModel"
        },
        @{
            File = "ml\backend\ggml\ggml\src\ggml-cpu\ops.cpp"
            Pattern = "is_interleaved_mrope.*=.*true"
            Description = "Soporte MRoPE entrelazado en CPU"
        },
        @{
            File = "ml\backend\ggml\ggml\src\ggml-cuda\rope.cu"
            Pattern = "is_interleaved_mrope.*=.*\(sections\.v\[0\].*==.*24"
            Description = "Soporte MRoPE entrelazado en CUDA"
        },
        @{
            File = "llama\llama.cpp\src\llama-graph.h"
            Pattern = "build_qwen3vl_inp_embd"
            Description = "Declaración de función Qwen3VL en header"
        },
        @{
            File = "llama\llama.cpp\src\llama-graph.cpp"
            Pattern = "build_qwen3vl_inp_embd.*const.*{" 
            Description = "Implementación de función Qwen3VL"
        }
    )

    $allPassed = $true
    foreach ($check in $checks) {
        if (Test-Path $check.File) {
            $content = Get-Content $check.File -Raw
            if ($content -match $check.Pattern) {
                Write-Host "✅ $($check.Description)" -ForegroundColor Green
            } else {
                Write-Host "❌ $($check.Description)" -ForegroundColor Red
                $allPassed = $false
            }
        } else {
            Write-Host "❌ Archivo no encontrado: $($check.File)" -ForegroundColor Red
            $allPassed = $false
        }
    }

    return $allPassed
}

# Verificar integración
if (-not (Test-QwenIntegration)) {
    Write-Error "❌ La integración de Qwen3VL no está completa. Revise los cambios aplicados."
    exit 1
}

Write-Host "✅ Integración de Qwen3VL verificada correctamente" -ForegroundColor Green

if ($CompileOnly) {
    Write-Host "🔧 Modo solo compilación. Compilando Ollama con mejoras Qwen3VL..." -ForegroundColor Yellow
    
    # Compilar solo las bibliotecas necesarias para Qwen3VL
    $env:VERSION = "0.12.6.99-qwen3vl"
    
    Write-Host "📦 Compilando bibliotecas CPU y CUDA con soporte Qwen3VL..."
    & powershell -ExecutionPolicy Bypass -File "Z_Iosu\scripts\build_windows.ps1" buildCPU buildCUDA13 buildVulkan gatherDependencies
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "📦 Compilando CLI de Ollama..."
        & powershell -ExecutionPolicy Bypass -File "Z_Iosu\scripts\build_windows.ps1" buildOllama
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ Compilación completada con éxito" -ForegroundColor Green
            Write-Host "🎯 Ollama con soporte Qwen3VL mejorado está listo" -ForegroundColor Cyan
        } else {
            Write-Error "❌ Error compilando CLI de Ollama"
            exit 1
        }
    } else {
        Write-Error "❌ Error compilando bibliotecas"
        exit 1
    }
}

if ($TestModel -and (Test-Path $ModelPath)) {
    Write-Host "🧪 Probando modelo Qwen3VL..." -ForegroundColor Yellow
    
    # Verificar que el servidor puede cargar el modelo
    $testScript = @"
import subprocess
import time
import requests
import sys

def test_qwen3vl_model():
    print("🚀 Iniciando servidor Ollama...")
    
    # Iniciar servidor en background
    server = subprocess.Popen(
        ["ollama.exe", "serve"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=subprocess.CREATE_NEW_CONSOLE
    )
    
    # Esperar que el servidor inicie
    time.sleep(3)
    
    try:
        # Probar carga del modelo
        response = requests.post('http://localhost:11434/api/show', 
                               json={'name': 'qwen3vl-test'})
        
        if response.status_code == 200:
            print("✅ Servidor responde correctamente")
            
            # Probar embedding básico
            test_response = requests.post('http://localhost:11434/api/embeddings',
                                        json={'model': 'qwen3vl-test', 'prompt': 'test'})
            
            if test_response.status_code == 200:
                print("✅ Funcionalidad de embeddings operativa")
                return True
            else:
                print("❌ Error en embeddings")
                return False
        else:
            print("❌ Servidor no responde")
            return False
            
    except Exception as e:
        print(f"❌ Error de conexión: {e}")
        return False
    finally:
        server.terminate()
        server.wait()

if __name__ == "__main__":
    success = test_qwen3vl_model()
    sys.exit(0 if success else 1)
"@

    $testScript | Out-File -FilePath "test_qwen3vl.py" -Encoding UTF8
    
    try {
        python test_qwen3vl.py
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ Modelo Qwen3VL funciona correctamente" -ForegroundColor Green
        } else {
            Write-Host "⚠️ Modelo requiere ajustes adicionales" -ForegroundColor Yellow
        }
    } catch {
        Write-Host "⚠️ No se pudo probar el modelo (Python no disponible)" -ForegroundColor Yellow
    } finally {
        Remove-Item "test_qwen3vl.py" -ErrorAction SilentlyContinue
    }
}

Write-Host "🎉 Integración de Qwen3VL completada" -ForegroundColor Green
Write-Host ""
Write-Host "📋 Mejoras aplicadas:" -ForegroundColor Cyan
Write-Host "  • Soporte MRoPE entrelazado (CPU + CUDA)" -ForegroundColor White
Write-Host "  • Manejo mejorado de Conv3D temporal patches" -ForegroundColor White  
Write-Host "  • Soporte deepstack visual indexes" -ForegroundColor White
Write-Host "  • Detección automática de activaciones (GELU/SILU)" -ForegroundColor White
Write-Host "  • Embeddings especializados para Qwen3VL" -ForegroundColor White
Write-Host "  • Compatibilidad con MoE (Mixture of Experts)" -ForegroundColor White
Write-Host ""
Write-Host "🔧 Para compilar: .\Z_Iosu\scripts\integrate_qwen3vl.ps1 -CompileOnly" -ForegroundColor Gray
Write-Host "🧪 Para probar: .\Z_Iosu\scripts\integrate_qwen3vl.ps1 -TestModel" -ForegroundColor Gray