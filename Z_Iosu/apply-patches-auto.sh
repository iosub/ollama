#!/bin/bash
# Script maestro para aplicar parches de llama.cpp resolviendo conflictos automáticamente
# Uso: bash Z_Iosu/apply-patches-auto.sh

set -e  # Salir si hay error

OLLAMA_ROOT="$(pwd)"
VENDOR_DIR="llama/vendor"

echo "======================================"
echo "Aplicando parches de llama.cpp"
echo "======================================"
echo ""

# Función para resolver conflicto en ggml-impl.h
resolve_ggml_impl_conflict() {
    local file="$VENDOR_DIR/ggml/src/ggml-impl.h"
    
    if grep -q "<<<<<<< HEAD" "$file" 2>/dev/null; then
        echo "Resolviendo conflicto en ggml-impl.h..."
        
        # Crear backup
        cp "$file" "$file.backup"
        
        # Eliminar marcadores de conflicto manteniendo todo el código
        sed -i '/^<<<<<<< HEAD$/d' "$file"
        sed -i '/^=======$/d' "$file"
        sed -i '/^>>>>>>> GPU discovery enhancements$/d' "$file"
        
        cd "$VENDOR_DIR"
        git add ggml/src/ggml-impl.h
        git am --continue
        cd "$OLLAMA_ROOT"
        
        echo "✓ Conflicto en ggml-impl.h resuelto"
    fi
}

# Función para resolver conflicto en ggml-cuda.cu
resolve_cuda_conflict() {
    local file="$VENDOR_DIR/ggml/src/ggml-cuda/ggml-cuda.cu"
    
    if grep -q "<<<<<<< HEAD" "$file" 2>/dev/null; then
        echo "Resolviendo conflicto en ggml-cuda.cu..."
        
        # Crear backup
        cp "$file" "$file.backup"
        
        # Eliminar marcadores de conflicto manteniendo el código de HEAD
        sed -i '/^<<<<<<< HEAD$/d' "$file"
        sed -i '/^=======$/d' "$file"
        sed -i '/^>>>>>>> ggml: Enable resetting backend devices$/d' "$file"
        
        cd "$VENDOR_DIR"
        git add ggml/src/ggml-cuda/ggml-cuda.cu
        # Este parche ya está aplicado, así que skip
        git am --skip
        cd "$OLLAMA_ROOT"
        
        echo "✓ Conflicto en ggml-cuda.cu resuelto (parche skipped)"
    fi
}

# Función para resolver conflictos en argsort.cu y cpy.cu
resolve_argsort_cpy_conflicts() {
    local argsort="$VENDOR_DIR/ggml/src/ggml-cuda/argsort.cu"
    local cpy="$VENDOR_DIR/ggml/src/ggml-cuda/cpy.cu"
    local resolved=false
    
    if grep -q "<<<<<<< HEAD" "$argsort" 2>/dev/null; then
        echo "Resolviendo conflicto en argsort.cu (manteniendo ambas partes)..."
        cp "$argsort" "$argsort.backup"
        # Eliminar solo los marcadores, mantener todo el código
        sed -i '/^<<<<<<< HEAD$/d' "$argsort"
        sed -i '/^=======$/d' "$argsort"
        sed -i '/^>>>>>>> add argsort and cuda copy for i32$/d' "$argsort"
        resolved=true
    fi
    
    if grep -q "<<<<<<< HEAD" "$cpy" 2>/dev/null; then
        echo "Resolviendo conflicto en cpy.cu (manteniendo ambas partes)..."
        cp "$cpy" "$cpy.backup"
        # Eliminar solo los marcadores, mantener todo el código  
        sed -i '/^<<<<<<< HEAD$/d' "$cpy"
        sed -i '/^=======$/d' "$cpy"
        sed -i '/^>>>>>>> add argsort and cuda copy for i32$/d' "$cpy"
        resolved=true
    fi
    
    if [ "$resolved" = true ]; then
        cd "$VENDOR_DIR"
        git add ggml/src/ggml-cuda/argsort.cu ggml/src/ggml-cuda/cpy.cu 2>/dev/null
        git am --continue
        cd "$OLLAMA_ROOT"
        echo "✓ Conflictos en argsort.cu y cpy.cu resueltos"
    fi
}

# Aplicar parches con manejo de conflictos
apply_patches_with_conflicts() {
    while true; do
        echo ""
        echo "Ejecutando: make -f Makefile.sync clean apply-patches"
        
        if make -f Makefile.sync clean apply-patches 2>&1 | tee /tmp/patch_output.log; then
            echo ""
            echo "✓ Todos los parches aplicados exitosamente"
            break
        else
            # Verificar qué tipo de conflicto ocurrió
            if grep -q "ggml/src/ggml-impl.h" /tmp/patch_output.log; then
                resolve_ggml_impl_conflict
            elif grep -q "ggml/src/ggml-cuda/ggml-cuda.cu" /tmp/patch_output.log; then
                resolve_cuda_conflict
            else
                echo ""
                echo "✗ Error desconocido aplicando parches"
                echo "Revisa el log en /tmp/patch_output.log"
                exit 1
            fi
        fi
    done
}

# Formatear parches primero
echo "Formateando parches existentes..."
make -f Makefile.sync format-patches

echo ""
echo "Iniciando aplicación de parches..."
apply_patches_with_conflicts

echo ""
echo "======================================"
echo "✓ Proceso completado exitosamente"
echo "======================================"
echo ""
echo "Los parches de llama.cpp han sido aplicados."
echo "Ahora puedes compilar Ollama con:"
echo "  powershell -ExecutionPolicy Bypass -File Z_Iosu\\scripts\\build_windows.ps1 buildCPU buildVulkan gatherDependencies buildOllama"
