#!/bin/bash
# Script seguro para aplicar parches de llama.cpp resolviendo conflictos conocidos y saltando parches fallidos
# Uso: bash Z_Iosu/patches/apply-patches-auto.sh

# Detectar rutas absolutas correctamente
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
OLLAMA_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
VENDOR_DIR="$OLLAMA_ROOT/llama/vendor"
PATCHES_DIR="$OLLAMA_ROOT/llama/patches"

# Limpieza preventiva SOLO del directorio de rebase de git, no borra nada más
if [ -d "$OLLAMA_ROOT/.git/rebase-apply" ]; then
    echo "Eliminando directorio residual de rebase (.git/rebase-apply) para evitar conflictos previos..."
    rm -rf "$OLLAMA_ROOT/.git/rebase-apply"
fi

# También limpiar en llama/vendor/.git por si los parches se aplican ahí
if [ -d "$VENDOR_DIR/.git/rebase-apply" ]; then
    echo "Eliminando directorio residual de rebase en vendor (.git/rebase-apply)..."
    rm -rf "$VENDOR_DIR/.git/rebase-apply"
fi


echo "======================================"
echo "Aplicando parches de llama.cpp"
echo "======================================"
echo ""

resolve_ggml_impl_conflict() {
    local file="$VENDOR_DIR/ggml/src/ggml-impl.h"
    if grep -q "<<<<<<< HEAD" "$file" 2>/dev/null; then
        echo "Resolviendo conflicto en ggml-impl.h..."
        sed -i '/^<<<<<<< HEAD$/d' "$file"
        sed -i '/^=======$/d' "$file"
        sed -i '/^>>>>>>> GPU discovery enhancements$/d' "$file"
        cd "$VENDOR_DIR"; git add ggml/src/ggml-impl.h; git am --continue; cd "$OLLAMA_ROOT"
        echo "✓ Conflicto en ggml-impl.h resuelto"
        return 0
    fi
    return 1
}
resolve_cuda_conflict() {
    local file="$VENDOR_DIR/ggml/src/ggml-cuda/ggml-cuda.cu"
    if grep -q "<<<<<<< HEAD" "$file" 2>/dev/null; then
        echo "Resolviendo conflicto en ggml-cuda.cu..."
        sed -i '/^<<<<<<< HEAD$/d' "$file"
        sed -i '/^=======$/d' "$file"
        sed -i '/^>>>>>>> ggml: Enable resetting backend devices$/d' "$file"
        cd "$VENDOR_DIR"; git add ggml/src/ggml-cuda/ggml-cuda.cu; git am --skip; cd "$OLLAMA_ROOT"
        echo "✓ Conflicto en ggml-cuda.cu resuelto (parche skipped)"
        return 0
    fi
    return 1
}
resolve_argsort_cpy_conflicts() {
    local argsort="$VENDOR_DIR/ggml/src/ggml-cuda/argsort.cu"
    local cpy="$VENDOR_DIR/ggml/src/ggml-cuda/cpy.cu"
    local has_argsort=false
    local has_cpy=false
    if grep -q "<<<<<<< HEAD" "$argsort" 2>/dev/null; then
        echo "Resolviendo conflicto en argsort.cu..."
        sed -i '/^<<<<<<< HEAD$/d' "$argsort"
        sed -i '/^=======$/d' "$argsort"
        sed -i '/^>>>>>>> add argsort and cuda copy for i32$/d' "$argsort"
        has_argsort=true
    fi
    if grep -q "<<<<<<< HEAD" "$cpy" 2>/dev/null; then
        echo "Resolviendo conflicto en cpy.cu..."
        sed -i '/^<<<<<<< HEAD$/d' "$cpy"
        sed -i '/^=======$/d' "$cpy"
        sed -i '/^>>>>>>> add argsort and cuda copy for i32$/d' "$cpy"
        has_cpy=true
    fi
    if [ "$has_argsort" = true ] || [ "$has_cpy" = true ]; then
        cd "$VENDOR_DIR"; git add ggml/src/ggml-cuda/argsort.cu ggml/src/ggml-cuda/cpy.cu; git am --continue; cd "$OLLAMA_ROOT"
        echo "✓ Conflictos en argsort.cu y/o cpy.cu resueltos"
        return 0
    fi
    return 1
}
try_resolve_conflicts() {
    resolve_ggml_impl_conflict && return 0
    resolve_cuda_conflict && return 0
    resolve_argsort_cpy_conflicts && return 0
    return 1
}
apply_patches_with_conflicts() {
    cd "$VENDOR_DIR"
    if git am --show-current-patch &>/dev/null; then
        echo "Limpiando sesión git am anterior..."
        git am --abort 2>/dev/null || true
    fi
    cd "$OLLAMA_ROOT"
    echo "Formateando parches existentes..."
    make -f Makefile.sync format-patches
    echo ""
    local error_count=0
    for patch in "$PATCHES_DIR"/*.patch; do
        echo "Aplicando: $(basename "$patch")"
        cd "$VENDOR_DIR"
        # Bucle para saltar todos los parches fallidos
        local patch_applied=false
        while true; do
            if git am "$patch" 2>&1; then
                echo "  ✓ Aplicado sin conflictos"
                patch_applied=true
                break
            else
                echo "  ⚠ Conflicto detectado"
                cd "$OLLAMA_ROOT"
                # Si el parche falla y deja rebase-apply, lo eliminamos y saltamos
                if [ -d "$OLLAMA_ROOT/.git/rebase-apply" ]; then
                    rm -rf "$OLLAMA_ROOT/.git/rebase-apply"
                    cd "$VENDOR_DIR"; git am --skip; cd "$OLLAMA_ROOT"
                    echo "  ⚠ Parche saltado por error irreparable"
                    error_count=$((error_count + 1))
                    break
                fi
                max_attempts=3
                attempt=1
                while [ $attempt -le $max_attempts ]; do
                    echo "  Intento $attempt de resolución..."
                    if try_resolve_conflicts; then
                        echo "  ✓ Conflicto resuelto"
                        patch_applied=true
                        break 2
                    else
                        cd "$VENDOR_DIR"
                        if git diff --name-only --diff-filter=U | grep -q .; then
                            echo "  ⚠ Aún hay archivos con conflictos:"
                            git diff --name-only --diff-filter=U
                            if [ $attempt -eq $max_attempts ]; then
                                echo ""
                                echo "❌ ERROR: No se pudieron resolver todos los conflictos"
                                echo "Archivos problemáticos:"
                                git diff --name-only --diff-filter=U | while read -r file; do
                                    echo "  - $file"
                                done
                                cd "$OLLAMA_ROOT"
                                error_count=$((error_count + 1))
                                break 2
                            fi
                        else
                            break 2
                        fi
                        cd "$OLLAMA_ROOT"
                    fi
                    attempt=$((attempt + 1))
                done
            fi
        done
        cd "$OLLAMA_ROOT"
        echo ""
    done
    cd "$OLLAMA_ROOT"
    echo "✓ Todos los parches procesados"
    if [ $error_count -gt 0 ]; then
        return 1
    fi
    return 0
}

echo "Iniciando aplicación de parches..."
if apply_patches_with_conflicts; then
    echo ""
    echo "======================================"
    echo "✓ Proceso completado exitosamente"
    echo "======================================"
    echo ""
    echo "Los parches de llama.cpp han sido aplicados."
    echo "Ahora puedes compilar Ollama con:"
    echo "  powershell -ExecutionPolicy Bypass -File Z_Iosu\\scripts\\build_windows.ps1 buildCPU buildVulkan gatherDependencies buildOllama"
else
    echo ""
    echo "======================================"
    echo "✗ Proceso terminado con errores"
    echo "======================================"
    exit 1
fi