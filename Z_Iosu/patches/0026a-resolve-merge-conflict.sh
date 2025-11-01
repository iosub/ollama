#!/bin/bash
# Script para resolver conflictos de merge en ggml-impl.h durante aplicación de parches
# Uso: bash Z_Iosu/patches/0026a-resolve-merge-conflict.sh

echo "Resolviendo conflicto de merge en ggml/src/ggml-impl.h..."

# Archivo con conflicto
CONFLICT_FILE="llama/vendor/ggml/src/ggml-impl.h"

# Verificar que estamos en el directorio correcto
if [ ! -f "$CONFLICT_FILE" ]; then
    echo "Error: No se encuentra el archivo $CONFLICT_FILE"
    echo "Asegúrate de ejecutar este script desde el directorio raíz de ollama"
    exit 1
fi

# Verificar que hay un conflicto
if ! grep -q "<<<<<<< HEAD" "$CONFLICT_FILE"; then
    echo "No se detectó conflicto en $CONFLICT_FILE"
    exit 0
fi

# Crear backup
cp "$CONFLICT_FILE" "$CONFLICT_FILE.backup"

# Resolver el conflicto manteniendo ambas partes
# Eliminamos los marcadores de conflicto y mantenemos el código de HEAD y del parche
sed -i '
/^<<<<<<< HEAD$/,/^=======$/!b
/^<<<<<<< HEAD$/d
/^=======$/d
' "$CONFLICT_FILE"

sed -i '
/^>>>>>>> GPU discovery enhancements$/d
' "$CONFLICT_FILE"

echo "Conflicto resuelto. Se mantuvieron ambas secciones:"
echo "  - Funciones ggml_can_fuse_subgraph_ext y ggml_can_fuse_subgraph (de HEAD)"
echo "  - Declaraciones NVML y HIP (del parche GPU discovery enhancements)"

# Agregar el archivo resuelto
cd llama/vendor
git add ggml/src/ggml-impl.h

echo ""
echo "Archivo agregado al staging. Ahora ejecuta:"
echo "  cd llama/vendor && git am --continue"
echo ""
echo "Si algo salió mal, restaura el backup con:"
echo "  cp $CONFLICT_FILE.backup $CONFLICT_FILE"
