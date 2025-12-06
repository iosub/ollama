# Ir al submódulo
cd llama/vendor

# Abortar cualquier operación de patch
git am --abort 2>/dev/null || true

# Eliminar el archivo problemático del índice y del disco
rm -f ggml/include/ggml-backend.h
git checkout HEAD -- ggml/include/ggml-backend.h

# Resetear todo al estado limpio
git reset --hard HEAD

# Limpiar cualquier archivo temporal
git clean -fd

# Verificar estado
git status

# Volver al directorio raíz
cd ../..

# Limpiar archivos de control de patches
rm -f llama/patches/.*.patched

# Aplicar patches
make -f Makefile.sync apply-patches