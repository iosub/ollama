# Patch 0026a: Resolver conflicto de merge en ggml-impl.h
git -C llama/vendor checkout -f d261223d24e97f2df50220e4a5b7f0adb69bba81


## Problema
Al aplicar el parche "GPU discovery enhancements", se produce un conflicto de merge en el archivo `llama/vendor/ggml/src/ggml-impl.h`.

## Causa
El conflicto ocurre porque:
- **HEAD** tiene las funciones `ggml_can_fuse_subgraph_ext` y `ggml_can_fuse_subgraph`
- El **parche** agrega declaraciones para NVML y HIP (gestión de memoria GPU)

Ambas partes son necesarias y deben mantenerse.

## Solución

### Opción 1: Script automático (Recomendado)

**En Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy Bypass -File Z_Iosu\patches\0026a-resolve-merge-conflict.ps1
```

**En Linux/Git Bash:**
```bash
bash Z_Iosu/patches/0026a-resolve-merge-conflict.sh
```

### Opción 2: Manual

1. Abre el archivo `llama/vendor/ggml/src/ggml-impl.h`
2. Busca las líneas con los marcadores de conflicto:
   ```
   <<<<<<< HEAD
   [código de HEAD]
   =======
   [código del parche]
   >>>>>>> GPU discovery enhancements
   ```
3. Elimina SOLO los marcadores (`<<<<<<<`, `=======`, `>>>>>>>`) manteniendo TODO el código
4. El resultado debe tener ambas partes:
   - Las funciones `ggml_can_fuse_subgraph_ext` y `ggml_can_fuse_subgraph`
   - Las declaraciones NVML/HIP (`ggml_nvml_init`, `ggml_hip_mgmt_init`, etc.)
5. Guarda el archivo
6. Ejecuta:
   ```bash
   cd llama/vendor
   git add ggml/src/ggml-impl.h
   git am --continue
   ```

## Después de resolver

Continúa con el proceso normal de sincronización:
```bash
make -f Makefile.sync format-patches
make -f Makefile.sync clean apply-patches
```

## Notas
- Los scripts crean un backup automáticamente (`.backup`)
- Si algo sale mal, puedes restaurar con: `cp ggml-impl.h.backup ggml-impl.h`
