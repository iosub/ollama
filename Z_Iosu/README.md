# Z_Iosu (local)

Esta carpeta se mantiene local y estable entre ramas.

Todos los archivos quedan versionados excepto los modelos `*.gguf` (ignorados por `Z_Iosu/.gitignore`).
Para que no cambien al hacer checkout de otras ramas, usa los scripts para marcar/desmarcar `skip-worktree`:

## PowerShell (Windows)

- Aplicar skip-worktree:

```
Z_Iosu\scripts\skip-worktree.ps1 apply
```

- Quitar skip-worktree:

```
Z_Iosu\scripts\skip-worktree.ps1 clear
```

## Bash (Git Bash)

- Aplicar skip-worktree:

```
./Z_Iosu/scripts/skip-worktree.sh apply
```

- Quitar skip-worktree:

```
./Z_Iosu/scripts/skip-worktree.sh clear
```

Notas:
- `skip-worktree` es una marca local al clon. En nuevos clones, vuelve a ejecutar el script.
- Los `.gguf` quedan fuera de Git por tamaño; cópialos manualmente cuando los necesites.