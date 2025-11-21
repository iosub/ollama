reset
git reset --hard upstream/main
Remove-Item -Recurse -Force .git/rebase-merge
git rebase upstream/main



https://github.com/dhiltgen/ollama/tree/ggml_bump

Cómo dejar todo sincronizado
Luego podrás hacer git checkout -b ggml_bump dhiltgen/ggml_bump o git merge dhiltgen/ggml_bump según necesites.
Remotos básicos:
git remote add dhiltgen https://github.com/dhiltgen/ollama.git

# ==============================
 make -f Makefile.sync clean
git fetch dhiltgen ggml_bump
Luego podrás hacer git checkout -b ggml_bump dhiltgen/ggml_bump o git merge dhiltgen/ggml_bump según necesites.
git fetch https://github.com/dhiltgen/ollama.git ggml_bump
git remote add dhiltgen https://github.com/dhiltgen/ollama.git
git fetch dhiltgen ggml_bump
git stash push --include-untracked -m "temp-backup-before-checkout"
git stash pop
==============================================
git checkout -b ggml_bump dhiltgen/ggml_bump
git fetch https://github.com/dhiltgen/ollama.git ggml_bump  
git remote add dhiltgen https://github.com/dhiltgen/ollama.git
git fetch dhiltgen ggml_bump
git checkout -b ggml_bump dhiltgen/ggml_bump

Luego podrás hacer git checkout -b ggml_bump dhiltgen/ggml_bump o git merge dhiltgen/ggml_bump según necesites.
Luego podrás hacer git checkout -b ggml_bump dhiltgen/ggml_bump o git merge dhiltgen/ggml_bump según necesites.
git submodule add --force  C:/IA/tools/ollama/Z_Iosu_repo.git Z_Iosu
git merge dhiltgen/ggml_bump
git checkout --theirs .
git add .
git commit
git submodule add --force  C:/IA/tools/ollama/Z_Iosu_repo.git Z_Iosu
git diff --stat HEAD origin/main
git diff --stat HEAD dhiltgen/ggml_bump

origin → tu fork en GitHub (iosub/ollama).
upstream → repo oficial (ollama/ollama).
Comprueba una sola vez: git remote -v debe mostrar ambos.
Actualizar desde upstream (sin PR):
no push
git remote set-url --push upstream https://github.com/ollama/ollama.git no_push
git remote set-url --push dhiltgen no_push
no push
git remote set-url --push upstream https://github.com/ollama/ollama.git no_push
git remote set-url --push dhiltgen no_push

# main
# ==============================================
git fetch upstream
git checkout main
git merge upstream/main (o git rebase upstream/main si prefieres historial lineal).
Así tu main local queda al día con el oficial, pero sin tocar upstream.
git reset --hard upstream/main
Remove-Item -Recurse -Force .git/rebase-merge
git rebase upstream/main

# ==================================================

Propagar a tu fork:

git push origin main
Esto sube a origin/main (tu GitHub) los cambios traídos de upstream para que tu fork refleje lo último.
Ramas de trabajo (12.7.b1, etc.):

git checkout 12.7.b1
git merge main (o git rebase main)
Resolución de conflictos si los hubiera.
git push origin 12.7.b1
De ese modo tu rama local y su homóloga en tu fork (origin/12.7.b1) quedan alineadas con los últimos cambios de ollama/main, sin enviar nada a upstream.

Repite cuando quieras actualizar: ejecutar fetch/merge en main, luego refrescar tus ramas y hacer push.

Con esa rutina mantienes tus branches (locales y en tu GitHub) sincronizados con el repo oficial, sin generar PRs hacia ollama/main.


#
git fetch upstream → trae los commits nuevos del repo oficial.
git checkout main → cambia a tu rama base local. (medio 
    error PS C:\IA\tools\ollama> git checkout main
    Switched to branch 'main'
    Your branch and 'upstream/main' have diverged,
    and have 3 and 21 different commits each, respectively.
    (use "git pull" if you want to integrate the remote branch with yours)

PS C:\IA\tools\ollama> )--- 2

git merge upstream/main (o git rebase upstream/main si prefieres rebase) → integra los cambios de Ollama en tu main.
git push origin main → sube tu main actualizado a tu fork.
git checkout 12.7.b1 → vuelve a tu rama de trabajo.
git merge main (o git rebase main) → actualiza 12.7.b1 con lo que acabas de traer de main.
git push origin 12.7.b1 → sincroniza tu rama en GitHub.

# 2----
Tu rama main local trae tres commits propios (b0113c168, fa3408988, 95d4004c5) y el upstream tiene 21 commits nuevos. Si quieres que tu main (local y en GitHub) sea una copia limpia de ollama/main, haz esto paso a paso:

1. Guardar lo que no pertenece a main

git stash push --include-untracked -m "temp-main-clean"
Esto salva GIT_IMPORTANTE.md, logs y cualquier otro archivo fuera de control para volver a aplicarlos después (en tu rama de trabajo).
2. Respaldar el estado actual de main (por si necesitas esos commits locales)

git branch main-backup
(Tendrás una rama con tu main tal cual estaba).
3. Dejar main exactamente igual que upstream/main

git fetch upstream
git reset --hard upstream/main
4. Sincronizar tu fork
4. Sincronizar tu fork

git push origin main --force-with-lease
(Así origin/main se iguala al upstream. El --force-with-lease evita sobrescribir algo ajeno accidentalmente.)
5. Volver a tu rama de trabajo y recuperar tus archivos

git checkout 12.7.b1
git stash pop
(Los archivos guardados en el stash se reaplican y puedes seguir trabajando en tu rama.)
Con esto, main y origin/main quedan alineados con ollama/main sin abrir PRs. Si algún día necesitas los commits viejos, ahí estará la rama main-backup.

Respecto a “otras conexiones”: revisa con 
git remote -v. Deberías ver solo:


origin   https://github.com/iosub/ollama.git
upstream 
https://github.com/ollama/ollama.git
Si aparece algo más que no reconozcas, puedes quitarlo con git remote remove <nombre>, pero con esos dos remotos ya tienes todo lo necesario para mantener tu fork al día sin interactuar directamente con upstream.

Para añadir un submódulo (y luego poder versionarlo junto con este repo):

1. Elige la URL del repo que quieres enlazar (por ejemplo `git@github.com:iosub/z_iosu.git` o `https://...`).
2. Desde la raíz del proyecto actual, ejecuta:

```
git submodule add <URL> <ruta/donde/quieres/que/viva>
```

Ejemplo para nuestro caso:

```
git submodule add git@github.com:iosub/z_iosu.git Z_Iosu
```

3. Git creará la carpeta y registrará el submódulo en .gitmodules. Añade y commitea ambos:

```
git add .gitmodules Z_Iosu
git commit -m "Add Z_Iosu as submodule"
```

4. La primera vez que otra persona clone el repo, deberá correr `git submodule update --init --recursive` para traer el contenido del submódulo.

Si el repo del submódulo es local (como el que creamos en Z_Iosu_repo.git), usa su URL:

```
git submodule add C:/IA/tools/Z_Iosu_repo.git Z_Iosu
```

y Git lo tratará igual.