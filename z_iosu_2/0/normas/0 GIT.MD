
Guarda el estado actual en un stash con todo incluido, crea una rama de respaldo desde ahí, aplica el stash y súbelo al remoto:


git stash push --include-untracked --message "snapshot-2025-11-11"
git switch -c backup-2025-11-11
git stash pop
git add -A
git commit -m "Backup snapshot 2025-11-11"
git push origin backup-2025-11-11

Regresa a tu rama habitual y vuelve a dejar los cambios como pendientes (sin commit):
git switch 12-11
git cherry-pick --no-commit backup-2025-11-11
git reset