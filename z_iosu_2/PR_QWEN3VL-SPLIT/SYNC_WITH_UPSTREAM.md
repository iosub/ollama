# How to Sync Your Branch with Upstream (Ollama)

This guide explains how to keep your local branch updated with the upstream ollama/ollama repository while preserving your PR changes.

## Prerequisites

Make sure you have the upstream remote configured:
```powershell
git remote add upstream https://github.com/ollama/ollama.git
```

Verify remotes:
```powershell
git remote -v
# Should show:
# origin    https://github.com/YOUR_USERNAME/ollama.git (fetch)
# upstream  https://github.com/ollama/ollama.git (fetch)
```

## Method 1: Rebase (Recommended)

Rebase puts your commits on top of the latest upstream. This keeps a clean linear history.

```powershell
# 1. Fetch latest changes from upstream
git fetch upstream

# 2. Rebase your branch on top of upstream/main
git rebase upstream/main

# 3. If there are conflicts:
#    - Edit the conflicted files to resolve
#    - Stage the resolved files: git add <file>
#    - Continue rebase: git rebase --continue
#    - Repeat until done

# 4. Force push to your fork (required after rebase)
git push origin your-branch --force
```

### If rebase goes wrong:
```powershell
# Abort and return to previous state
git rebase --abort
```

## Method 2: Merge

Merge combines upstream changes with yours. Creates a merge commit.

```powershell
# 1. Fetch latest changes from upstream
git fetch upstream

# 2. Merge upstream/main into your branch
git merge upstream/main

# 3. If there are conflicts:
#    - Edit the conflicted files to resolve
#    - Stage the resolved files: git add <file>
#    - Commit: git commit

# 4. Push to your fork
git push origin your-branch
```

## ⚠️ What NOT to Do

**DO NOT** do this (what went wrong in our session):
```powershell
# WRONG - This deletes your commits!
git reset --hard upstream/main
git cherry-pick <your-commit>  # Has to recover everything manually
```

This approach:
1. Destroys your local commits
2. Requires cherry-picking which can lose code during conflict resolution
3. Makes manual recovery necessary

## Preserving the z_iosu_2 Folder

The `z_iosu_2` folder contains local-only files that shouldn't conflict with upstream. If you're worried:

```powershell
# Before sync, backup z_iosu_2
Copy-Item -Recurse z_iosu_2 ../z_iosu_2_backup

# After sync, restore if needed
Copy-Item -Recurse ../z_iosu_2_backup/* z_iosu_2/
```

## Quick Reference

| Situation | Command |
|-----------|---------|
| Sync with upstream | `git fetch upstream && git rebase upstream/main` |
| Conflict during rebase | Edit files, `git add .`, `git rebase --continue` |
| Abort failed rebase | `git rebase --abort` |
| Push after rebase | `git push origin branch --force` |
| Check current status | `git status` |
| See commit history | `git log --oneline -10` |

## Files Modified by PR #13306 (M-RoPE Support)

When resolving conflicts, pay attention to these files:

- `llama/llama.go` - Core changes: `LoadModelFromFile`, `NEmbdInp`, `NewBatchMRoPE`, `AddImageMRoPE`
- `runner/llamarunner/runner.go` - `pflag`, `extraModelPaths`, `cleanup()`
- `runner/llamarunner/cache.go` - `inputsEqual()`
- `runner/llamarunner/image.go` - `UsesMRoPE()`, `ClearCache()`, `BatchSize()`
- `llama/patches/0032-fix-multimodal-embd-size-calculation.patch` - C++ fix for n_embd_inp

## Applying Patches After Fresh Clone

If you need to apply the PR changes to a fresh upstream clone:

```powershell
# From the ollama directory
git apply z_iosu_2/PR_QWEN3VL-SPLIT/llama__llama.go.patch
git apply z_iosu_2/PR_QWEN3VL-SPLIT/runner__llamarunner__runner.go.patch
git apply z_iosu_2/PR_QWEN3VL-SPLIT/runner__llamarunner__cache.go.patch
git apply z_iosu_2/PR_QWEN3VL-SPLIT/runner__llamarunner__image.go.patch

# Copy the C++ patch
Copy-Item z_iosu_2/PR_QWEN3VL-SPLIT/0032-fix-multimodal-embd-size-calculation.patch llama/patches/
```

Note: Patches may need line ending conversion on Windows:
```powershell
# Convert CRLF to LF if git apply fails
$content = Get-Content "patch-file.patch" -Raw
$content = $content -replace "`r`n", "`n"
[System.IO.File]::WriteAllText("patch-file.patch", $content)
```
