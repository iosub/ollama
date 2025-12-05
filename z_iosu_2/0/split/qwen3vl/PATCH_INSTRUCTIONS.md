# How to Generate and Apply Patches

## Generate Full Patch (All Changes vs Main)

```powershell
cd C:\IA\tools\ollama
git diff main > z_iosu_2\patches\full_mrope_split.patch
```

## Generate Patches by Category

### A. M-RoPE Core Changes Only
```powershell
git diff main -- llama/llama.go runner/llamarunner/runner.go runner/llamarunner/image.go runner/llamarunner/cache.go > z_iosu_2\patches\mrope_core.patch
```

### B. Split GGUF Support Only
```powershell
git diff main -- fs/ggml/ggml.go fs/ggml/gguf.go llm/server.go server/images.go server/create.go server/sched.go > z_iosu_2\patches\split_gguf.patch
```

### C. All Go Files (Excluding z_iosu_2)
```powershell
git diff main -- "*.go" ":!z_iosu_2/**" > z_iosu_2\patches\all_go_changes.patch
```

## Apply Patch to Fresh Clone

```powershell
# Clone fresh repo
git clone https://github.com/ollama/ollama ollama-fresh
cd ollama-fresh

# Apply patch
git apply ../ollama/z_iosu_2/patches/full_mrope_split.patch
```

## Create Commits for PR

The changes could be split into logical commits:

1. **Split GGUF Loading** (from PR #13259)
   - fs/ggml/ggml.go
   - fs/ggml/gguf.go
   - server/create.go
   - server/images.go

2. **M-RoPE Batch Support**
   - llama/llama.go (NewBatchMRoPE, AddImageMRoPE, IsMRoPE)

3. **M-RoPE Runner Integration**
   - runner/llamarunner/runner.go (input struct, numTokens, numPos)
   - runner/llamarunner/image.go (UsesMRoPE, BatchSize, EmbedSize)
   - runner/llamarunner/cache.go (KV cache clearing)

4. **Server Integration**
   - llm/server.go (extraModelPaths, split fallback)
   - server/sched.go (MetaGGML types)

## Files Summary

| Category | Files |
|----------|-------|
| M-RoPE Core | llama/llama.go, runner/llamarunner/*.go |
| Split GGUF | fs/ggml/*.go, server/create.go, server/images.go |
| Server | llm/server.go, server/sched.go |
| Tests | server/*_test.go |

## Verify Patch Content

```powershell
# Count lines in patch
Get-Content z_iosu_2\patches\mrope_core.patch | Measure-Object -Line

# View patch hunks
git diff main -- llama/llama.go | Select-String "^@@"
```

## Important Notes

1. **PR #13259** already has the split GGUF changes - coordinate to avoid conflicts
2. **M-RoPE changes** are new and should be in separate PR
3. **Test files** may need updates depending on upstream changes
4. **z_iosu_2/** folder should NOT be included in PR (local docs/logs)
