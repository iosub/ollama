# feat: M-RoPE support for Qwen2-VL/Qwen3-VL and split GGUF models

## Summary

This PR adds support for **M-RoPE (Multi-dimensional Rotary Position Embedding)** required by Qwen2-VL and Qwen3-VL vision models, and enables loading **split GGUF models** where text and vision components are stored in separate files.

## The Problem

Qwen3-VL models were **hallucinating** when processing images - producing garbage text instead of describing image content. The root cause: Ollama was setting 1 position per token, but M-RoPE requires **4 position values per token** with specific 2D spatial encoding.

## Why M-RoPE Matters

Traditional transformers use 1D positional encoding (token 0, 1, 2...). This loses spatial information for images.

Qwen3-VL processes images as a 2D grid of patches. For a 53×76 patch grid:
- 4,028 image tokens (53 × 76)
- Each token needs (x, y) position in the grid

M-RoPE encodes 4 position dimensions:
```
pos[0] = temporal (which frame/image)
pos[1] = y position (row: 0 to ny-1)  
pos[2] = x position (column: 0 to nx-1)
pos[3] = unused (reserved)
```

## Key Technical Decisions

### 1. Separate `AddImageMRoPE()` function (not modifying `Add()`)

**Rationale:** M-RoPE image processing is fundamentally different - all tokens at once, 2D positions. Keeps standard `Add()` simple. Matches llama.cpp's approach (`set_position_mrope_2d()` is separate).

### 2. Position stride = `n_tokens` (not `allocSize`)

**Critical bug fix:** llama.cpp reads positions with stride `batch.n_tokens`:
```cpp
// llama-batch.cpp
size_t src_off = batch.token ? 0 : j * batch.n_tokens;
```
Using `allocSize` placed positions at wrong offsets → garbage attention.

### 3. `numTokens()` vs `numPos()` distinction

From llama.cpp's `mtmd.cpp`:
```cpp
llama_pos mtmd_image_tokens_get_n_pos(...) {
    if (use_mrope_pos) {
        return std::max(nx, ny);  // NOT nx*ny!
    }
    return n_tokens();
}
```

- `numTokens()` = nx × ny = KV cache slots (4,028 for 53×76)
- `numPos()` = max(nx, ny) = temporal position advance (76 for 53×76)

**Why different?** M-RoPE encodes x/y separately; temporal axis is orthogonal.

### 4. Batch size 8192 for M-RoPE models

Default 512 too small for images (4,028+ tokens). Only increase when M-RoPE detected.

### 5. Fallback to llama.cpp runner for split models

New Ollama engine doesn't support multi-GGUF loading. Using proven llama.cpp path provides immediate working support without risky engine changes.

## Architecture

```
Is Split Model?
    │
    ├─Yes──▶ llama.cpp runner (proven, multi-GGUF support)
    │
    └─No───▶ New Ollama Engine or llama.cpp runner
                    │
                    ▼
            ┌─────────────────────────────┐
            │  runner/llamarunner/        │
            │  ├─ runner.go (M-RoPE batch)│
            │  ├─ image.go (BatchSize)    │
            │  └─ cache.go (KV clear)     │
            └─────────────────────────────┘
                    │
                    ▼
            ┌─────────────────────────────┐
            │  llama/llama.go             │
            │  ├─ NewBatchMRoPE()         │
            │  ├─ AddImageMRoPE()         │
            │  └─ MtmdChunk{Nx, Ny}       │
            └─────────────────────────────┘
                    │
                    ▼
            ┌─────────────────────────────┐
            │  llama.cpp (C++)            │
            │  Reads positions with       │
            │  stride = batch.n_tokens    │
            └─────────────────────────────┘
```

## Files Changed

| File | Description |
|------|-------------|
| `llama/llama.go` | M-RoPE batch creation, `AddImageMRoPE()`, `MtmdChunk` with Nx/Ny |
| `runner/llamarunner/runner.go` | M-RoPE batch processing, batch size 8192, signal cleanup |
| `runner/llamarunner/image.go` | `UsesMRoPE()`, `BatchSize()`, `EmbedSize()` using NEmbdInp |
| `runner/llamarunner/cache.go` | KV cache clearing for prompts with embeddings |
| `fs/ggml/ggml.go` | `MetaGGML`, `ForeignTensors`, split GGUF support |
| `llm/server.go` | `extraModelPaths`, fallback to llama.cpp for split models |
| `server/create.go` | Split GGUF validation, `broadcastKV()` |

## Critical Bug Fixes

1. **Batch size assertion failure** (`n_tokens_all <= cparams.n_batch`): Fixed by 8192 batch for multimodal
2. **Position array layout**: Fixed stride to use `n_tokens`, not `allocSize`  
3. **Position advancement**: Use `max(nx, ny)`, not `nx * ny`
4. **Batch processing**: Don't add inputs after M-RoPE image (`mropeBatchReady` flag)

## Testing

Tested with:
- **Qwen3-VL-8B-Instruct** (split: text + vision GGUF)
- Image: 53×76 patches = 4,028 tokens
- ✅ Correct image descriptions (no hallucinations)
- ✅ Position encoding matches llama.cpp reference

## Breaking Changes

None. Non-split models and non-M-RoPE models work unchanged.

## References

- [llama.cpp mtmd-helper.cpp](https://github.com/ggerganov/llama.cpp/blob/master/tools/mtmd/mtmd-helper.cpp) - `set_position_mrope_2d()`
- [llama.cpp mtmd.cpp](https://github.com/ggerganov/llama.cpp/blob/master/tools/mtmd/mtmd.cpp) - `mtmd_image_tokens_get_n_pos()`
- [Qwen2-VL Paper](https://arxiv.org/abs/2409.12191) - M-RoPE description
- Based on PR #13259 for split GGUF support

---

## Generating Patches

To generate patches for this PR:

```bash
cd c:\IA\tools\ollama

# Generate patches for all modified files
git diff main -- llama/llama.go > z_iosu_2/0/split/qwen3vl/patch/llama_llama.patch
git diff main -- runner/llamarunner/runner.go > z_iosu_2/0/split/qwen3vl/patch/runner.patch
git diff main -- runner/llamarunner/image.go > z_iosu_2/0/split/qwen3vl/patch/image.patch
git diff main -- runner/llamarunner/cache.go > z_iosu_2/0/split/qwen3vl/patch/cache.patch
git diff main -- fs/ggml/ggml.go > z_iosu_2/0/split/qwen3vl/patch/ggml.patch
git diff main -- llm/server.go > z_iosu_2/0/split/qwen3vl/patch/server.patch
git diff main -- server/create.go > z_iosu_2/0/split/qwen3vl/patch/create.patch

# Or generate single combined patch
git diff main > z_iosu_2/0/split/qwen3vl/patch/full_mrope_split.patch
```

To apply patches on a fresh clone:
```bash
git apply full_mrope_split.patch
```
