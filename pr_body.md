## Summary

This PR adds **M-RoPE (Multi-dimensional Rotary Position Embedding)** support to enable Qwen2-VL and Qwen3-VL vision-language models to work correctly in Ollama.

> **Dependency**: This PR requires the qwen3vl architecture support in llama.cpp (PR #12992) to be merged first.

## Problem Statement

### The Hallucination Bug

When testing Qwen3-VL with images, the model produced nonsensical output instead of describing the image content:

- Image was encoded correctly (verified: correct number of tokens, correct embedding dimensions)
- Text generation worked fine without images  
- With images: repetitive garbage text, hallucinations

**Root cause**: Ollama was setting only **1 position per token**, but Qwen3-VL's M-RoPE attention mechanism expects **4 positions per token** with specific 2D spatial encoding.

### Why M-RoPE Exists

Traditional transformers use 1D positional encoding (token 0, 1, 2...). This works for text but loses spatial information for images.

Qwen3-VL processes images as a 2D grid of patches. For a 53x76 patch grid:
- There are **4,028 image tokens** (53 x 76)
- The model needs to know each token's (x, y) position in the grid

M-RoPE solves this by encoding **4 position dimensions**:
- pos[0] = temporal (which frame/image, constant for single images)
- pos[1] = y position (row in grid: 0 to ny-1)
- pos[2] = x position (column in grid: 0 to nx-1)  
- pos[3] = unused (reserved, always 0)

## Why llama.cpp Runner Instead of New Engine

I initially attempted to implement M-RoPE support using Ollama's new engine (ml/backend/ggml/). However, I encountered a fundamental limitation: **the new engine doesn't currently support loading split GGUF files** (separate vision and text components).

Qwen2-VL and Qwen3-VL models are distributed as split GGUFs, which is the standard format for these vision-language models.

**I understand that Ollama's goal is to use the new engine**, and I fully support that direction. However, in the meantime, using the llama.cpp runner provides a valid and working solution:

- llama.cpp already supports split models perfectly
- The runner path is maintained and tested  
- Users get working Qwen3-VL support immediately
- The new engine can add native split model support later without blocking users

This is a pragmatic approach: **ship working functionality now** while the new engine matures.

## Technical Implementation

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| New AddImageMRoPE() function | Keeps standard Add() simple; M-RoPE image processing is fundamentally different (all tokens at once, 2D positions) |
| Position stride = n_tokens | Matches llama.cpp's expectation in llama-batch.cpp which reads positions with stride = batch.n_tokens |
| numTokens() vs numPos() | numTokens() = nx * ny (KV cache slots), numPos() = max(nx, ny) (temporal position advance) - matches llama.cpp's mtmd_image_tokens_get_n_pos() |
| Batch size 8192 for M-RoPE | Images can have 4000+ tokens; default 512 is too small |
| Clear KV cache for image prompts | Prevents stale cache interference with new image processing |

## Changes (5 files)

| File | Changes |
|------|---------|
| llama/llama.go | NewBatchMRoPE(), AddImageMRoPE(), NEmbdInp(), UsesMRoPE(), MtmdChunk{Nx, Ny} |
| runner/llamarunner/runner.go | M-RoPE batch handling with numTokens() vs numPos() distinction, mropeBatchReady flag |
| runner/llamarunner/image.go | BatchSize() returns 8192 for M-RoPE models |
| runner/llamarunner/cache.go | Clear KV cache when prompt contains embeddings |
| llama/patches/0032-fix-multimodal-embd-size-calculation.patch | Fix n_embd vs n_embd_inp for vision embeddings |

## Testing

Tested with:
- Qwen3-VL 2B (split model: text + vision GGUFs)
- Qwen3-VL 8B (split model)
- Single image prompts
- Multiple prompts with same image  
- Different images (cache invalidation)
- Large images (batch size handling)
- Text-only prompts (no regression)

## References

- [llama.cpp mtmd-helper.cpp](https://github.com/ggerganov/llama.cpp/blob/master/tools/mtmd/mtmd-helper.cpp) - set_position_mrope_2d()
- [llama.cpp mtmd.cpp](https://github.com/ggerganov/llama.cpp/blob/master/tools/mtmd/mtmd.cpp) - mtmd_image_tokens_get_n_pos()
- [Qwen2-VL Paper](https://arxiv.org/abs/2409.12191) - M-RoPE description
