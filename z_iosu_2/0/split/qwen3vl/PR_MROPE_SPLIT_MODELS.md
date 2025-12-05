# Pull Request: M-RoPE Support for Split Multimodal Models (Qwen3-VL)

## Summary

This PR adds support for **M-RoPE (Multi-dimensional Rotary Position Embedding)** required by Qwen2-VL and Qwen3-VL vision models, and enables loading **split GGUF models** where the text and vision components are stored in separate files.

## Motivation

Qwen3-VL models use a unique position encoding scheme called M-RoPE that requires 4 position values per token instead of the standard 1. Without proper M-RoPE support, images are processed but the model hallucinates because the spatial position information is lost.

Additionally, some multimodal models are distributed as split GGUF files (e.g., separate text and vision shards), which wasn't supported by Ollama.

## Key Changes

### 1. M-RoPE Batch Processing (`llama/llama.go`)

- Added `NewBatchMRoPE()` to create batches with 4 position values per token
- Added `AddImageMRoPE()` to set 2D grid positions for image tokens:
  - Position layout: `[temporal...][y...][x...][unused...]`
  - Stride uses `n_tokens` (final token count), not allocation size
  - Temporal = pos0, y = pos0 + row, x = pos0 + column
- Added `IsMRoPE()` to check if batch uses M-RoPE
- Added `NEmbdInp()` to get vision projector embedding dimension
- Modified `MtmdChunk` to include `Nx, Ny` grid dimensions
- Modified `MultimodalTokenize()` to:
  - Use `NEmbdInp()` instead of `NEmbd()` for embedding size
  - Extract grid dimensions from `mtmd_image_tokens_get_nx/ny`
  - Group all image embeddings into single chunk for M-RoPE models

### 2. Runner Modifications (`runner/llamarunner/runner.go`)

- Added `imageNx, imageNy` fields to `input` struct for M-RoPE images
- Added `numTokens()` method: returns `nx * ny` for M-RoPE images (KV cache slots)
- Added `numPos()` method: returns `max(nx, ny)` for position advancement
- Added `mropeBatchReady` flag to prevent adding more inputs after M-RoPE image
- Modified batch creation to use `NewBatchMRoPE()` when vision model requires it
- Modified `loadModel()` to increase batch size to 8192 for multimodal models
- Added repetition loop detection for stuck generations
- Added signal handler for proper resource cleanup on SIGINT/SIGTERM
- Changed from `flag` to `pflag` to support repeated `--model` arguments

### 3. Image Context (`runner/llamarunner/image.go`)

- Added `UsesMRoPE()` to query `mtmd_decode_use_mrope`
- Modified `BatchSize()` to return 8192 for M-RoPE models
- Modified `EmbedSize()` to use `NEmbdInp()` for proper embedding dimensions
- Disabled image cache to prevent KV cache consistency issues

### 4. KV Cache (`runner/llamarunner/cache.go`)

- Clear KV cache when prompt contains embeddings (images)
- Added conservative input comparison for embeddings (pointer-based, not value-based)

### 5. Split GGUF Support (`fs/ggml/ggml.go`)

- Added `MetaGGML` struct to aggregate multiple GGUF shards
- Added `GGUFSplitInfo` to read `split.no` and `split.count` keys
- Added `ForeignTensors` to track tensors across multiple files
- Modified `LoadModel()` to accept `extraModels` paths
- Modified `GraphSize()` and other methods to work with `MetaGGML`

### 6. Server Changes (`llm/server.go`, `server/sched.go`, `server/images.go`)

- Added `extraModelPaths` field throughout the server pipeline
- Modified `NewLlamaServer()` to pass extra model paths to runner
- Modified runner to accept multiple `--model` arguments for split files
- Added fallback to llama.cpp runner when split models are detected (new engine doesn't support them)

### 7. Model Creation (`server/create.go`)

- Added validation for split GGUF files (complete set required)
- Added `broadcastKV()` to propagate metadata across shards
- Added sorting by `split.no` to ensure correct order

## Critical Bug Fixes

### 1. Context Batch Size (Root Cause of Assertion Failure)
```
GGML_ASSERT(n_tokens_all <= cparams.n_batch) failed
```
The context was created with `n_batch=512` but M-RoPE images can have 4000+ tokens. Fixed by increasing batch size to 8192 for multimodal models in `loadModel()`.

### 2. Position Array Layout
The M-RoPE position array must use `n_tokens` as stride, not `allocSize`. This was causing corrupted position encoding.

### 3. Position Advancement
Position should advance by `max(nx, ny)` for M-RoPE images, not `nx * ny`. This matches the temporal dimension semantics.

### 4. Batch Processing Flow
Must not add more inputs to batch after M-RoPE image because position layout depends on final token count. Added `mropeBatchReady` flag.

## Files Modified

| File | Changes |
|------|---------|
| `llama/llama.go` | +228 lines (M-RoPE batch, AddImageMRoPE, MtmdChunk with Nx/Ny) |
| `runner/llamarunner/runner.go` | +316 lines (M-RoPE processing, batch size, cleanup) |
| `runner/llamarunner/image.go` | +91 lines (UsesMRoPE, batch size, embed size) |
| `runner/llamarunner/cache.go` | +52 lines (KV cache clearing, input comparison) |
| `fs/ggml/ggml.go` | +183 lines (MetaGGML, ForeignTensors, split support) |
| `llm/server.go` | +114 lines (extraModelPaths, split fallback) |
| `server/create.go` | +112 lines (split validation) |
| `server/sched.go` | +10 lines (MetaGGML types) |
| `server/images.go` | +43 lines (ExtraModelPaths) |

## Testing

Tested with:
- **Qwen3-VL-2B-Instruct-abliterated-hf** (split model: text + vision)
- Image: 53x76 patches = 4028 tokens
- Verified correct image description output (no hallucinations)
- Verified position encoding matches llama.cpp mtmd-helper.cpp reference

## Breaking Changes

None. Non-split models and non-M-RoPE models continue to work as before.

## Related

- Based on PR #13259 for split GGUF file support
- References `llama.cpp/tools/mtmd/mtmd-helper.cpp` for M-RoPE implementation
- llama.cpp function `mtmd_decode_use_mrope()` for model detection

## Checklist

- [x] M-RoPE batch creation with 4 positions per token
- [x] Correct 2D position encoding for image tokens  
- [x] Proper batch size for large images (8192)
- [x] Split model loading via multiple --model arguments
- [x] Fallback to llama.cpp runner for split models (new engine doesn't support)
- [x] KV cache clearing for prompts with images
- [x] Resource cleanup on shutdown
- [ ] Unit tests for M-RoPE batch operations
- [ ] Integration tests for split models
