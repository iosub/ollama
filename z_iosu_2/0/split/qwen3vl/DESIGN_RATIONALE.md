# Design Rationale: M-RoPE and Split Model Support for Qwen3-VL

## Executive Summary

This document explains the technical decisions made to support Qwen2-VL and Qwen3-VL models in Ollama. These models present two unique challenges:

1. **M-RoPE (Multi-dimensional Rotary Position Embedding)**: Requires 4 position values per token instead of 1
2. **Split GGUF files**: Vision and text components stored in separate files

## Problem Statement

### The Hallucination Bug

When testing Qwen3-VL with images, the model produced nonsensical output instead of describing the image content. The symptoms were:

- Image was encoded correctly (verified: correct number of tokens, correct embedding dimensions)
- Text generation worked fine without images
- With images: repetitive garbage text, hallucinations

**Root cause discovered**: Ollama was setting only 1 position per token, but Qwen3-VL's M-RoPE attention mechanism expects 4 positions per token with specific 2D spatial encoding.

### Why M-RoPE Exists

Traditional transformers use 1D positional encoding - token 0, token 1, token 2, etc. This works well for text but loses spatial information for images.

Qwen3-VL processes images as a 2D grid of patches. For a 53×76 patch grid:
- There are 4,028 image tokens (53 × 76)
- But the model needs to know each token's (x, y) position in the grid

M-RoPE solves this by encoding 4 position dimensions:
```
pos[0] = temporal (which frame/image, constant for single images)
pos[1] = y position (row in grid: 0 to ny-1)
pos[2] = x position (column in grid: 0 to nx-1)
pos[3] = unused (reserved, always 0)
```

## Design Decisions

### Decision 1: Extend Batch API with M-RoPE Support

**Options considered:**

| Option | Description | Pros | Cons |
|--------|-------------|------|------|
| A. Modify existing `Add()` | Add M-RoPE logic to current function | Minimal API change | Complex conditionals, harder to maintain |
| B. New `AddImageMRoPE()` | Separate function for M-RoPE images | Clean separation, explicit | Slight API expansion |
| C. Separate batch type | Create `MRoPEBatch` type | Maximum isolation | Major refactor, code duplication |

**Chosen: Option B** - `AddImageMRoPE()` function

**Rationale:**
- M-RoPE image processing is fundamentally different (all tokens at once, 2D positions)
- Keeps standard `Add()` simple and unchanged for non-M-RoPE models
- Clear intent when reading code: "this is an M-RoPE image operation"
- Matches llama.cpp's approach (`set_position_mrope_2d()` is a separate function)

### Decision 2: Position Stride = n_tokens (not allocSize)

**The bug:** Original implementation used `allocSize` (batch capacity) as the stride for position arrays.

**Why it matters:** llama.cpp reads positions with stride `batch.n_tokens` (actual token count):
```cpp
// From llama-batch.cpp
size_t src_off = batch.token ? 0 : j * batch.n_tokens;
```

**Fix:** Set positions AFTER updating `n_tokens` to final value, using that value as stride.

**Rationale:** This matches llama.cpp's expectation exactly. Using `allocSize` would place positions at wrong offsets, causing the attention mechanism to read garbage positions.

### Decision 3: numTokens() vs numPos() Distinction

**Key insight from llama.cpp:**
```cpp
// mtmd.cpp
llama_pos mtmd_image_tokens_get_n_pos(const mtmd_image_tokens * image_tokens) {
    if (image_tokens->use_mrope_pos) {
        return std::max(image_tokens->nx, image_tokens->ny);  // NOT nx*ny!
    }
    return image_tokens->n_tokens();
}
```

**What this means:**
- `numTokens()` = nx × ny = KV cache slots consumed = 4,028 for 53×76 image
- `numPos()` = max(nx, ny) = temporal position advance = 76 for 53×76 image

**Why the difference?**
- Each image token occupies one KV cache slot (needs `numTokens()`)
- But temporal position only advances by the larger grid dimension (needs `numPos()`)
- This is because M-RoPE encodes x and y separately; the temporal axis is orthogonal

**Implementation:**
```go
func (inp *input) numTokens() int {
    if inp.isImageMRoPE() {
        return inp.imageNx * inp.imageNy
    }
    return 1
}

func (inp *input) numPos() int {
    if inp.isImageMRoPE() {
        return max(inp.imageNx, inp.imageNy)
    }
    return 1
}
```

### Decision 4: Break Both Loops After M-RoPE Image

**The bug:** After adding an M-RoPE image to the batch, `break` only exited the inner loop, allowing more tokens to be added to the same batch.

**Why this is wrong:** M-RoPE images must be processed alone because:
1. The position array is set up specifically for the image grid
2. Adding text tokens would corrupt the position layout
3. llama.cpp expects embedding batches to be homogeneous

**Fix:** Added `mropeBatchReady` flag to break outer loop:
```go
mropeBatchReady := false
for _, seq := range s.seqs {
    for i, inp := range seq.pendingInputs {
        if inp.isImageMRoPE() {
            batch.AddImageMRoPE(...)
            mropeBatchReady = true
            break  // inner loop
        }
        // ... handle text tokens
    }
    if mropeBatchReady {
        break  // outer loop
    }
}
```

### Decision 5: Batch Size 8192 for M-RoPE Models

**The problem:** Default batch size (512) is too small for images:
- A 53×76 image = 4,028 tokens
- Larger images could have 6,000+ tokens

**Options:**
| Option | Description | Pros | Cons |
|--------|-------------|------|------|
| A. Increase default for all | Set default batch to 8192 | Simple | Wastes memory for text-only models |
| B. Dynamic per-model | Query model for max image size | Optimal | Complex, may not be exposed |
| C. Fixed for M-RoPE | 8192 only when M-RoPE detected | Targeted | Slight complexity |

**Chosen: Option C**

```go
func (c *ImageContext) BatchSize(configuredBatchSize int) int {
    if c.UsesMRoPE() {
        const mropeBatchSize = 8192
        if configuredBatchSize < mropeBatchSize {
            return mropeBatchSize
        }
    }
    return configuredBatchSize
}
```

**Rationale:** Only allocate large batches when needed. 8192 accommodates images up to ~90×90 patches, which covers typical use cases.

### Decision 6: Clear KV Cache for Prompts with Embeddings

**The problem:** Stale KV cache entries could interfere with new image processing, causing inconsistent outputs.

**Fix:** When a prompt contains embeddings (images) and there's existing cache, clear it:
```go
if hasEmbeddings && numPast > 0 {
    c.lc.KvCacheSeqRm(slot.Id, 0, -1)
    numPast = 0
}
```

**Rationale:** This is a conservative approach that ensures fresh state for each image prompt. The performance cost is acceptable because:
- Image encoding is the expensive part, not KV cache rebuild
- Prevents subtle bugs from cache state mismatch
- Can be optimized later if profiling shows it's a bottleneck

### Decision 7: Fallback to llama.cpp Runner for Split Models

**The problem:** Ollama's new engine (`ml/backend/ggml/`) doesn't support loading multiple GGUF files.

**Options:**
| Option | Description | Effort | Risk |
|--------|-------------|--------|------|
| A. Implement split in new engine | Full multi-GGUF support | Weeks | High - major changes |
| B. Fallback to llama.cpp runner | Use existing working code | Hours | Low - proven path |
| C. Block split models | Error message | Minutes | Poor UX |

**Chosen: Option B**

```go
// llm/server.go
if len(ggml.ExtraModelPaths) > 0 {
    slog.Info("using llama.cpp runner for split model")
    return newLlamaCppServer(...)
}
```

**Rationale:**
- llama.cpp already supports split models perfectly
- The runner path is maintained and tested
- Users get working split model support immediately
- New engine can add native support later without blocking users

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         Model Loading                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Is Split Model?  ──Yes──▶  llama.cpp runner (llm/server.go)    │
│       │                            │                             │
│       No                           │                             │
│       │                            │                             │
│       ▼                            ▼                             │
│  New Ollama Engine          ┌─────────────────────────────┐     │
│  (ml/backend/)              │  runner/llamarunner/        │     │
│                             │  ├─ runner.go (M-RoPE batch)│     │
│                             │  ├─ image.go (BatchSize)    │     │
│                             │  └─ cache.go (KV clear)     │     │
│                             └─────────────────────────────┘     │
│                                          │                       │
│                                          ▼                       │
│                             ┌─────────────────────────────┐     │
│                             │  llama/llama.go             │     │
│                             │  ├─ NewBatchMRoPE()         │     │
│                             │  ├─ AddImageMRoPE()         │     │
│                             │  └─ MtmdChunk{Nx, Ny}       │     │
│                             └─────────────────────────────┘     │
│                                          │                       │
│                                          ▼                       │
│                             ┌─────────────────────────────┐     │
│                             │  llama.cpp (C++)            │     │
│                             │  Reads positions with       │     │
│                             │  stride = batch.n_tokens    │     │
│                             └─────────────────────────────┘     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Testing Strategy

### What We Tested

1. **Single image, single prompt**: Basic functionality
2. **Same image, multiple prompts**: Cache behavior
3. **Different images**: Cache invalidation
4. **Large images**: Batch size handling
5. **Text-only prompts**: No regression for non-multimodal use

### Verification Methods

1. **Token counts**: Verified nx×ny matches expected grid size
2. **Position values**: Logged and verified temporal/y/x positions
3. **Output quality**: Model correctly describes image content
4. **Memory**: No leaks with repeated prompts (use `/bye` to exit)

## Future Improvements

1. **Image cache optimization**: Currently disabled for safety; could re-enable with proper hash validation
2. **Native split model support**: Add to new engine when resources allow
3. **Video support**: M-RoPE temporal dimension enables video (multiple frames)
4. **Batch coalescing**: Process multiple small images in one batch

## References

- [llama.cpp mtmd-helper.cpp](https://github.com/ggerganov/llama.cpp/blob/master/tools/mtmd/mtmd-helper.cpp) - `set_position_mrope_2d()`
- [llama.cpp mtmd.cpp](https://github.com/ggerganov/llama.cpp/blob/master/tools/mtmd/mtmd.cpp) - `mtmd_image_tokens_get_n_pos()`
- [llama.cpp llama-batch.cpp](https://github.com/ggerganov/llama.cpp/blob/master/src/llama-batch.cpp) - Position reading logic
- [Qwen2-VL Paper](https://arxiv.org/abs/2409.12191) - M-RoPE description

## Conclusion

The implementation follows these principles:

1. **Correctness over performance**: Clear KV cache, process M-RoPE images alone
2. **Minimal invasive changes**: New functions rather than modifying existing ones
3. **Fallback safety**: Use proven llama.cpp runner for split models
4. **Match llama.cpp behavior**: Position stride, grid encoding, n_pos calculation

These decisions prioritize reliability and maintainability while providing full support for Qwen2-VL and Qwen3-VL vision-language models.
