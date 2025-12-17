# M-RoPE Implementation for Qwen3-VL in Ollama

## Overview

M-RoPE (Multi-dimensional Rotary Position Embedding) is used by Qwen2-VL and Qwen3-VL models for vision-language tasks. Unlike standard RoPE which uses 1 position per token, M-RoPE uses 4 position values per token:

- `pos[0]`: Temporal position (for video frames, or constant for images)
- `pos[1]`: Y position (row in image grid)
- `pos[2]`: X position (column in image grid)  
- `pos[3]`: Unused (always 0)

## Key Formulas

For an image with grid dimensions `nx × ny`:

| Metric | Formula | Example (53×76) |
|--------|---------|-----------------|
| **numTokens** | `nx × ny` | 4028 tokens |
| **numPos** | `max(nx, ny)` | 76 positions |
| **embed size** | `numTokens × n_embd_inp` | 4028 × 16384 |

**Critical distinction:**
- `numTokens()` = how many KV cache slots the image occupies
- `numPos()` = how much the temporal position advances

## Position Array Layout

For a batch with `n_tokens` embeddings, llama.cpp expects:

```
pos[0..n_tokens-1]           = temporal positions
pos[n_tokens..2*n_tokens-1]  = Y positions
pos[2*n_tokens..3*n_tokens-1] = X positions
pos[3*n_tokens..4*n_tokens-1] = zeros (unused)
```

For image tokens at position `pos0`:
- Temporal: `pos0` (same for all tokens in image)
- Y: `pos0 + y` where y = 0..ny-1
- X: `pos0 + x` where x = 0..nx-1

## Files Modified

### 1. `llama/llama.go` - CGO Bindings

#### Batch Structure
```go
type Batch struct {
    c           C.struct_llama_batch
    batchSize   int
    maxSeq      int
    embedSize   int
    nPosPerEmbd int        // 1 for standard, 4 for M-RoPE
    mropePos    []C.llama_pos  // Go-managed position array
}
```

#### NewBatchMRoPE()
Creates a batch with 4 positions per token for M-RoPE models.

#### AddImageMRoPE()
Adds all image embeddings at once with correct 2D position encoding:
```go
func (b *Batch) AddImageMRoPE(embeddings []float32, pos0 int, nx int, ny int, logitsLast bool, seqIds ...int)
```

Key implementation details:
1. Uses `batch.n_tokens` as stride (NOT `allocSize`)
2. Sets positions AFTER updating `n_tokens` to final value
3. All tokens use same temporal position `pos0`
4. Y and X positions are `pos0 + y` and `pos0 + x`

#### Add() for M-RoPE Text Tokens
For text tokens in M-RoPE batches:
- Uses `allocSize` as stride
- llama.cpp broadcasts first position to all dimensions (src_off=0 when batch.token != NULL)

#### MtmdChunk
Extended to include grid dimensions:
```go
type MtmdChunk struct {
    Embed  []float32
    Tokens []int
    Nx     int  // Grid width (for M-RoPE)
    Ny     int  // Grid height (for M-RoPE)
}
```

### 2. `runner/llamarunner/runner.go` - Batch Processing

#### input Structure
```go
type input struct {
    token   int
    embed   []float32
    imageNx int  // M-RoPE grid width
    imageNy int  // M-RoPE grid height
}
```

#### Position Calculation Methods
```go
func (inp *input) numTokens() int  // nx*ny for M-RoPE images, 1 otherwise
func (inp *input) numPos() int     // max(nx,ny) for M-RoPE images, 1 otherwise
func (inp *input) isImageMRoPE() bool
```

#### processBatch()
Modified to:
1. Use `numPos()` for position calculation
2. Use `AddImageMRoPE()` for M-RoPE images
3. Break from both loops after adding M-RoPE image (via `mropeBatchReady` flag)
4. Process batch immediately after M-RoPE image (no more tokens added)

#### Position Calculation
```go
pos := 0
for _, pi := range seq.cache.Inputs {
    pos += pi.numPos()  // NOT numTokens()!
}
for _, pi := range seq.pendingInputs {
    pos += pi.numPos()
}
```

### 3. `runner/llamarunner/image.go` - Image Context

#### BatchSize()
Returns 8192 for M-RoPE models to accommodate large images:
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

#### UsesMRoPE()
Queries mtmd to check if model uses M-RoPE:
```go
func (c *ImageContext) UsesMRoPE() bool {
    return c.mtmd.UsesMRoPE()
}
```

### 4. `runner/llamarunner/cache.go` - KV Cache

#### LoadCacheSlot()
Clears KV cache when prompt contains embeddings to prevent stale data issues:
```go
if hasEmbeddings && numPast > 0 {
    c.lc.KvCacheSeqRm(slot.Id, 0, -1)
    numPast = 0
}
```

## How llama.cpp Reads M-RoPE Positions

From `llama-batch.cpp`:
```cpp
for (size_t j = 0; j < n_pos_per_embd; ++j) {
    // For embeddings: src_off = j * batch.n_tokens
    // For text tokens: src_off = 0 (broadcast)
    size_t src_off = batch.token ? 0 : j*batch.n_tokens;
    udata->pos[j*n_tokens + i] = batch.pos[src_off + idxs[i]];
}
```

This means:
- **Text tokens** (`batch.token != NULL`): Reads only first position, broadcasts to all dimensions
- **Embeddings** (`batch.embd != NULL`): Reads with stride `batch.n_tokens`

## Critical Bug Fixes

### Bug 1: Wrong Position Stride
**Problem:** Original code used `allocSize` as stride for M-RoPE positions.
**Fix:** Use `batch.n_tokens` (final count) as stride in `AddImageMRoPE()`.

### Bug 2: Position Advance Formula
**Problem:** Used `numTokens()` (nx*ny) for position advance.
**Fix:** Use `numPos()` (max(nx,ny)) per mtmd.cpp:
```cpp
llama_pos mtmd_image_tokens_get_n_pos(const mtmd_image_tokens * image_tokens) {
    if (image_tokens->use_mrope_pos) {
        return std::max(image_tokens->nx, image_tokens->ny);
    }
    return image_tokens->n_tokens();
}
```

### Bug 3: Loop Break Issue
**Problem:** `break` after M-RoPE image only exited inner loop, allowing more tokens to be added.
**Fix:** Added `mropeBatchReady` flag to break outer loop too.

### Bug 4: Batch Size Too Small
**Problem:** Default batch size (512) couldn't fit large images (4028+ tokens).
**Fix:** `BatchSize()` returns 8192 for M-RoPE models.

### Bug 5: Context Length Calculation
**Problem:** Used `len(seq.cache.Inputs)` to count cached tokens, but M-RoPE images have many tokens per input.
**Fix:** Sum `numTokens()` for each cached input:
```go
cachedTokens := 0
for _, ci := range seq.cache.Inputs {
    cachedTokens += ci.numTokens()
}
```

## Testing

### Debug Log Messages
Enable with `OLLAMA_DEBUG=1`:
```
multimodal tokenize nChunks=... numEmbed=... usesMRoPE=true
image grid dimensions nx=53 ny=76 nx*ny=4028
M-RoPE image chunk embeddings=4028 nx=53 ny=76
added M-RoPE image to batch pos=X nx=53 ny=76 numTokens=4028 numPos=76
```

### Verify Positions
For image at `pos0=0`, `nx=53`, `ny=76`:
- Token (0,0): temporal=0, y=0, x=0
- Token (52,75): temporal=0, y=75, x=52
- Next text token position: `0 + max(53,76) = 76`

## Reference: llama.cpp Code

### mtmd-helper.cpp set_position_mrope_2d()
```cpp
void set_position_mrope_2d(llama_pos pos_0, int nx, int ny, llama_seq_id seq_id) {
    for (int y = 0; y < ny; y++) {
        for (int x = 0; x < nx; x++) {
            int i = y * nx + x;
            pos[i                     ] = pos_0;
            pos[i + batch.n_tokens    ] = pos_0 + y;
            pos[i + batch.n_tokens * 2] = pos_0 + x;
            pos[i + batch.n_tokens * 3] = 0;
        }
    }
}
```

### mtmd.cpp mtmd_image_tokens_get_n_pos()
```cpp
llama_pos mtmd_image_tokens_get_n_pos(const mtmd_image_tokens * image_tokens) {
    if (image_tokens->use_mrope_pos) {
        return std::max(image_tokens->nx, image_tokens->ny);
    }
    return image_tokens->n_tokens();
}
```

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    Image Processing Flow                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. MultimodalTokenize (llama.go)                           │
│     ├─ Get nx, ny from mtmd_image_tokens_get_nx/ny          │
│     ├─ Encode image → embeddings (nx*ny * n_embd_inp)       │
│     └─ Return MtmdChunk{Embed, Nx, Ny}                      │
│                                                              │
│  2. inputs() (runner.go)                                     │
│     └─ Create input{embed, imageNx, imageNy}                │
│                                                              │
│  3. processBatch() (runner.go)                               │
│     ├─ Calculate pos using numPos() for each cached input   │
│     ├─ Call AddImageMRoPE(embed, pos, nx, ny, ...)          │
│     ├─ Set mropeBatchReady = true                           │
│     └─ Break both loops                                      │
│                                                              │
│  4. AddImageMRoPE (llama.go)                                 │
│     ├─ Copy all embeddings to batch                         │
│     ├─ Update n_tokens = nTokensFinal                       │
│     └─ Set positions with stride = nTokensFinal:            │
│        pos[i]               = pos0      (temporal)          │
│        pos[i + nTokensFinal]   = pos0 + y  (row)            │
│        pos[i + 2*nTokensFinal] = pos0 + x  (col)            │
│        pos[i + 3*nTokensFinal] = 0         (unused)         │
│                                                              │
│  5. Decode (llama.cpp)                                       │
│     └─ Reads positions with stride = batch.n_tokens         │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```
