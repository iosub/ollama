# llama.cpp PR #16745: Fix Qwen2.5 VL Cache Causal Masking

## Overview
This PR fixes causal masking issues in Qwen2.5 Vision-Language models by tracking actual KV cache positions instead of assuming consecutive token positions. This resolves inference errors when processing vision embeddings with non-consecutive position IDs.

## Source
- **Upstream PR**: https://github.com/ggml-org/llama.cpp/pull/16745
- **Applied**: October 25, 2025
- **Branch**: 12_07_mio
- **Commit**: e1a3d8557

## Problem Statement
Qwen2.5 VL models use vision embeddings with **non-consecutive position IDs**:
- Text tokens: positions 0, 1, 2, 3, ...
- Vision embeddings: positions 100, 200, 300, ...
- Continuation: positions 4, 5, 6, ...

The old implementation assumed **consecutive positions** for causal masking, causing:
1. Incorrect attention masks for vision tokens
2. Model inference failures
3. Poor generation quality with vision inputs

## Changes Made

### 1. Batch Structure Enhancement
**File**: `llama/llama.cpp/src/llama-batch.h`

**Added to `llama_ubatch` struct**:
```cpp
int32_t * kv_position_of_token;  // actual KV cache position for each token
```

**Added to `llama_ubatch::data_t` struct**:
```cpp
std::vector<int32_t> kv_position_of_token;  // storage for KV positions
```

**Purpose**: Track the actual KV cache position for each token in the batch, independent of temporal position.

### 2. Batch Initialization
**File**: `llama/llama.cpp/src/llama-batch.cpp`

**Commented out strict position validation** (lines 259-289):
```cpp
// GGML_ASSERT(ubatch.n_tokens  > 0);
// GGML_ASSERT(batch->pos[0] >= 0);
// for (int i = 1; i < ubatch.n_tokens; ++i) {
//     GGML_ASSERT(batch->pos[i] == batch->pos[i-1] + 1);  // No longer required
// }
```

**Added kv_position_of_token initialization** in 3 locations:
1. Standard batch split (line ~175)
2. Equal split mode (line ~230)
3. Batch sequence processing (line ~315)

**Added code**:
```cpp
ubatch.kv_position_of_token = ubatch_data->kv_position_of_token.data();
```

**Rationale**: Vision embeddings can have non-consecutive positions, validation was too strict.

### 3. KV Cache Causal Masking Rewrite
**File**: `llama/llama.cpp/src/llama-kv-cache.cpp`

**Function**: `llama_kv_cache_update_impl()`

**Old behavior**:
- Used token's temporal position for masking
- Assumed consecutive positions
- Couldn't handle vision embedding position jumps

**New behavior**:
- Builds `map_kv_to_batch` vector to track actual KV positions
- Updates `ubatch.kv_position_of_token[i]` with actual cache position
- Uses batch position indices for causal masking instead of temporal positions

**Key code**:
```cpp
// Build mapping from KV cache position to batch index
std::vector<int32_t> map_kv_to_batch(kv_self.size, -1);
for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
    for (int32_t s = 0; s < ubatch.n_seq_tokens[i]; ++s) {
        const llama_seq_id seq_id = ubatch.seq_id[i][s];
        // ... find cache position for this token ...
        ubatch.kv_position_of_token[i] = (int32_t)idx;  // Store actual position
        map_kv_to_batch[idx] = (int32_t)i;              // Map position to batch index
    }
}

// Causal masking using batch indices
for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
    if (has_mask) {
        int32_t pos_kv_i = ubatch.kv_position_of_token[i];
        for (int32_t s = 0; s < ubatch.n_seq_tokens[i]; ++s) {
            const llama_seq_id seq_id = ubatch.seq_id[i][s];
            for (uint32_t j = 0; j < ubatch.n_tokens; ++j) {
                int32_t pos_kv_j = ubatch.kv_position_of_token[j];
                // Check if j can attend to i using batch positions
                ubatch.mask[i*ubatch.n_tokens + j] = (
                    ubatch_seq_id_cmp(ubatch, j, seq_id) && 
                    pos_kv_j <= pos_kv_i  // Causal masking based on KV position
                );
            }
        }
    }
}
```

**Benefits**:
- Handles non-consecutive positions correctly
- Vision embeddings masked properly
- Preserves causal attention semantics

### 4. M-RoPE Position Calculation
**File**: `llama/llama.cpp/tools/mtmd/mtmd.cpp`

**Function**: `llama_mtmd_input_text_template::get_position()`

**Changed** (line 113):
```cpp
// Old: return 1;  // Always returned 1 for images
// New:
return std::max(nx, ny);  // Return max(width, height) for proper image dimensions
```

**Rationale**: Qwen VL uses image dimensions for RoPE position calculation. Returning 1 broke positional encoding for vision embeddings.

### 5. Documentation Update
**File**: `llama/llama.cpp/tools/mtmd/mtmd.h`

**Updated comment** (line 112):
```cpp
// Old comment: return temporal position (usually 1 for images)
// New comment: 
// return temporal position for embeddings
// Note: Qwen VL models expect max(image_width, image_height) here
//       to properly calculate M-RoPE positions for vision embeddings
```

## Technical Details

### Position Tracking Flow
1. **Batch Creation**: Initialize `kv_position_of_token` array
2. **KV Cache Update**: 
   - Find actual cache position for each token
   - Store in `ubatch.kv_position_of_token[i]`
3. **Masking**:
   - Use `kv_position_of_token` for causal checks
   - Token j can attend to token i if `pos_kv_j <= pos_kv_i`

### Example: Vision Processing
**Input sequence**:
```
Token 0: "Describe"     -> pos=0,   kv_pos=0
Token 1: "this"         -> pos=1,   kv_pos=1
Token 2: <vision_emb_0> -> pos=100, kv_pos=2  // Non-consecutive!
Token 3: <vision_emb_1> -> pos=101, kv_pos=3
Token 4: "image"        -> pos=2,   kv_pos=4  // Position resets
```

**Causal mask** (kv_position_of_token based):
```
     0  1  2  3  4
0 [  T  F  F  F  F ]  Token 0 sees only itself
1 [  T  T  F  F  F ]  Token 1 sees 0,1
2 [  T  T  T  F  F ]  Vision 0 sees 0,1,itself
3 [  T  T  T  T  F ]  Vision 1 sees 0,1,2,itself
4 [  T  T  T  T  T ]  Token 4 sees all previous
```

Without this fix, vision tokens would have incorrect masks based on pos=100,101.

### M-RoPE Position Fix
**Qwen VL M-RoPE** uses 3D positional encoding:
- **Temporal dimension**: Token sequence position
- **Height dimension**: For vision, use image height
- **Width dimension**: For vision, use image width

**Old code**: `return 1` made all vision embeddings have position=1
**New code**: `return max(nx, ny)` uses actual image dimensions
**Result**: Correct RoPE frequencies for vision embeddings

## Benefits
1. **Correct Vision Processing**: Qwen VL models work properly
2. **Flexible Position IDs**: Supports non-consecutive positions
3. **Maintains Causality**: Attention masking still correct
4. **M-RoPE Fix**: Vision embeddings get proper positional encoding
5. **No Performance Impact**: Minimal computational overhead

## Testing Recommendations

### Basic Vision Test
```bash
ollama run qwen2.5-vl:7b "Describe this image" --image test.jpg
```

### Multi-Image Test
```bash
ollama run qwen2.5-vl:7b "Compare these images" --image img1.jpg --image img2.jpg
```

### Position Tracking Verification
Add debug logging:
```cpp
for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
    printf("Token %d: pos=%d, kv_pos=%d\n", 
           i, ubatch.pos[i], ubatch.kv_position_of_token[i]);
}
```

### Expected Behavior
- No inference errors with vision inputs
- Coherent image descriptions
- Proper multi-image reasoning
- No position validation assertions

## Models Affected
- **Qwen2.5-VL** (all sizes: 3B, 7B, 32B, 72B)
- **Qwen-VL** (original)
- **Qwen2-VL**
- Any vision-language model using non-consecutive position IDs

## Known Limitations
- Assumes vision embeddings use higher position IDs than text
- M-RoPE calculation depends on correct image dimensions
- Batch size limited by KV cache size (standard limitation)

## Files Modified
```
llama/llama.cpp/src/llama-batch.h
llama/llama.cpp/src/llama-batch.cpp
llama/llama.cpp/src/llama-kv-cache.cpp
llama/llama.cpp/tools/mtmd/mtmd.cpp
llama/llama.cpp/tools/mtmd/mtmd.h
```

## Statistics
- **Files Changed**: 5
- **Insertions**: 111
- **Deletions**: 95
- **Net Change**: +16 lines

## Related Issues
- Fixes vision processing errors in Qwen VL models
- Resolves "position assertion failed" errors
- Improves multi-modal inference quality

## References
- **Upstream Discussion**: https://github.com/ggml-org/llama.cpp/issues/16207
- **Qwen VL Discussion**: https://github.com/ggml-org/llama.cpp/issues/16207#issuecomment-3443868720
- **Related Work**: https://github.com/LETS-BEE/llama.cpp/commits/qwen3vl/
