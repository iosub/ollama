# Qwen3-VL Split GGUF Implementation in Ollama Go Runner

## Background

Qwen3-VL models use a "deepstack" vision architecture where vision embeddings are processed through multiple projection layers that feed into different LLM layers. The official llama.cpp implementation (PR #13305) supports this, but Ollama's Go runner needs modifications to handle split GGUF models correctly.

## Split vs Non-Split Models

**Non-Split (NOSPLIT)**: Single .gguf file containing all weights
- Works correctly - vision processing functions properly
- Generates correct responses to images

**Split**: Two .gguf files (text model + vision encoder/projector)
- Main model: Text LLM weights
- Vision file (mmproj): Vision encoder + deepstack projection layers
- **Current status**: Crashes with assertion failure or hallucinates

## Technical Architecture

### Deepstack Vision Architecture

From llama.cpp `qwen3vl.cpp`:

1. **Vision Encoder Output**: Generates embeddings of dimension `n_embd_vision` (1152 for this model)
2. **Spatial Merge**: Merges 2×2 patches → dimension becomes `n_embd_vision × spatialMergeSize²` = 1152 × 4 = **4608**
3. **Deepstack Layers**: 3 projection layers (at vision encoder layers 8, 16, 24) that each:
   - Input: 4608-dim embeddings
   - Output: 4096-dim embeddings (matching LLM `n_embd`)
   - Implemented via FC1 (4608→4608) + GELU + FC2 (4608→4096)

4. **Concatenation**: 
   ```
   main_vision:    [n_tokens, 4096]
   deepstack_0:    [n_tokens, 4096]  
   deepstack_1:    [n_tokens, 4096]
   deepstack_2:    [n_tokens, 4096]
   ────────────────────────────────
   concatenated:   [n_tokens, 16384]  // n_embd × 4
   ```

5. **LLM Integration**: During LLM forward pass:
   - Main embeddings go to layer 0
   - Deepstack features are ADDED to hidden states at layers 0, 1, 2

## Problem Analysis

### Issue 1: Deepstack Features Not Generated

**Expected behavior**: Vision encoder should generate 4 separate tensors:
- 1 main vision output
- 3 deepstack outputs (one per projection layer)

**Actual behavior**: Only main vision output generated, deepstack array is empty

**Root cause**: `DeepstackMerger` FC1/FC2 weights not loading from split GGUF

### Issue 2: Dimension Mismatch

**Log evidence**:
```
main_shape=[4608 2025]        ← Should be [4096 × tokens]
concatenated_shape=[4608 8100] ← Should be [16384 × tokens]
```

**Problem**: Vision embeddings are 4608-dim (unprojected) instead of 4096-dim (projected)

**Consequence**: Creating tensor views with wrong dimensions → assertion failure at `ggml.c:1669`:
```c
GGML_ASSERT(view_src == NULL || data_size == 0 || 
            data_size + view_offs <= ggml_nbytes(view_src)) failed
```

This assertion prevents reading beyond tensor boundaries. When commented, model reads corrupt memory → hallucinations.

### Issue 3: FC Weights Not Loading

**GGUF contains**:
```
v.deepstack.8.fc1.weight  [4608, 4608]
v.deepstack.8.fc2.weight  [4608, 4096]
v.deepstack.8.norm.weight [4608]
v.deepstack.16.* (same structure)
v.deepstack.24.* (same structure)
```

**Struct definition**:
```go
type VisionPatchMerger struct {
    Norm *nn.LayerNorm `gguf:"norm,alt:ln_merger"`
    FC1  *nn.Linear    `gguf:"fc1,alt:linear_fc1,alt:ffn_up"`  
    FC2  *nn.Linear    `gguf:"fc2,alt:linear_fc2,alt:ffn_down"`
}
```

**Problem**: vision_bridge populates structs using gguf tags + tensor names. Despite tag saying `fc1`, weights still not loading into struct.

**Runtime result**: All DeepstackMerger instances have FC1=nil, FC2=nil

## Code Locations

### Key Files Modified

1. **`model/models/qwen3vl/model.go`**
   - Lines 114-122: Pre-initialize DeepstackMerger array (3 elements)
   - Lines 152-177: EncodeMultimodal - concatenates vision + deepstack
   - Lines 260-289: Forward - splits concatenated embeddings and adds to LLM layers

2. **`model/models/qwen3vl/model_vision.go`**
   - Lines 124-128: VisionPatchMerger struct with gguf tags
   - Lines 142-150: Forward method (attempts projection, falls back if FC nil)
   - Lines 346-358: VisionModel.Forward generates deepstack states
   - Lines 371-383: newVisionModel pre-allocates DeepstackMerger array
   - Lines 397-401: Sets deepstackVisualIndexes = [0,1,2]

3. **`ml/backend/ggml/ggml/src/ggml.c`**
   - Line 1669: Assertion that fails with wrong dimensions

## What's Missing

After 10 hours of debugging, the fundamental issue is:

**The gguf tag-based tensor loading mechanism (vision_bridge) is not populating the DeepstackMerger FC weights, despite:**
- Weights existing in GGUF
- Struct tags being correct
- Array being pre-initialized

**Unknown**: Why vision_bridge fails to load these specific tensors when:
- It successfully loads vision encoder layers (v.blk.*)
- It successfully loads other vision tensors (v.post_ln.*, etc.)
- The naming pattern matches (v.deepstack.8.fc1.weight)

## Critical Question for Next Session

The tensor loading mechanism needs investigation:
- How does vision_bridge map gguf tags to tensor names?
- Why does it work for v.blk.* but not v.deepstack.*?
- Is there a prefix/scope issue with DeepstackMerger being in an array?
- Does vision_bridge support loading into array elements?

**Hypothesis**: vision_bridge may not support populating struct fields that are inside array elements. The DeepstackMerger is an array `[]*VisionPatchMerger`, and vision_bridge might only populate top-level fields.
