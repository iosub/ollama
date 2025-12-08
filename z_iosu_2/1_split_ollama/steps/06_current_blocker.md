# Step 6: Current Blocker - Main Vision Dimension Mismatch

## Current Status: BLOCKED

### What Works ✅
1. **FC weight loading**: 6/6 weights loaded (layers 8, 16, 24 × FC1, FC2)
2. **Deepstack projection**: 3/3 projections successful
3. **Projection output**: All produce correct shape [4096 2025]
4. **No FC nil warnings**: PatchMerger properly skipped

### What Fails ❌
**Error**: `ggml.c:2515: GGML_ASSERT(a->ne[d] == b->ne[d]) failed`

**Location**: During concatenation in `model.go` line 252:
```go
concatenated := allEmbeds[0].Concat(ctx, allEmbeds[1], 1)
```

## Problem Analysis

### Expected Concatenation
Per SPLIT_MODEL_ANALYSIS.md lines 31-39:

```
main_vision:    [n_tokens, 4096]  ← Should be this
deepstack_0:    [n_tokens, 4096]  ✅ Verified
deepstack_1:    [n_tokens, 4096]  ✅ Verified
deepstack_2:    [n_tokens, 4096]  ✅ Verified
────────────────────────────────
concatenated:   [n_tokens, 16384]  // n_embd × 4
```

### Actual Situation
- **Deepstack outputs**: 3× [4096 2025] ✅CORRECT
- **Main vision output**: Unknown, but causes dimension mismatch
- **GGML assertion**: Fails because tensors have different dimensions in some axis

### Dimension Flow
```
Vision Encoder (hiddenSize=1152)
    ↓
Spatial Merge (1152 × 4 = 4608)
    ↓
DeepstackMerger Projection (4608 → 4096) ✅ WORKS
    ↓
Main Vision ??? (1152? 4608? needs to be 4096)
```

## The Question
**How should main vision output be projected to 4096 dimensions?**

### Investigation Needed
1. **Check llama.cpp source**: How does qwen3vl.cpp handle main vision projection in split models?
2. **GGUF tensors**: Are there merger weights for main projection? (v.merger.fc1/fc2)
3. **Alternative**: Should main use one of the deepstack mergers?

### Evidence from Logs
- Split model has NO `v.merger` tensors in GGUF
- Only `v.deepstack.8/16/24.fc1/fc2.weight` exist
- PatchMerger exists in struct but has no FC weights

## Possible Solutions

### Option 1: Use Deepstack Merger for Main
Use one of the loaded deepstack mergers (e.g., layer 24's) to project main vision output.

**Pros**: Weights already loaded, same projection (4608→4096)
**Cons**: May not be architecturally correct

### Option 2: Different Dimension Handling
Main vision doesn't need projection - keep as 4608 or 1152 and handle differently in concatenation.

**Pros**: No additional loading needed
**Cons**: Doesn't match documented architecture (4× 4096 = 16384)

### Option 3: Missing Implementation
Split models use different approach than documented - need to study llama.cpp implementation.

## Next Actions Required
1. **Research llama.cpp**: Study `qwen3vl.cpp` to see how main vision projection is handled
2. **Test hypothesis**: Try using deepstack merger for main projection
3. **Check GGUF metadata**: Look for config indicating main projection strategy

## Files to Investigate
- llama.cpp source: `src/models/qwen3vl.cpp`
- GGUF structure: Check for merger-related metadata 
- Current code: `model/models/qwen3vl/model.go` lines 240-265 (concatenation logic)

---

**Status**: Need user input or additional research to proceed.
**Progress**: ~85% complete (FC loading works, projections work, blocked on concatenation)
