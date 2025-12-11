# CONFIRMED FACTS - Split Qwen3-VL

## FACT 1: Non-split model works correctly
- Evidence: `z_iosu_2/logs/nosplit14.log`
- Model `qwen3-vl:8b-instruct` returns `C25-16499-R` correctly
- Extracts complete product table, client info, financial summary
- Responds in Spanish as expected

## FACT 2: Split model does NOT work
- Evidence: Multiple test logs (17-31)
- Model `split-qwen3vl-8b:latest` returns corrupted or hallucinated data
- Says "corrupted or garbled text"
- Missing product table and client details
- Responds in English instead of Spanish
- Results vary between runs (hallucinations)

## FACT 3: Split model works with llama.cpp PR 13306
- Evidence: PR 13306 test logs
- The GGUF file is valid
- Problem is in Ollama Go code, not in the model

## FACT 4: Key differences between split and non-split
- temporalPatchSize: Split=1, NonSplit=2
- Patch Embedding: Split uses dual Conv2D, NonSplit uses Conv3D
- QKV Attention: Split uses fused attn_qkv, NonSplit uses separate q/k/v

## FACT 5: isSplitArchitecture detection works
- Evidence: Logs show `isSplitArchitecture=true` for split model
- Tensor loading with LoadSecondary works
- Deepstack layers [8, 16, 24] detected correctly

## FACT 6: The problem is in vision encoder
- The split model sees the image as blurry or noisy
- It loses significant visual information
- This is NOT a subtle difference - it is a major failure

## WHAT WE TRIED AND FAILED

1. Spatial merge order variations - No improvement
2. Inline vs Forward() position embedding - Same result
3. Different Permute indices - No improvement
4. Bilinear interpolation changes - No improvement
5. CHW vs HWC format changes - No improvement

## WHAT WE HAVE NOT TRIED

1. ~~Numerical tensor comparison between split and non-split~~ → **DONE (Test #33)**
2. Testing with a simple image (white square)
3. Deep debugging at GGML level

---

## FACT 10: Ollama Converter Analysis (2025-12-11 03:04)

**Source**: `convert/convert_qwen3vl.go` lines 83-96

The Ollama converter flattens patch_embed.weight and splits QKV for nosplit models.

---

## FACT 11: ROOT CAUSE FOUND (2025-12-11 03:22)

**Source**: Debug logging in test33.log

### The Problem

Split GGUF from Unsloth has **TWO** patch embedding kernels:
```
v.patch_embd.weight   → [768, 1152]     ← USED (flattened)
v.patch_embd.weight.1 → [16,16,3,1152]  ← NOT USED!
```

The Go code only loads and uses the **FIRST kernel**, losing half the information.

### Debug Evidence

| Stage | Nosplit | Split |
|-------|---------|-------|
| input_pixels | `[1536, 8100]` | `[768, 8100]` |
| after_patch_embed | `[1152, 8100]` | `[1152, 8100]` |

Split input has HALF the data (768 vs 1536) because temporalPatchSize=1.
The second kernel is meant to compensate, but it's never used.

### Fix Required

1. Add `PatchEmbedding1` field to VisionModel for second kernel
2. Flatten `v.patch_embd.weight.1` from 4D to 2D during loading  
3. Sum outputs of both kernels in Forward()

---

## FACT 12: Dual Kernel Fix Implemented But Still Fails (2025-12-11 03:57)

**Source**: Tests 35-38 logs

### What Was Implemented

1. ✅ `PatchEmbedding1` field added to VisionModel (`model_vision.go`)
2. ✅ `patchEmbedShape` updated to flatten `v.patch_embd.weight.1` (`ggml.go`)
3. ✅ Manual tensor loading in both `loadVisionModel()` and `HasProjector()` path (`model.go`)
4. ✅ Second kernel verified loading: logs show `has_kernel2=true`
5. ✅ Both kernels summed in Forward()

### Tests Performed

| Test | Approach | Result |
|------|----------|--------|
| #35-36 | Manual loading code not reached | `has_kernel2=false` |
| #37 | Fixed: added loading in HasProjector() path | `has_kernel2=true`, but still "corrupted text" |
| #38 | Use only first kernel (no sum) | Still "corrupted text" |

### Conclusion

**~~The dual kernel fix is NOT the solution.~~** ← **OUTDATED! See FACT 13 below.**

### Files Modified (Current State)

```
ml/backend/ggml/ggml.go          - patchEmbedShape handles .weight.1
model/models/qwen3vl/model.go    - Manual PatchEmbedding1 loading
model/models/qwen3vl/model_vision.go - PatchEmbedding1 field + Forward() logic
```

---

## FACT 13: BREAKTHROUGH - Dual Kernel on SAME Input Works! (2025-12-11 04:55)

**Source**: Test #43 in `z_iosu_2/logs/test43.log`

### The Fix

The correct implementation applies **BOTH kernels to the SAME input** (not split halves):

```go
// CORRECT (per clip.cpp reference):
h1 := m.PatchEmbedding.Weight.Mulmat(ctx, pixelValues)  // kernel0 → same input
h2 := m.PatchEmbedding1.Weight.Mulmat(ctx, pixelValues) // kernel1 → same input
hiddenStates = h1.Add(ctx, h2)                          // sum outputs
```

This matches clip.cpp implementation (lines 929-937).

### Evidence

Test #43 output:
- ✅ Model recognizes it's a **Spanish invoice** (FACTURA)
- ✅ Identifies **FAKTURA monte, Beasain, Guipuzcoa**
- ✅ Extracts invoice data structure
- ✅ Sees `C25-1649-R` (vs target `C25-16499-R` - very close!)

### Remaining Issues

1. Missing one digit: extracts `C25-1649-R` instead of `C25-16499-R`
2. **Article lines not extracted** - model says "Items: *Not visible*" while non-split extracts all
3. Confuses which field is "invoice number"
4. Responds in English instead of Spanish
5. Resolution/detail loss vs non-split (temporalPatchSize=1 vs 2)

### Key Learning

Previous tests split the input temporally (first half → kernel0, second half → kernel1).
This was WRONG. Both kernels are **parallel processors** of the SAME spatial data.

---

## FACT 14: Forcing temporalPatchSize=2 Breaks Everything (2025-12-11 05:16)

**Source**: Test #45 in `z_iosu_2/logs/test45.log`

### What We Tried

Hypothesis: Split model needs more input data like non-split's `[1536, N]`.

Changes:
1. Forced `temporalPatchSize=2` in ImageProcessor for split models
2. Split the `[1536, N]` input into halves for each kernel

### Result

**COMPLETE FAILURE** - Model outputs "garbled text" again, worse than Test #43.

### Conclusion

The split model's architecture with `temporalPatchSize=1` is fundamentally different from non-split.
Trying to make it behave like non-split by duplicating temporal frames doesn't work.
The remaining issues (missing digits, no article lines) require a different solution.
