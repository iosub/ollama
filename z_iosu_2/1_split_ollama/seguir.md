# Split Qwen3-VL Fix Progress

## 📅 Date: 2025-12-10

---

## ⚠️ CURRENT STATUS

**The split model is NOT working correctly.** The non-split (unified) model works perfectly.

### Results Comparison

| Model | Result | Invoice Number |
|-------|--------|----------------|
| **Non-split** (qwen3-vl:8b-instruct) | ✅ Extracts ALL data correctly | **C25-16499-R** ✅ |
| **Split** (split-qwen3vl-8b:latest) | ❌ "corrupted or garbled text" + hallucinations | ~25-1649-R (partial) |

### What the Non-split model extracts (correct):
- Company: MONTTE S.L.
- Client: KORTA, S.A. with full address
- **Complete product table** with codes, descriptions, quantities, prices
- Financial summary: 309.03€ + VAT = 373.93€
- Exact invoice number: C25-16499-R

### What the Split model extracts (incorrect):
- Partial and mixed information
- Says "corrupted or garbled text"
- **Does NOT extract product table**
- Missing client information
- Responds in English instead of Spanish
- **Inconsistent results between runs** (hallucinations)

---

## Test History

| Test | What Changed | Result |
|------|--------------|--------|
| #14 | Indexed bilinear position embedding | "corrupted" message |
| #15 | Reverted CHW→HWC | Same as #14 |
| #16 | Added spatial merge after indexed bilinear | "monte" repeated |
| #17 | Native Interpolate + spatial merge | "no corruption" - **BEST** |
| #23 | Various spatial merge tweaks | "corrupted" - regression |
| #25 | PositionEmbedding.Forward() | **Extracted "25-1649-R"** - Best approximation |
| #29 | Restored Test #17 approach | Still "corrupted" |
| #30 | Inline position embedding with same spatial merge | "corrupted" |
| #31 | Reverted to PositionEmbedding.Forward() | Similar to #24, "corrupted" |

---

## Technical Analysis

### Key difference between Split and Non-split:

1. **Non-split (WORKS):**
   - Uses `Conv3D` for patch embedding
   - `temporalPatchSize = 2`
   - Position embedding handled internally by `PositionEmbedding.Forward()`

2. **Split (NOT WORKING):**
   - Uses `dual Conv2D` (kernel0 + kernel1 summed)
   - `temporalPatchSize = 1`
   - Manual 2x2 spatial merge applied
   - Interpolated position embedding + spatial merge

### Possible causes:

1. **Permute operation order** - may be reordering patches incorrectly
2. **Bilinear interpolation** - may have subtle numerical differences
3. **Spatial merge pattern** - 2x2 pattern may not match clip.cpp
4. **Numerical precision** - dual Conv2D vs unified Conv3D

---

## Modified Files (Working Directory)

```
M ml/backend/ggml/ggml.go
M model/models/qwen3vl/imageprocessor.go
M model/models/qwen3vl/model.go
M model/models/qwen3vl/model_vision.go
```

---

## Suggested Next Steps

1. **Deep numerical debugging:** Compare tensor values between split and non-split
2. **Review clip.cpp line by line:** Especially `build_qwen3vl()` and `resize_position_embeddings()`
3. **Test with simple image:** (white square) to isolate the problem
4. **Verify m.gridPerSide:** Confirm it's correct for split models

---

## Reference Files

- `model/models/qwen3vl/model_vision.go` - Main modified file
- `z_iosu_2/funcionaaaaaaaaa/ollama/` - Code copy that WORKS (non-split)
- `z_iosu_2/logs/*.log` - Test logs
- `z_iosu_2/logs/viejos/resultadoBien.md` - Correct result from non-split model

---

## Conclusion

The problem is NOT a "subtle difference". The split model's vision encoder is losing significant information. The model sees the image as if it were very blurry or noisy, losing most of the invoice details.

---

## Test #32 - clip.cpp Strategy (2025-12-11 02:47) - FAILED

### What We Tried
Based on clip.cpp analysis, tried:
1. Dual Conv2D + sum
2. Spatial merge 2x2
3. Keep 4D kernels for Conv2D

### Result
**FAILED** - Model still says "gray background with white noise"

### Why It Failed
**WRONG APPROACH**: clip.cpp is for llama.cpp, NOT for Ollama Go engine.
The Ollama Go runtime expects the format from `convert/convert_qwen3vl.go`, not clip.cpp.

**Reverted all changes at 03:04**

---

## Test #33 - Corrected Strategy (2025-12-11 03:04)

### Key Discovery: Ollama Converter Analysis

Analyzed `convert/convert_qwen3vl.go` (lines 83-96):

```go
// Line 83-88: Split attn_qkv → attn_q, attn_k, attn_v
case strings.Contains(t.Name(), "attn_qkv"):
    out = append(out, slices.Collect(splitDim(t, 0,
        split{Replacer: strings.NewReplacer("attn_qkv", "attn_q")},
        ...
    ))...)

// Line 89-96: Flatten patch_embed.weight 4D → 3D
case strings.Contains(t.Name(), "patch_embed"):
    Shape: append([]uint64{shape[0] * shape[1]}, shape[2:]...)  // [16,16,3,1152] → [256,3,1152]
```

### Format Comparison

| Tensor | Split GGUF (Unsloth) | Nosplit GGUF (Ollama) | Go Runtime Expects |
|--------|---------------------|----------------------|-------------------|
| QKV | `attn_qkv` fused | `attn_q/k/v` split | **Split** |
| Patch | 4D `[16,16,3,1152]` | 3D `[256,3,1152]` | **3D flattened** |

### Correct Strategy

The runtime ALREADY has flattening logic (`patchEmbedShape` in ggml.go).
Added debug logging to compare tensor shapes at key points.

### Debug Results (2025-12-11 03:22)

With `OLLAMA_DEBUG_VISION=1`:

| Stage | Nosplit | Split |
|-------|---------|-------|
| input_pixels | `[1536, 8100]` | `[768, 8100]` |
| after_patch_embed | `[1152, 8100]` | `[1152, 8100]` |

### ROOT CAUSE FOUND

Split GGUF has TWO patch embedding kernels:
```
v.patch_embd.weight   → [768, 1152]     ← USED
v.patch_embd.weight.1 → [16,16,3,1152]  ← NOT USED!
```

The Go code only uses ONE kernel, losing half the information!

### Result
**ROOT CAUSE CONFIRMED** - Fix: add PatchEmbedding1 field and sum both kernel outputs.

---

## Test #34 - Dual Kernel Implementation (2025-12-11 03:24)

### What Was Changed
1. Added `PatchEmbedding1` field to VisionModel (`model_vision.go`)
2. Updated `patchEmbedShape` to flatten `v.patch_embd.weight.1` (`ggml.go`)
3. Added manual loading in `loadVisionModel()` (`model.go`)
4. Sum both kernel outputs in Forward()

### Result
**FAILED** - Compile succeeded but code path not reached. `has_kernel2=false` in logs.

---

## Test #35-36 - Debugging Kernel Loading (2025-12-11 03:40)

### What Was Changed
Changed `slog.Debug` to `slog.Info` for tensor loading diagnostics.

### Result
**FAILED** - Manual loading code in `loadVisionModel()` never executed because `ensureVisionReady()` has early-return when `HasProjector()` is true.

---

## Test #37 - Fixed Loading Path (2025-12-11 03:47)

### What Was Changed
Added duplicate tensor loading code in the `if m.HasProjector()` block of `ensureVisionReady()`, not just in `loadVisionModel()`.

### Result
**PARTIAL SUCCESS** - `has_kernel2=true` now appears in logs! Second kernel IS loading.
But model still outputs "corrupted text" and wrong data.

---

## Test #38 - Single Kernel Only (2025-12-11 03:55)

### What Was Changed
Changed to use ONLY the first kernel for 768-dim input (hypothesis: second kernel is for video frames).

### Result
**FAILED** - Same "corrupted text" output. Problem is NOT the kernel combination logic.

---

## 🚨 CURRENT STATE (2025-12-11 03:58)

### What Works
- ✅ Second kernel loading (`has_kernel2=true`)
- ✅ Kernel flattening from 4D to 2D
- ✅ Code compiles without errors

### What Still Fails
- ❌ Model outputs "corrupted text"
- ❌ Cannot extract invoice number `C25-16499-R`
- ❌ Responds in English instead of Spanish

### Remaining Hypotheses

1. **ImageProcessor pixel layout** - Input format may be wrong
2. **Position embeddings** - Interpolation or order issues
3. **Attention layers** - fused QKV vs split QKV handling
4. **Multimodal projector** - FC1/FC2 projection issues
5. **Fundamental GGUF incompatibility** - Split model needs different handling

### Files Modified (for next session)

```
ml/backend/ggml/ggml.go
  - patchEmbedShape() handles v.patch_embd.weight.1

model/models/qwen3vl/model.go
  - Manual PatchEmbedding1 loading in HasProjector() path
  - Manual PatchEmbedding1 loading in loadVisionModel()

model/models/qwen3vl/model_vision.go
  - PatchEmbedding1 field added to VisionModel struct
  - Forward() handles dual kernel logic (sum or single)
```

### Next Steps for New Session

1. Compare **numerical values** of tensors at each stage (split vs nosplit)
2. Focus on **ImageProcessor** - verify pixel layout matches expectations
3. Check **position embedding interpolation** against Python reference
4. Consider **reverting to single kernel** and focus on other pipeline issues

---

## Test #39 - Dual Kernel Sum for Static Images (2025-12-11 04:05)

### What Was Changed

Modified `model_vision.go` lines 355-363:

**Before (Test #38):**
```go
// For static images, use only the FIRST kernel.
h1 := m.PatchEmbedding.Weight.Mulmat(ctx, pixelValues)
hiddenStates = h1
```

**After (Test #39):**
```go
// For static images, use BOTH kernels with same input and SUM outputs.
h1 := m.PatchEmbedding.Weight.Mulmat(ctx, pixelValues)
h2 := m.PatchEmbedding1.Weight.Mulmat(ctx, pixelValues)
hiddenStates = h1.Add(ctx, h2)
```

### Hypothesis

Split model dual kernels simulate Conv3D's temporal processing. For static images, both kernels should see the same frame (temporal duplication).

### Result

**FAILED** - Model still outputs "text is corrupted or garbled". Dual kernel sum does not fix the issue.

The problem is NOT in the patch embedding dual kernel logic.

---

## Test #43 - Apply BOTH Kernels to SAME Input (2025-12-11 04:55)

### What Was Changed

**Files Modified:**

1. `model/models/qwen3vl/model.go` (lines 184-188):
   - REVERTED Test #42's forced `temporalPatchSize=2`
   - Split models now use original `temporalPatchSize=1`

2. `model/models/qwen3vl/model_vision.go` (lines 356-377):
   - Apply BOTH kernels to the SAME `[768, N]` input
   - Sum the outputs (per clip.cpp reference implementation)

**Before (Test #42 - wrong):**
```go
// Split input into halves, kernel0→first half, kernel1→second half
firstHalf := pixelValues.View(ctx, 0, kernelDim*numPatches)
h1 := m.PatchEmbedding.Weight.Mulmat(ctx, firstHalf)
secondHalf := pixelValues.View(ctx, kernelDim*numPatches, kernelDim*numPatches)
h2 := m.PatchEmbedding1.Weight.Mulmat(ctx, secondHalf)
hiddenStates = h1.Add(ctx, h2)
```

**After (Test #43 - correct):**
```go
// Apply BOTH kernels to SAME input and sum (per clip.cpp)
h1 := m.PatchEmbedding.Weight.Mulmat(ctx, pixelValues)
h2 := m.PatchEmbedding1.Weight.Mulmat(ctx, pixelValues)
hiddenStates = h1.Add(ctx, h2)
```

### Hypothesis

Based on clip.cpp reference (lines 929-937), both kernels should process the SAME input:
```cpp
ggml_tensor * inp = ggml_conv_2d(ctx0, model.patch_embeddings_0, inp_raw, ...);
auto inp_1 = ggml_conv_2d(ctx0, model.patch_embeddings_1, inp_raw, ...);
inp = ggml_add(ctx0, inp, inp_1);
```

### Result

**MAJOR SUCCESS!** 🎉

The model now:
- ✅ Sees the image correctly (no more "corrupted text")
- ✅ Recognizes it's a Spanish invoice (FACTURA)
- ✅ Identifies FAKTURA monte, Beasain, Guipuzcoa
- ✅ Extracts invoice-related data
- ✅ Mentions `C25-1649-R` (very close to target `C25-16499-R`)

**Remaining issues:**
- ❓ Extracts `C25-1649-R` instead of `C25-16499-R` (missing one "9")
- ❓ Confuses which field is the "invoice number"
- ❓ Responds in English instead of Spanish

### Log Reference

`z_iosu_2/logs/test43.log`

---

## Test #45 - Force temporalPatchSize=2 + Split Input for Dual Kernels (2025-12-11 05:16)

### What Was Changed

**Files Modified:**

1. `model/models/qwen3vl/model.go`:
   - Forced `m.ImageProcessor.temporalPatchSize = 2` for split models
   - This makes ImageProcessor produce `[1536, N]` input

2. `model/models/qwen3vl/model_vision.go`:
   - Split the `[1536, N]` input into two `[768, N]` halves
   - Apply `kernel0` to first half, `kernel1` to second half, then sum

### Hypothesis

The split model needs more input data (like `[1536, N]` from temporalPatchSize=2) to capture fine details. The previous Test #43 with `[768, N]` worked but missed details.

### Result

**FAILED** ❌

Model outputs "garbled text" again - worse than Test #43. Forcing temporalPatchSize=2 breaks the split model vision pipeline.

### Log Reference

`z_iosu_2/logs/test45.log`

---

## Test #46 - Revert to Test #43 (2025-12-11 05:20)

### What Was Changed

- Reverted `model.go` to original temporalPatchSize sync (no forced =2)
- `model_vision.go` remains with dual kernel logic supporting both `[768, N]` and `[1536, N]` inputs

### Result

Confirmed working with Test #43 behavior - model can process images.

### Log Reference

`z_iosu_2/logs/test46.log`

---

## Current State (2025-12-11 05:23)

**Working (Test #43 logic):**
- Both kernels applied to SAME `[768, N]` input and summed
- Model sees image, extracts some data

**Still broken:**
- Missing digits in extracted numbers (e.g., `C25-1649-R` vs `C25-16499-R`)
- **Article lines not extracted** - model says "Items: *Not visible*" while non-split extracts all lines
- Model responds in English instead of Spanish
- Resolution/detail loss compared to non-split model (temporalPatchSize=1 vs 2)

**Next steps:**
- Investigate numerical differences in vision pipeline
- Compare tensor values between split and non-split models
