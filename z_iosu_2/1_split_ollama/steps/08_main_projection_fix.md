# Step 8: Main Vision Projection Fix + Concatenation Dimension Fix

## Date: 2025-12-08

## Problem 1: Crash on Concatenation (FIXED)

After Step 7, all 3 deepstack projections work correctly (`[4608 2025] → [4096 2025]`), but the concatenation crashes with:

```
ggml.c:2515: GGML_ASSERT(a->ne[d] == b->ne[d]) failed
```

### Root Cause 1: Duplicate else-if Statement
In `model_vision.go` lines 375-376, there was a **duplicate else-if statement**:

```go
} else if len(deepstackStates) > 0 {
} else if len(deepstackStates) > 0 {  // BUG: This never executes!
```

### Root Cause 2: Missing Main Vision Projection
Main vision output had **4608 dimensions** but deepstacks had **4096**.

### Solution 1: Use DeepstackMerger[2] for main projection
Now uses `DeepstackMerger[2]` (layer 24) to project main vision from `4608 → 4096`.

---

## Problem 2: Hallucinations / Garbage Output (NEW FIX)

After fixing the crash, model runs but outputs:
```
It seems like you've pasted a corrupted or garbled text...
```

### Root Cause: Wrong Concatenation Dimension
Log showed:
```
main_shape="[4096 682]"
concatenated_shape="[4096 2728]"  ← Should be [16384 682]!
```

The concatenation was happening on **dim=1 (tokens)** instead of **dim=0 (features)**.

### GGML Tensor Layout
GGML uses **column-major** layout:
- Shape `[4096, 682]` means `[features, tokens]`
- Features (4096) are contiguous in memory
- To concatenate features, use `dim=0`

### Solution 2: Fix concatenation and splitting dimensions

**In `EncodeMultimodal()` (model.go ~line 252):**
```go
// OLD (wrong): Concat on dim=1
concatenated := allEmbeds[0].Concat(ctx, allEmbeds[1], 1)

// NEW (correct): Concat on dim=0 for features
concatenated := allEmbeds[0].Concat(ctx, allEmbeds[1], 0)
```

**In `Forward()` (model.go ~line 353):**
```go
// OLD (wrong): nEmbdFull from Dim(1)
nEmbdFull := visionOutputs.Dim(1)

// NEW (correct): nEmbdFull from Dim(0) since features are in dim 0
nEmbdFull := visionOutputs.Dim(0)
```
