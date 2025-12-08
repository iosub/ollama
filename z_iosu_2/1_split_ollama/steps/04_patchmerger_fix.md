# Step 4: PatchMerger FC Nil Check

## Problem Discovered
After Step 3, go12.log showed:
- 3 successful "DeepstackMerger projecting" messages ✅
- 1 "DeepstackMerger FC layers are nil" warning ❌
- Error: `ggml.c:2515: GGML_ASSERT(a->ne[d] == b->ne[d]) failed`

## Root Cause
There's a separate `PatchMerger` (not DeepstackMerger) at the end of VisionModel.Forward:

```go
// model_vision.go lines 357-359
if m.PatchMerger != nil {
    hiddenStates = m.PatchMerger.Forward(ctx, hiddenStates, false, m.VisionOptions)
}
```

This PatchMerger exists but has no FC weights in split models. When Forward is called without FC weights, it returns unprojected tensor with wrong dimensions (4608 instead of expected 4096).

## Code Change
**File**: `model/models/qwen3vl/model_vision.go` line 358

**Before**:
```go
// PatchMerger may be nil for split models
if m.PatchMerger != nil {
    hiddenStates = m.PatchMerger.Forward(ctx, hiddenStates, false, m.VisionOptions)
}
```

**After**:
```go
// PatchMerger may be nil for split models  
// Even if non-nil, skip if it doesn't have FC weights (would return wrong dimensions)
if m.PatchMerger != nil && m.PatchMerger.FC1 != nil && m.PatchMerger.FC2 != nil {
    hiddenStates = m.PatchMerger.Forward(ctx, hiddenStates, false, m.VisionOptions)
}
```

## Why This Fix
In VisionPatchMerger.Forward:
```go
if m.FC1 != nil && m.FC2 != nil {
    // Project: 4608 → 4096
    return m.FC2.Forward(ctx, m.FC1.Forward(ctx, visionOutputs).GELU(ctx))
}
// This shouldn't happen for Qwen3-VL split models
slog.Warn("DeepstackMerger FC layers are nil", "dim", hiddenSize)
return visionOutputs  // Returns UNPROJECTED tensor (wrong dimensions!)
```

Without FC weights, it returns unprojected tensor causing dimension mismatch downstream.

## Test Results (go12.log)
✅ No more "FC layers are nil" warnings
✅ All 3 DeepstackMerger projections complete successfully:
```
DEBUG DeepstackMerger projecting input_dim=4608
DEBUG DeepstackMerger projecting input_dim=4608
DEBUG DeepstackMerger projecting input_dim=4608
```

❌ Still crashes with ggml.c:2515 (but different reason - see Step 5)

## Impact
Prevents PatchMerger from being called when it can't project correctly, avoiding one source of dimension errors.

## Next Step
→ See `05_projection_verification.md`
