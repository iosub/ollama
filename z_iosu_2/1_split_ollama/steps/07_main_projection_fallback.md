# Step 7: Main Vision Projection Fallback & Verification

## Problem Analysis
Previous state (Step 6) was blocked by concatenation failure due to dimension mismatch.
- Deepstack outputs: [4096, 2025]
- Main vision output: [1152, 2025] (after post_ln)
- Target: All must be [4096, 2025] to concatenate into [16384, 2025]

Split GGUF models lack `v.merger` (PatchMerger) weights, leaving no obvious way to project the main vision output from 1152 → 4096.

## Solution Implemented
Implemented a fallback strategy: **Reuse DeepstackMerger weights to project main vision output**.

**Code Change**:
`model/models/qwen3vl/model_vision.go` lines 365-374:

```go
} else if len(m.DeepstackMerger) > 0 && m.DeepstackMerger[0] != nil && m.DeepstackMerger[0].FC1 != nil {
    // Fallback for split models: use DeepstackMerger[0] to project main output
    // This ensures dimensions match (4608 -> 4096) for concatenation
    slog.Debug("Projecting main vision via DeepstackMerger[0] (fallback)", "input_shape", hiddenStates.Shape())
    hiddenStates = m.DeepstackMerger[0].Forward(ctx, hiddenStates, true, m.VisionOptions)
    slog.Debug("Projected main vision via DeepstackMerger[0]", "shape", hiddenStates.Shape())
}
```

This applies the same logic as deepstack layers:
1. Reshape 1152 → 4608 (spatial merge)
2. Project 4608 → 4096 (FC1 + GELU + FC2)

## Test Results (go15.log)
### ✅ Success
- **No Crash**: Model loads successfully
- **Concatenation works**: Implicitly confirmed as execution proceeds to inference
- **Response generated**: Model producing output

### ❌ Issue: Hallucinations
- Output is Chinese text / garbage (Hallucinations)
- Indicates that while "tensor plumbing" is correct (dimensions match), the **semantic content** is wrong.
- Likely cause: Reusing `DeepstackMerger[0]` (from layer 8) to project final vision output (layer 27) applies incorrect transformation weights.

## Hypothesis
Using weights trained for Layer 8 features to project Layer 27 features corrupts the visual embeddings.
Alternative ideas:
1. Use `DeepstackMerger[2]` (Layer 24) weights? (Closer to end)
2. Is there a different projection mechanism we missed?
3. Should main vision be projected at all, or handled differently?

## Next Step
Investigate better projection strategy for main vision output to fix hallucinations.
→ See `08_fixing_hallucinations.md`
