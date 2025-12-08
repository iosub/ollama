# Step 5: Projection Verification & Debugging

## Added Detailed Logging
**File**: `model/models/qwen3vl/model_vision.go` lines 143-152

Changed from single-line projection to step-by-step with logging:

```go
// Before
if m.FC1 != nil && m.FC2 != nil {
    slog.Debug("DeepstackMerger projecting", "input_dim", hiddenSize)
    return m.FC2.Forward(ctx, m.FC1.Forward(ctx, visionOutputs).GELU(ctx))
}

// After
if m.FC1 != nil && m.FC2 != nil {
    slog.Debug("DeepstackMerger projecting", "input_dim", hiddenSize, "input_shape", visionOutputs.Shape())
    fc1Out := m.FC1.Forward(ctx, visionOutputs)
    slog.Debug("FC1 output", "shape", fc1Out.Shape())
    activated := fc1Out.GELU(ctx)
    slog.Debug("After GELU", "shape", activated.Shape())
    result := m.FC2.Forward(ctx, activated)
    slog.Debug("FC2 output", "shape", result.Shape())
    return result
}
```

## Test Results (go13.log, go14.log)

### ✅ All Projections Work Perfectly

**Projection 1 (layer 8)**:
```
DEBUG DeepstackMerger projecting input_dim=4608 input_shape=[4608 2025]
DEBUG FC1 output shape=[4608 2025]
DEBUG After GELU shape=[4608 2025]
DEBUG FC2 output shape=[4096 2025] ✅
```

**Projection 2 (layer 16)**:
```
DEBUG DeepstackMerger projecting input_dim=4608 input_shape=[4608 2025]
DEBUG FC1 output shape=[4608 2025]
DEBUG After GELU shape=[4608 2025]
DEBUG FC2 output shape=[4096 2025] ✅
```

**Projection 3 (layer 24)**:
```
DEBUG DeepstackMerger projecting input_dim=4608 input_shape=[4608 2025]
DEBUG FC1 output shape=[4608 2025]
DEBUG After GELU shape=[4608 2025]
DEBUG FC2 output shape=[4096 2025] ✅
```

### Key Findings
1. ✅ Input shape correct: [4608 2025] (spatial merged dimension)
2. ✅ FC1 maintains shape: [4608 2025]
3. ✅ GELU maintains shape: [4608 2025]
4. ✅ FC2 projects correctly: [4096 2025] - **matches target LLM dimension**

### Where Error Occurs
```
DEBUG FC2 output shape=[4096 2025]  ← Last successful operation
ggml.c:2515: GGML_ASSERT(a->ne[d] == b->ne[d]) failed  ← Crash here
```

**NO "Concatenated vision + deepstack embeddings" message** → Error happens BEFORE concatenation in model.go

## Conclusion
All 3 deepstack projections work correctly (4608→4096). The error occurs during concatenation, suggesting main vision output has incompatible dimensions.

## Next Step
→ See `06_current_blocker.md`
