# Step 2: Fixed deepstackVisualIndexes

## What Was Done
Changed `deepstackVisualIndexes` from `[0,1,2]` to `[8,16,24]` to match actual vision encoder layer IDs.

## Code Change
**File**: `model/models/qwen3vl/model_vision.go` line 406

**Before**:
```go
deepstackVisualIndexes: []int32{0, 1, 2}, // 3 deepstack layers
```

**After**:
```go
deepstackVisualIndexes: []int32{8, 16, 24}, // Deepstack extraction layers matching GGUF tensor naming
```

## Why This Change
The `deepstackVisualIndexes` array specifies which vision encoder layers should have deepstack features extracted. In VisionModel.Forward:

```go
for i, layer := range m.Layers {
    hiddenStates = layer.Forward(...)
    if i := slices.Index(m.deepstackVisualIndexes, int32(i)); i >= 0 && m.DeepstackMerger[i] != nil {
        deepstackStates[i] = m.DeepstackMerger[i].Forward(...)
    }
}
```

This code checks if current layer index `i` is in `deepstackVisualIndexes`. With old values [0,1,2], it would extract at layers 0,1,2. With correct values [8,16,24], it extracts at the right layers matching llama.cpp implementation.

## Verification
Matches llama.cpp qwen3vl.cpp implementation where deepstack features are extracted from vision encoder layers 8, 16, and 24.

## Impact
Ensures deepstack projection is called at correct layers during vision encoding.

## Next Step
→ See `03_manual_fc_loading.md`
