# Step 3: Manual FC Weight Loading

## What Was Done
Implemented manual loading of DeepstackMerger FC1/FC2 weights using `Backend.Get()`.

## Code Changes

### Added Imports
**File**: `model/models/qwen3vl/model.go` lines 3-16

```go
import (
    "fmt"              // Added for tensor name formatting
    "github.com/ollama/ollama/ml/nn"  // Added for Linear construction
    // ... existing imports
)
```

### Implementation Location
**File**: `model/models/qwen3vl/model.go` lines 61-100

**Placement**: Inside `ensureVisionReady()` → `if m.HasProjector()` block

**Why this location**: 
- This block executes for split models that already have vision tensors loaded
- Runs AFTER `InferOptionsFromTensors()` which sets up vision dimensions
- Ensures DeepstackMerger array is pre-allocated

### Code Implementation
```go
// MANUAL LOADING: DeepstackMerger FC weights
// vision_bridge cannot populate FC weights because array indices (0,1,2) don't match GGUF layer IDs (8,16,24)
// vision_bridge uses strconv.Itoa(j) which builds "v.deepstack.0.fc1" but GGUF has "v.deepstack.8.fc1.weight"
if m.VisionModel.DeepstackMerger != nil && len(m.VisionModel.DeepstackMerger) >= 3 {
    layerIDs := []int{8, 16, 24}
    for idx, layerID := range layerIDs {
        prefix := fmt.Sprintf("v.deepstack.%d", layerID)

        // Try to find FC1 weight tensor
        fc1WeightName := prefix + ".fc1.weight"
        if fc1Weight := m.Backend().Get(fc1WeightName); fc1Weight != nil {
            fc1BiasName := prefix + ".fc1.bias"
            fc1Bias := m.Backend().Get(fc1BiasName) // May be nil
            m.VisionModel.DeepstackMerger[idx].FC1 = &nn.Linear{
                Weight: fc1Weight,
                Bias:   fc1Bias,
            }
            slog.Info("Manually loaded DeepstackMerger FC1", "layer", layerID, "idx", idx, "shape", fc1Weight.Shape())
        }

        // Try to find FC2 weight tensor
        fc2WeightName := prefix + ".fc2.weight"
        if fc2Weight := m.Backend().Get(fc2WeightName); fc2Weight != nil {
            fc2BiasName := prefix + ".fc2.bias"
            fc2Bias := m.Backend().Get(fc2BiasName) // May be nil
            m.VisionModel.DeepstackMerger[idx].FC2 = &nn.Linear{
                Weight: fc2Weight,
                Bias:   fc2Bias,
            }
            slog.Info("Manually loaded DeepstackMerger FC2", "layer", layerID, "idx", idx, "shape", fc2Weight.Shape())
        }
    }
}
```

## How It Works
1. Iterates through layer IDs: 8, 16, 24
2. For each layer, constructs exact tensor names (e.g., "v.deepstack.8.fc1.weight")
3. Uses `Backend.Get()` to retrieve tensors directly from backend
4. Manually constructs `nn.Linear` instances with Weight and Bias tensors
5. Assigns to `DeepstackMerger[idx].FC1` and `.FC2`

## Test Results (go11.log)
```
INFO Manually loaded DeepstackMerger FC1 layer=8 idx=0 shape=[4608 4608]
INFO Manually loaded DeepstackMerger FC2 layer=8 idx=0 shape=[4608 4096]
INFO Manually loaded DeepstackMerger FC1 layer=16 idx=1 shape=[4608 4608]
INFO Manually loaded DeepstackMerger FC2 layer=16 idx=1 shape=[4608 4096]
INFO Manually loaded DeepstackMerger FC1 layer=24 idx=2 shape=[4608 4608]
INFO Manually loaded DeepstackMerger FC2 layer=24 idx=2 shape=[4608 4096]
```

✅ All 6 weights loaded successfully (3 layers × 2 FC layers each)

## Next Step
→ See `04_patchmerger_fix.md`
