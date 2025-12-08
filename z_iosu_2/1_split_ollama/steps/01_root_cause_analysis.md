# Step 1: Root Cause Analysis

## Problem Identified
DeepstackMerger FC weights were not loading despite existing in GGUF file.

## Investigation Process

### 1. Analyzed vision_bridge.go
**File**: `model/vision_bridge.go` lines 93-105

**Finding**: vision_bridge DOES support arrays, but uses `strconv.Itoa(j)` where j is array index.

```go
for j := 0; j < vv.Len(); j++ {
    elem := vv.Index(j)
    layerTags := append(tagsCopy, Tag{name: strconv.Itoa(j)})
    // Recursively populates array element
}
```

### 2. Tensor Name Mismatch Discovered

**vision_bridge builds**:
- `v.deepstack.0.fc1` (array index 0)
- `v.deepstack.1.fc1` (array index 1)
- `v.deepstack.2.fc1` (array index 2)

**GGUF contains**:
- `v.deepstack.8.fc1.weight` (vision encoder layer 8)
- `v.deepstack.16.fc1.weight` (vision encoder layer 16)
- `v.deepstack.24.fc1.weight` (vision encoder layer 24)

**Conclusion**: Array indices (0,1,2) ≠ Layer IDs (8,16,24)

## Root Cause
vision_bridge cannot auto-populate DeepstackMerger FC weights because tensor naming convention doesn't match array indexing scheme.

## Evidence
From go09.log (before fix):
```
WARN DeepstackMerger FC layers are nil dim=4608 (×4)
main_shape=[4608 X] (should be [4096 X])
ggml.c:1669 ASSERT failed
```

Tensors exist in backend:
```
TRACE created tensor name=v.deepstack.8.fc1.weight shape=[4608 4608]
TRACE created tensor name=v.deepstack.8.fc2.weight shape=[4608 4096]
...
```

## Solution Approach
Manual tensor loading using `Backend.Get()` with correct tensor names.

## Next Step
→ See `02_deepstack_visual_indexes.md`
