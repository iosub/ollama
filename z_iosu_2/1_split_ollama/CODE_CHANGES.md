# Code Changes Summary

## Files Modified (4 total)

### 1. model/models/qwen3vl/model.go
**Lines changed**: 114-122, 158-177, 260-289

**Changes**:
- Pre-initialize DeepstackMerger array with 3 VisionPatchMerger instances
- Concatenate main vision + 3 deepstack embeddings in EncodeMultimodal  
- Split concatenated embeddings in Forward and add to LLM hidden states

### 2. model/models/qwen3vl/model_vision.go  
**Lines changed**: 124-128, 142-150, 371-383, 397-401

**Changes**:
- Updated VisionPatchMerger gguf struct tags (fc1/fc2)
- Added warning log when FC layers are nil
- Pre-allocate DeepstackMerger array in newVisionModel
- Hardcode deepstackVisualIndexes = [0,1,2]

### 3. ml/backend/ggml/ggml/src/ggml.c
**Line changed**: 1669

**Change**:
- Re-enabled assertion (was commented to bypass errors)

### 4. Nothing else
**Status**: Only these 3 files modified

## Revert Instructions

To undo all changes (if needed to start fresh):

```powershell
cd c:\IA\tools\ollama

# Revert Go files
git checkout model/models/qwen3vl/model.go
git checkout model/models/qwen3vl/model_vision.go

# Revert C file  
git checkout ml/backend/ggml/ggml/src/ggml.c

# Rebuild
go build .
```

## Key Code to Remember

**Concatenation** (model.go:164):
```go
concatenated := allEmbeds[0].Concat(ctx, allEmbeds[1], 1)
for i := 2; i < len(allEmbeds); i++ {
    concatenated = concatenated.Concat(ctx, allEmbeds[i], 1)
}
```

**Splitting** (model.go:268-274):
```go
mainVision := visionOutputs.View(ctx, nEmbd, -1)
deepstackVisualEmbeds := make([]ml.Tensor, nDeepstackLayers)
for i := 0; i < nDeepstackLayers; i++ {
    offset := (i + 1) * nEmbd
    deepstackVisualEmbeds[i] = visionOutputs.View(ctx, nEmbd, -1, offset, ...)
}
```

**Pre-init array** (model.go:117):
```go
m.VisionModel.DeepstackMerger = make([]*VisionPatchMerger, 3)
for i := range m.VisionModel.DeepstackMerger {
    m.VisionModel.DeepstackMerger[i] = &VisionPatchMerger{}
}
```
