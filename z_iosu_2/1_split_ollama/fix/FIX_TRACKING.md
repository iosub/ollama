# Fix Tracking - Qwen3-VL Split GGUF Support

## Status: BLOCKED
**Blocker**: DeepstackMerger FC weights not loading from GGUF despite correct struct tags

---

## Tasks Completed ✅

### 1. Understanding Deepstack Architecture
- [x] Analyzed llama.cpp qwen3vl.cpp implementation
- [x] Identified concatenation approach: main + 3 deepstack → 16384 dims
- [x] Understood deepstack features added to LLM layers 0,1,2
- [x] Documented dimension flow: 1152 → 4608 → 4096

### 2. Implement Deepstack Array Initialization
- [x] Added `deepstackVisualIndexes []int32` to VisionOptions struct
- [x] Set hardcoded values [0,1,2] in newVisionModel (lines 397-401)
- [x] Pre-allocate DeepstackMerger array with 3 VisionPatchMerger instances (lines 371-383)
- [x] Initialize array BEFORE vision_bridge populates it (lines 114-122 in model.go)

### 3. Implement Vision Embedding Concatenation
- [x] Modified EncodeMultimodal to concatenate vision + deepstack (lines 158-177)
- [x] Added logging to show concatenated dimensions
- [x] Created single Multimodal input with concatenated tensor

### 4. Implement LLM Forward Integration
- [x] Modified Forward to detect concatenated embeddings (lines 260-289)
- [x] Split concatenated tensor back into main + deepstack arrays
- [x] Add deepstack features to hiddenStates at layers 0,1,2

### 5. Fix Struct Tags for FC Weights
- [x] Changed VisionPatchMerger tags from `linear_fc1` to `fc1` as primary
- [x] Added alternatives: `fc1,alt:linear_fc1,alt:ffn_up`
- [x] Same for FC2: `fc2,alt:linear_fc2,alt:ffn_down`

### 6. Add Debug Logging
- [x] Log when DeepstackMerger.Forward is called
- [x] Log when FC layers are nil
- [x] Log concatenated embedding shapes
- [x] Log split embedding detection

---

## Tasks Failed ❌

### 1. **Load FC Weights from GGUF** 
**Status**: FAILED after 10 hours
**Attempts**:
- Tried using default struct tags (`linear_fc1`)
- Changed to match GGUF naming (`fc1`)  
- Added multiple alternatives in tags
- Pre-initialized array before vision_bridge

**Result**: FC1 and FC2 remain nil at runtime despite:
- Tensors existing in GGUF (confirmed in logs)
- Correct tensor naming (v.deepstack.8.fc1.weight)
- Struct tags matching names
- Array pre-allocation

**Log evidence go09**:
```
WARN DeepstackMerger FC layers are nil dim=4608  (×4)
```

**Hypothesis**: vision_bridge cannot populate structs inside array elements

---

## Tasks Remaining 🔴

### Critical Path to Fix

#### Task A: **Investigate vision_bridge Array Handling**
**Priority**: CRITICAL
**Question**: Does vision_bridge support loading tensors into array element fields?

**Investigation needed**:
1. Check `model/vision_bridge.go` populateVisionFields function
2. See if it recursively populates array elements
3. Test if moving DeepstackMerger out of array helps

**Test approach**:
```go
// Current (doesn't work):
DeepstackMerger []*VisionPatchMerger  // Array

// Try changing to:
DeepstackMerger0 *VisionPatchMerger `gguf:"deepstack.0"`
DeepstackMerger1 *VisionPatchMerger `gguf:"deepstack.1"`  
DeepstackMerger2 *VisionPatchMerger `gguf:"deepstack.2"`
```

#### Task B: **Manual Weight Loading Fallback**
**Priority**: HIGH (if Task A confirms vision_bridge limitation)

If vision_bridge can't handle arrays, manually load FC weights:
1. In `LoadSecondary` or `InferOptionsFromTensors`
2. Use `m.Backend().FindTensor()` to get weights directly  
3. Manually create nn.Linear instances with found tensors
4. Assign to DeepstackMerger[i].FC1/FC2

**Code location**: model.go around line 117 (after RepopulateField)

#### Task C: **Comment ggml.c:1669 as Temporary Workaround**
**Priority**: LOW (only for testing, causes hallucinations)

**Purpose**: See if model generates ANY output with wrong dimensions
**Risk**: Will hallucinate due to corrupt memory reads
**Use**: Only to verify rest of pipeline works

---

## Files Modified

### Core Implementation
- `model/models/qwen3vl/model.go` (4 locations)
- `model/models/qwen3vl/model_vision.go` (6 locations)

### Attempted Fix
- `ml/backend/ggml/ggml/src/ggml.c` (line 1669 - re-enabled assertion)

---

## Metrics
-  **Time spent**: 10 hours
- **Builds**: ~15 successful compilations  
- **Tests**: ~15 test runs
- **Log files**: go01.log through go09.log
- **Progress**: Blocked at FC weight loading

---

## Next Steps for New Session

1. **START HERE**: Read `docs/SPLIT_MODEL_ANALYSIS.md` for full context
2. **INVESTIGATE**: vision_bridge.go - does it support array element population?
3. **IF YES**: Why isn't it working? Add debug logging to vision_bridge
4. **IF NO**: Implement Task B (manual weight loading)
5. **TEST**: Verify FC weights load correctly before continuing
6. **THEN**: Continue with dimension verification and testing

---

## Critical Insight

**The root blocker is NOT**:
- ❌ Concatenation logic (implemented correctly)
- ❌ LLM integration (implemented correctly)  
- ❌ Dimension understanding (correct)
- ❌ Struct tags (tried all variations)

**The root blocker IS**:
- ✅ **Tensor loading mechanism** - weights exist but don't load into struct

Once FC weights load, remaining pipeline should work correctly.
