# QUICK START - Next Session

## 🎯 THE PROBLEM
FC weights exist in GGUF but vision_bridge doesn't load them → embeddings stay 4608-dim → assertion fails at ggml.c:1669

## ⚡ WHAT TO DO

**Investigate**: `model/vision_bridge.go`  
**Question**: Does `populateVisionFields()` support array elements?

**If NO**: Manually load FC weights in `model.go` line ~117 after `RepopulateField`

## 📝 WHAT WAS CHANGED

**3 files modified** (see CODE_CHANGES.md for details):
1. `model/models/qwen3vl/model.go` - Lines 114-122, 158-177, 260-289
2. `model/models/qwen3vl/model_vision.go` - Lines 124-128, 142-150, 371-383, 397-401  
3. `ml/backend/ggml/ggml/src/ggml.c` - Line 1669

## 🧪 WHAT WAS TRIED (all failed)

5 attempts to load FC weights via struct tags - none worked.

**Current state go09.log**:
```
❌ WARN DeepstackMerger FC layers are nil (×4)
❌ main_shape=[4608 X] (should be [4096 X])
❌ ggml.c:1669 ASSERT failed
```

## ✅ WHAT TO DO NEXT

1. Open `vision_bridge.go`
2. Check if it handles array elements
3. If not → manual loading (Backend.FindTensor)
4. Verify FC1/FC2 not nil
5. Test → should see "DeepstackMerger projecting"

**Time estimate**: 30 min

---

**Full details**: See other docs in this folder
