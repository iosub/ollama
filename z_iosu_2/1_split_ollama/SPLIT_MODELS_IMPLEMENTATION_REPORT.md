# Qwen3-VL Split Models Implementation Report

**Date:** December 8, 2025  
**Branch:** `OllamaSplitRunner`  
**Goal:** Run Qwen3-VL split (and unified) models in Ollama using the new GGML engine

---

## 📋 Executive Summary

This document describes the modifications made to enable **Qwen3-VL multimodal models** (both split and unified) to work in Ollama. Split models come in 2 separate GGUF files (text + vision), while unified models come in a single file.

### Target Models
| Model | Type | Status |
|--------|------|--------|
| Qwen3-VL 4B Split (unsloth) | Split GGUF | ✅ Working |
| Qwen3-VL 8B Split (unsloth) | Split GGUF | ✅ Working |
| Qwen3-VL 4B Unified (ollama) | Unified GGUF | ✅ Working |
| Qwen3-VL 8B Unified (ollama) | Unified GGUF | ✅ Working |

---

## 🐛 Identified Problems and Solutions

### 1. KV Cache Corruption (CRITICAL)

**Problem:** When switching between models without restarting the server, the KV cache contained stale data that caused crashes or incorrect results.

**File:** `runner/ollamarunner/cache.go`

**Solution:** Clear the KV cache when the prompt contains multimodal embeddings (images).

```go
// Check if prompt contains any multimodal embeddings (images)
hasMultimodal := false
for _, inp := range prompt {
    if inp.Multimodal != nil || inp.MultimodalHash != 0 {
        hasMultimodal = true
        break
    }
}

// Clear cache for multimodal prompts to avoid stale embedding data
if hasMultimodal && numPast > 0 {
    slog.Debug("clearing KV cache for prompt with multimodal embeddings", "id", slot.Id, "numPast", numPast)
    if c.cache != nil {
        c.cache.Remove(slot.Id, 0, math.MaxInt32)
    }
    numPast = 0
}
```

**Note:** This fix already existed in `runner/llamarunner/cache.go` (old runner) but was missing from `runner/ollamarunner/cache.go` (new runner used by Qwen3-VL).

---

### 2. Vulkan Device Detection Bug

**Problem:** The new GGML engine detected Vulkan devices even when `OLLAMA_VULKAN=false`, causing incorrect GPU selection.

**File:** `ml/backend/ggml/ggml.go`

**Solution:** Filter Vulkan devices based on the environment variable.

```go
case C.GGML_BACKEND_DEVICE_TYPE_GPU, C.GGML_BACKEND_DEVICE_TYPE_IGPU:
    // Skip Vulkan devices if OLLAMA_VULKAN is not enabled
    name := C.GoString(C.ggml_backend_dev_name(d))
    if !envconfig.EnableVulkan() && strings.Contains(strings.ToLower(name), "vulkan") {
        slog.Info("skipping Vulkan device (OLLAMA_VULKAN not enabled)", "device", name)
        continue
    }
    gpus = append(gpus, d)
```

---

### 3. GGUF Tensor Name Mismatch

**Problem:** Unified models use `v.pos_embed` while split models use `v.position_embd` for the same tensor.

**File:** `model/models/qwen3vl/model_vision.go`

**Solution:** Use GGUF aliases to support both names.

```go
type VisionPositionEmbedding struct {
    PositionEmbedding *nn.Embedding `gguf:"pos_embed,alt:position_embd"` // Unified: pos_embed, Split: position_embd
}
```

---

### 4. Deepstack Layer ID Detection

**Problem:** DeepstackMerger indices in code (0,1,2) don't match the layer IDs in GGUF:
- 8B model: layers 8, 16, 24
- 4B model: layers 5, 11, 17

**File:** `model/models/qwen3vl/model.go`

**Solution:** Auto-detect layer IDs from GGUF tensors.

```go
func detectDeepstackLayerIDs(backend ml.Backend) []int {
    var layerIDs []int
    candidateLayers := []int{4, 5, 6, 8, 9, 10, 11, 12, 14, 16, 17, 18, 20, 24, 28, 32}

    for _, layerID := range candidateLayers {
        tensorName := fmt.Sprintf("v.deepstack.%d.fc1.weight", layerID)
        if tensor := backend.Get(tensorName); tensor != nil {
            layerIDs = append(layerIDs, layerID)
        }
    }
    return layerIDs
}
```

---

### 5. Position Embedding Interpolation

**Problem:** The code was using nearest-neighbor interpolation which produced artifacts. Unified models require bilinear interpolation.

**File:** `model/models/qwen3vl/model_vision.go`

**Solution:** Restore bilinear interpolation from the original upstream code.

```go
// UNIFIED MODEL: Use bilinear interpolation (original upstream code)
for h := range grid.Height {
    for w := range grid.Width {
        y, x := float32(h)*stepHeight, float32(w)*stepWidth
        floorY, floorX := int32(y), int32(x)
        ceilY, ceilX := min(floorY+1, int32(opts.gridPerSide-1)), min(floorX+1, int32(opts.gridPerSide-1))

        // 4 corner indices for bilinear interpolation
        indexSlice[0][i] = floorY*int32(opts.gridPerSide) + floorX
        indexSlice[1][i] = floorY*int32(opts.gridPerSide) + ceilX
        indexSlice[2][i] = ceilY*int32(opts.gridPerSide) + floorX
        indexSlice[3][i] = ceilY*int32(opts.gridPerSide) + ceilX

        // Bilinear weights
        weightSlice[0][i] = (1 - (y - float32(floorY))) * (1 - (x - float32(floorX)))
        // ... etc
    }
}
```

---

### 6. Tensor Concatenation Dimension

**Problem:** GGML uses column-major order. Embedding concatenation was using `dim=1` but should be `dim=0`.

**File:** `model/models/qwen3vl/model.go`

**Solution:** Fix the concatenation dimension.

```go
// Before (WRONG):
concatenated := allEmbeds[0].Concat(ctx, allEmbeds[1], 1)

// After (CORRECT - column-major GGML):
concatenated := allEmbeds[0].Concat(ctx, allEmbeds[1], 0)
```

---

### 7. Multimodal Projector (mm.0/mm.2) Loading

**Problem:** Split models have `mm.0.*` and `mm.2.*` projectors that are DIFFERENT from DeepstackMerger FC. The original code didn't load them.

**Files:** `model/models/qwen3vl/model.go`, `model/models/qwen3vl/model_vision.go`

**Solution:** Add fields for multimodal projectors and use them in vision processing.

```go
// In Model struct
MultimodalProjectorFC1 *nn.Linear `gguf:"mm.0"` // [4608, 4608] with GELU
MultimodalProjectorFC2 *nn.Linear `gguf:"mm.2"` // [4608, 4096]

// In VisionModel
MultimodalFC1 *nn.Linear
MultimodalFC2 *nn.Linear
```

---

### 8. PostNorm for Split Models

**Problem:** Split models have `v.post_ln` that must be applied before the final projection.

**File:** `model/models/qwen3vl/model_vision.go`

**Solution:** Add and apply PostNorm.

```go
type VisionModel struct {
    // ...
    PostNorm *nn.LayerNorm `gguf:"post_ln"` // Present in split models (1152 dim)
}

// Apply before projection
if m.PostNorm != nil {
    hiddenStates = m.PostNorm.Forward(ctx, hiddenStates, m.VisionOptions.eps)
}
```

---

### 9. Repetition Loop Detection (Enhancement)

**Problem:** The model sometimes entered repetition loops generating garbage text.

**File:** `runner/llamarunner/runner.go`

**Solution:** Make detection more sensitive.

```go
// Buffer increased: 256 -> 512 tokens
recentTokens: make([]int, 512)

// Pattern range expanded: 8-64 -> 6-128 tokens
for patternLen := 6; patternLen <= 128; patternLen++ {

// More sensitive trigger: 3 -> 2 detections
if seq.repetitionCount >= 2 {
```

---

## 📁 Modified Files

| File | Changes |
|---------|---------|
| `runner/ollamarunner/cache.go` | KV cache clearing for multimodal |
| `ml/backend/ggml/ggml.go` | Vulkan device filtering |
| `model/models/qwen3vl/model_vision.go` | GGUF tag, bilinear interpolation, PostNorm, MM projectors, deepstack layer calc |
| `model/models/qwen3vl/model.go` | Deepstack auto-detection, MM projectors, concatenation fix |
| `runner/llamarunner/runner.go` | More sensitive repetition detection |

---

## 🔧 Environment Configuration

```bash
# Disable Vulkan (use CUDA)
set OLLAMA_VULKAN=false

# Enable debug logs
set OLLAMA_DEBUG=1

# Start server
go run . serve

# In another terminal, run model
.\ollama.exe run qwen3-vl:8b-instruct-q8_0 "describe this image" --images image.jpg
```

---

## 📊 Execution Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    Ollama New Engine                          │
│                  (ml/backend/ggml/)                           │
│                                                               │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐       │
│  │   Model     │───▶│   Vision    │───▶│   Runner    │       │
│  │   model.go  │    │ model_vis.. │    │  cache.go   │       │
│  └─────────────┘    └─────────────┘    └─────────────┘       │
│                                                               │
│  Qwen3-VL uses: runner/ollamarunner/ (NOT llamarunner)       │
└──────────────────────────────────────────────────────────────┘
```

---

## 🎯 Next Steps (Optional)

1. **Upstream PR**: Consider sending critical fixes (cache, Vulkan) to the main repository
2. **Tests**: Add unit tests for the changes
3. **Other models**: Verify that other multimodal models (LLaVA, etc.) are not affected

---

## 📝 Reference Logs

Test session logs are located at:
- `z_iosu_2/logs/` - Execution logs
- `z_iosu_2/logs3/` - Previous session logs

---

## ✅ Verification

To verify everything works:

```bash
# 1. Build
cmake -B build
cmake --build build --config Release
go build .

# 2. Start server
set OLLAMA_VULKAN=false
set OLLAMA_DEBUG=1
.\ollama.exe serve

# 3. Test with image (another terminal)
.\ollama.exe run qwen3-vl:4b "What do you see?" --images test.jpg
```

If the response correctly describes the image, it works! 🎉
