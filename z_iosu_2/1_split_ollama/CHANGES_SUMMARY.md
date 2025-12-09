# Changes Summary - Qwen3-VL Split Models

## Files Modified

### 1. runner/ollamarunner/cache.go
**Purpose:** Fix KV cache corruption when switching between models

```diff
+// Check if prompt contains any multimodal embeddings (images)
+hasMultimodal := false
+for _, inp := range prompt {
+    if inp.Multimodal != nil || inp.MultimodalHash != 0 {
+        hasMultimodal = true
+        break
+    }
+}
+
+// Clear KV cache for multimodal to avoid stale embeddings
+if hasMultimodal && numPast > 0 {
+    slog.Debug("clearing KV cache for prompt with multimodal embeddings")
+    if c.cache != nil {
+        c.cache.Remove(slot.Id, 0, math.MaxInt32)
+    }
+    numPast = 0
+}
```

### 2. ml/backend/ggml/ggml.go
**Purpose:** Fix Vulkan device detection ignoring OLLAMA_VULKAN env var

```diff
 case C.GGML_BACKEND_DEVICE_TYPE_GPU, C.GGML_BACKEND_DEVICE_TYPE_IGPU:
+    // Skip Vulkan devices if OLLAMA_VULKAN is not enabled
+    name := C.GoString(C.ggml_backend_dev_name(d))
+    if !envconfig.EnableVulkan() && strings.Contains(strings.ToLower(name), "vulkan") {
+        slog.Info("skipping Vulkan device (OLLAMA_VULKAN not enabled)", "device", name)
+        continue
+    }
     gpus = append(gpus, d)
```

### 3. model/models/qwen3vl/model_vision.go
**Purpose:** Support both unified and split GGUF tensor names, fix position embedding

```diff
 type VisionPositionEmbedding struct {
-    PositionEmbedding *nn.Embedding `gguf:"position_embed,alt:position_embd"`
+    PositionEmbedding *nn.Embedding `gguf:"pos_embed,alt:position_embd"` // Unified: pos_embed, Split: position_embd
 }

+type VisionModel struct {
+    // ...
+    PostNorm *nn.LayerNorm `gguf:"post_ln"` // Present in split models
+    MultimodalFC1 *nn.Linear // FC1: 4608 -> 4608 with GELU (from mm.0.*)
+    MultimodalFC2 *nn.Linear // FC2: 4608 -> 4096 (from mm.2.*)
+    deepstackLayerIDs []int
+}

+// Position embedding: restored bilinear interpolation (upstream code)
+// calculateDeepstackLayerIDs: auto-calculate based on model size
```

### 4. model/models/qwen3vl/model.go
**Purpose:** Auto-detect deepstack layers, load mm.0/mm.2 projectors, fix concatenation

```diff
+// detectDeepstackLayerIDs probes GGUF for v.deepstack.N.fc1.weight tensors
+func detectDeepstackLayerIDs(backend ml.Backend) []int

+// loadDeepstackMergerWeights loads FC weights with correct layer ID mapping
+func (m *Model) loadDeepstackMergerWeights(layerIDs []int)

+// Model struct additions
+MultimodalProjectorFC1 *nn.Linear `gguf:"mm.0"`
+MultimodalProjectorFC2 *nn.Linear `gguf:"mm.2"`

 // Fix concatenation dimension (GGML column-major)
-concatenated := allEmbeds[0].Concat(ctx, allEmbeds[1], 1)
+concatenated := allEmbeds[0].Concat(ctx, allEmbeds[1], 0)
```

### 5. runner/llamarunner/runner.go
**Purpose:** More sensitive repetition loop detection

```diff
-recentTokens: make([]int, 256)
+recentTokens: make([]int, 512)

-for patternLen := 8; patternLen <= 64; patternLen++ {
+for patternLen := 6; patternLen <= 128; patternLen++ {

-if seq.repetitionCount >= 3 {
+if seq.repetitionCount >= 2 {
```

---

## Key Insights

1. **Two Runners**: Ollama has two runners:
   - `runner/llamarunner/` - Old runner (llama.cpp based)
   - `runner/ollamarunner/` - New runner (GGML based) ← Qwen3-VL uses this

2. **Cache Fix Already Existed**: The multimodal cache fix existed in `llamarunner` but was missing from `ollamarunner`

3. **GGUF Tensor Naming**:
   - Unified models (ollama): `v.pos_embed`, `v.deepstack_merger.0.*`
   - Split models (unsloth): `v.position_embd`, `v.deepstack.8.*`, `v.deepstack.16.*`, `v.deepstack.24.*`

4. **Deepstack Layer IDs Vary by Model Size**:
   - 8B: layers 8, 16, 24 (27 vision encoder layers)
   - 4B: layers 5, 11, 17 (24 vision encoder layers)
