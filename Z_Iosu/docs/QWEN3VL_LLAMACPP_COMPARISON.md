# Qwen3-VL: llama.cpp vs Ollama Implementation Comparison

## Critical Differences Found

### 1. **Patch Embedding: Conv2D vs Conv3D**

**llama.cpp (clip.cpp:917-925):**
```cpp
// Two separate Conv2D operations
ggml_tensor * inp = ggml_conv_2d(ctx0, model.patch_embeddings_0, inp_raw, 
                                  patch_size, patch_size, 0, 0, 1, 1);
                                  // stride: 14,14  pad: 0,0  dilation: 1,1

auto inp_1 = ggml_conv_2d(ctx0, model.patch_embeddings_1, inp_raw, 
                           patch_size, patch_size, 0, 0, 1, 1);
                           // stride: 14,14  pad: 0,0  dilation: 1,1

inp = ggml_add(ctx0, inp, inp_1);  // Element-wise ADD
```

**Ollama (current):**
```go
// Single Conv3D with dual weights
t1 := m.Weight.Conv3D(ctx, t, c, s0, s1, 1, p0, p1, p2, d0, d1, d2)
      // s0,s1 = patchSize (14,14)  s2 = 1 (temporal)
      
t2 := m.Weight1.Conv3D(ctx, t, c, s0, s1, 1, p0, p1, p2, d0, d1, d2)
      // s0,s1 = patchSize (14,14)  s2 = 1 (temporal)

t = t1.Add(ctx, t2)  // Element-wise ADD
```

**Status:** ✅ Similar approach (ADD of two convolutions)
**Issue:** Conv3D behavior vs Conv2D - temporal dimension handling

---

### 2. **SPATIAL MERGE RESHAPE (MISSING IN OLLAMA)**

**llama.cpp (clip.cpp:927-937):**
```cpp
// CRITICAL: Spatial merge / pixel shuffle after ADD
inp = ggml_permute(ctx0, inp, 1, 2, 0, 3);  // [w, h, c, b] -> [c, w, h, b]

inp = ggml_cont_4d(ctx0, inp,
    n_embd * 2, n_patches_x / 2, n_patches_y, batch_size);
    
inp = ggml_reshape_4d(ctx0, inp,
    n_embd * 2, n_patches_x / 2, 2, batch_size * (n_patches_y / 2));
    
inp = ggml_permute(ctx0, inp, 0, 2, 1, 3);

inp = ggml_cont_3d(ctx0, inp,
    n_embd, n_patches_x * n_patches_y, batch_size);
```

**Purpose:**
- Converts `[?, n_patches_x, n_patches_y, ?]` → `[n_embd, n_patches_x * n_patches_y, batch]`
- Reduces spatial dimensions by factor 2
- Merges patches spatially
- Final shape: `[1152, n_patches_x * n_patches_y, 1]`

**Ollama (current):**
```go
// NO SPATIAL MERGE RESHAPE
// Expect Conv3D to produce output directly
hiddenStates := m.PatchEmbedding.Forward(...)
// hiddenStates should be [1152, spatial] but may be different
```

**Status:** ❌ **MISSING** - This is likely why we get wrong channel counts
**Impact:** Conv3D may produce different spatial/channel layout

---

### 3. **Patch Bias Application**

**llama.cpp (clip.cpp:940-943):**
```cpp
// AFTER spatial merge, ADD patch bias
if (model.patch_bias != nullptr) {
    inp = ggml_add(ctx0, inp, model.patch_bias);
    cb(inp, "patch_bias", -1);
}
```

**Ollama (current):**
```go
// Bias applied INSIDE Conv3D.Forward
// Standard path after ADD of t1 and t2
if bias != nil {
    t = t.Add(ctx, bias.Reshape(...))
}
```

**Status:** ⚠️ Different timing - llama.cpp applies AFTER spatial merge
**Impact:** May affect final dimensions

---

### 4. **Position Embedding**

**llama.cpp (clip.cpp:946-959):**
```cpp
// Resize position embeddings dynamically
ggml_tensor * learned_pos_embd = resize_position_embeddings();

// Apply same spatial merge to position embeddings
learned_pos_embd = ggml_cont_4d(..., n_embd * 2, n_patches_x / 2, ...);
learned_pos_embd = ggml_reshape_4d(..., n_embd * 2, n_patches_x / 2, 2, ...);
learned_pos_embd = ggml_permute(...);
learned_pos_embd = ggml_cont_3d(..., n_embd, n_patches_x * n_patches_y, ...);

// ADD to input
inp = ggml_add(ctx0, inp, learned_pos_embd);
```

**Ollama (current):**
```go
// Skip position embedding for split GGUF
if actualHiddenSize != opts.hiddenSize {
    logutil.Trace("SPLIT GGUF Vision: skipping position embedding")
    // No position embedding applied
}
```

**Status:** ❌ **SKIPPED** - We don't resize/apply position embeddings
**Impact:** May affect model accuracy but shouldn't crash

---

## Key Findings

### **ROOT CAUSE: Spatial Merge Reshape**

The SPATIAL MERGE RESHAPE (lines 927-937 in llama.cpp) is **CRITICAL** and **MISSING** in our implementation.

**What it does:**
1. Takes output from ADD of two Conv2D operations
2. Performs complex permute/reshape operations
3. Reduces spatial dimensions by factor 2
4. Produces final shape: `[n_embd, n_patches_x * n_patches_y, batch]`

**Why we need it:**
- Conv3D with temporal dimension may not produce the same output layout as Conv2D
- Without spatial merge, channel/spatial dimensions may be wrong
- This explains why we see 384 or 768 channels instead of 1152

### **Expected Flow (llama.cpp):**

```
inp_raw: [3, H, W, batch]
  ↓
Conv2D (weight_0): [?, patches_x, patches_y, ?]
Conv2D (weight_1): [?, patches_x, patches_y, ?]
  ↓
ADD: [?, patches_x, patches_y, ?]
  ↓
SPATIAL MERGE RESHAPE: [1152, patches_x * patches_y, 1]  ← CRITICAL
  ↓
Add patch_bias: [1152, patches_x * patches_y, 1]
  ↓
Add position_embd: [1152, patches_x * patches_y, 1]
  ↓
Vision Layers (QKV [1152, 3456]): ✅ Dimensions match
```

### **Current Flow (Ollama):**

```
pixelValues: [patchSize, patchSize, temporalPatchSize, -1]
  ↓
Conv3D (weight_0, stride_temporal=1): [?, spatial]
Conv3D (weight_1, stride_temporal=1): [?, spatial]
  ↓
ADD: [?, spatial]  ← Wrong dimensions?
  ↓
NO SPATIAL MERGE  ← MISSING
  ↓
Skip position_embd
  ↓
Vision Layers (QKV [1152, 3456]): ❌ Dimension mismatch if ADD != [1152]
```

---

## Recommendations

### **Option 1: Implement Spatial Merge Reshape (Complex)**

Add the missing spatial merge reshape operations after Conv3D ADD:
- Requires understanding exact Conv3D output layout
- Need to replicate permute/reshape sequence
- High complexity, more code

### **Option 2: Investigate Conv3D Parameters (Current Approach)**

- Verify Conv3D with stride_temporal=1 produces correct output
- Check if Conv3D implicitly does spatial merge
- Add logging to see actual output shapes
- Simpler if Conv3D already handles it

### **Option 3: Switch to Conv2D (Major Refactor)**

- Replace Conv3D with two Conv2D operations
- Implement spatial merge reshape exactly as llama.cpp
- Guaranteed compatibility
- Requires significant code changes

---

## Action Items

1. ✅ **Add comprehensive logging** (DONE)
   - Log Conv3D input/output shapes
   - Log channel counts at each step
   - Detect dimension mismatches early

2. 🔄 **Test current implementation**
   - Compile with stride_temporal=1 fix
   - Check if Conv3D produces [1152, spatial]
   - Verify QKV matmul dimensions

3. ⏳ **If Conv3D produces wrong dimensions:**
   - Implement spatial merge reshape
   - Or investigate Conv3D parameters further
   - Or switch to Conv2D approach

4. ⏳ **Position embedding:**
   - Low priority (model can work without it)
   - Implement resize_position_embeddings if needed
   - Apply after spatial merge

---

## Conclusion

The **SPATIAL MERGE RESHAPE** is the most critical missing piece. Our logs will show if:
- Conv3D with stride_temporal=1 produces [1152, spatial] directly ✅
- Or if we need to implement the spatial merge reshape manually ❌

Once we see the actual Conv3D output shapes, we'll know which path to take.
