# Split Model Investigation - Qwen3-VL Issue Tracking

## Problem Summary

**Issue**: Qwen3-VL split models (2-file GGUF: text + vision) fail to load with error:
```
ggml-backend.cpp:832: pre-allocated tensor (cache_k_l0 (view)) in a buffer (CUDA0) 
that cannot run the operation (SET_ROWS)
```

## Key Findings

### 1. New Ollama Engine Does NOT Work with Split Models
- ✅ **CONFIRMED**: New Ollama engine (`OLLAMA_NEW_ENGINE=true`) does not implement split model support
- ❌ When `qwen3vl` is **uncommented** in `fs/ggml/ggml.go` → Routes to new engine → **FAILS**
- Keep `qwen3vl` **COMMENTED** to force llama.cpp runner path

### 2. llama.cpp Works WITHOUT Ollama Wrapper
- ✅ **CONFIRMED**: llama.cpp b7108 has native qwen3vl support (qwen3vl.cpp exists)
- ✅ llama.cpp works correctly when used standalone
- 🎯 **CONCLUSION**: The bug is NOT in llama.cpp

### 3. Root Cause: Ollama Configuration Issue
- 🔴 **Problem is in Ollama**: Not sending correct information to llama.cpp
- The `SET_ROWS` operation error indicates Ollama is incorrectly configuring:
  - CUDA buffer allocation
  - Context parameters
  - Split model file handling

## Technical Analysis

### Error Location
```
ggml-backend.cpp:832: pre-allocated tensor (cache_k_l0 (view)) in a buffer (CUDA0) 
that cannot run the operation (SET_ROWS)
```

**What this means:**
- Ollama pre-allocates a tensor view in CUDA buffer
- The tensor requires `SET_ROWS` operation (found in `ml/backend/ggml/ggml.go:1383`)
- CUDA backend reports it cannot execute this operation on the pre-allocated buffer
- This is a **buffer configuration mismatch**

### Code Flow Investigation

**Split Model Path Discovery** (`llm/server.go`):
```go
// Line 93: extraModelPaths stores split files
extraModelPaths []string

// Line 161: Check for split model
"non-split gguf in extra model paths while main model path is split gguf"

// Line 203: Load model with splits
llamaModel, err = llama.LoadModelFromFile(modelPath, extraModelPaths, ...)

// Line 359-361: Pass splits to runner
for i := range extraModelPaths {
    params = append(params, "--model", extraModelPaths[i])
}
```

**llama.cpp Split Handling** (`llama/llama.cpp/src/llama-model-loader.cpp`):
```cpp
// Line 532: Split detection
uint16_t n_split = 0;
get_key(llm_kv(LLM_KV_SPLIT_COUNT), n_split, false);

// Line 535-540: Load additional contexts if n_split > 1
if (n_split > 1) {
    uint16_t idx = 0;
    const std::string kv_split_no = llm_kv(LLM_KV_SPLIT_NO);
    get_key(kv_split_no, idx);
    // ...
}
```

## Hypothesis

**Buffer Allocation Mismatch:**
1. Ollama allocates tensors to CUDA buffer with `no_alloc=true` initially
2. Later tries to perform `SET_ROWS` operation on pre-allocated tensor view
3. CUDA backend doesn't support this specific operation on this buffer type
4. llama.cpp expects different buffer configuration for split models

**Possible causes:**
- Ollama may not be setting correct `ggml_context` parameters for split models
- Buffer type selection (`ggml_backend_buffer_type_t`) might be wrong
- Split model tensors require host-visible memory but are allocated in device-only memory

## Next Steps for Investigation

### 1. Check Context Initialization
**File**: `ml/backend/ggml/ggml.go`
- Line 438: `ggml_backend_alloc_ctx_tensors_from_buft`
- Verify if split models need different buffer type

### 2. Check SET_ROWS Usage
**File**: `ml/backend/ggml/ggml.go`
- Line 1383: `ggml_set_rows` implementation
- Understand when this is called and why it fails for split models

### 3. Compare with llama.cpp Standalone
- Run llama.cpp directly with split model (works)
- Compare initialization parameters vs Ollama runner
- Check environment variables passed

### 4. Check Runner Startup
**File**: `llm/server.go`
- Line 329: `StartRunner` function
- Line 356-361: Model path parameters
- Verify extraModelPaths are correctly passed

## Workaround Status

### Current Configuration
- `qwen3vl` **COMMENTED** in `fs/ggml/ggml.go` (lines 276, 1000)
- Forces llama.cpp runner path
- Still fails with SET_ROWS error

### Failed Approaches
1. ❌ Uncommenting qwen3vl → New engine doesn't support splits
2. ❌ Local create.go fix → Bypassed by HuggingFace pull
3. ❌ Current llama.cpp runner → Buffer configuration issue

## Test Logs
- `q2_04split.log` through `q2_08split.log` - Old llama.cpp (no architecture support)
- `q2_09split.log` - New llama.cpp b7108 (SET_ROWS error) - **RESOLVED** after model re-download
- `q2_10split.log` - **NEW ERROR**: `GGML_ASSERT((n_outputs_prev + n_outputs)*n_embd <= (int64_t) embd_size) failed`
- `q2_11split.log` - Testing after re-download

## Root Cause Identified

**Location**: `llama/llama.cpp/src/llama-context.cpp:1309`

```cpp
embd_size = has_embd ? n_embd*n_outputs_max : 0;
```

**Problem**: 
- Calculation uses `model.hparams.n_embd` (2048 for text model)
- Does NOT account for vision projector embedding size (1024)
- When generating with multimodal input, needs combined embedding space
- Buffer too small → assertion fails at line 1185

**Evidence from q2_10split.log**:
```
print_info: n_embd           = 2048    ← Text model embedding
print_info: n_embd_inp       = 8192    ← Input embedding
load_hparams: n_embd:        1024      ← Vision projector embedding  
load_hparams: projection_dim: 2048     ← Projects vision → text space
```

**Error occurs during generation**:
```
llama-context.cpp:1185: GGML_ASSERT((n_outputs_prev + n_outputs)*n_embd <= (int64_t) embd_size) failed
```
- Model loads successfully ✅
- Runner starts successfully ✅
- Fails only when trying to generate tokens ❌

## Required Changes

**Priority 1: Fix embd_size calculation in llama.cpp**

**File**: `llama/llama.cpp/src/llama-context.cpp` line 1309

**Current code**:
```cpp
embd_size = has_embd ? n_embd*n_outputs_max : 0;
```

**Proposed fix**:
```cpp
// Account for multimodal projectors (e.g., qwen3vl vision → text projection)
uint32_t n_embd_total = n_embd;
if (model.has_projector() && model.projector_n_embd > 0) {
    n_embd_total = std::max(n_embd, model.projector_n_embd);
}
embd_size = has_embd ? n_embd_total*n_outputs_max : 0;
```

**Alternative workaround**: Increase buffer size multiplier for known multimodal architectures:
```cpp
uint32_t embd_multiplier = 1;
if (model.arch == LLM_ARCH_QWEN3VL || model.arch == LLM_ARCH_MLLAMA) {
    embd_multiplier = 2;  // Account for vision projector
}
embd_size = has_embd ? n_embd*n_outputs_max*embd_multiplier : 0;
```

**Priority 2: Patch llama.cpp in Ollama fork**

Since this is a llama.cpp bug, options:
1. Create patch file in `llama/patches/` directory
2. Submit fix to llama.cpp upstream (PR to ggerganov/llama.cpp)
3. Wait for upstream fix and update FETCH_HEAD

**Priority 3: Test with non-split qwen3vl model**

Compare behavior with single-file qwen3vl model to verify:
- Is this specific to split models? 
- Or affects all qwen3vl multimodal models?

---

## Fix Applied - ITERATION 3 (Complete - Image Support)

**Date**: 2025-11-28 19:10
**Patch**: `llama/patches/0032-fix-multimodal-embd-size-calculation.patch` (updated)
**File Modified**: `llama/llama.cpp/src/llama-context.cpp` lines 913, 1182-1189, 1318

### Issue 1 (Iteration 1):
Test q2_12split.log showed different error after first patch:
```
ggml-backend.cpp:285: GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor read out of bounds") failed
```

**Cause**: First patch only fixed buffer allocation, but tensor read/write operations still used `n_embd` (2048) instead of actual tensor dimension (8192).

### Final Fix (Iteration 2):

**Line 913** - Use actual tensor dimension:
```cpp
// OLD:
GGML_ASSERT(n_tokens*n_embd <= (int64_t) embd_size);
ggml_backend_tensor_get_async(..., n_tokens*n_embd*sizeof(float));

// NEW:
const uint32_t n_embd_actual = t_embd->ne[0];
GGML_ASSERT(n_tokens*n_embd_actual <= (int64_t) embd_size);
ggml_backend_tensor_get_async(..., n_tokens*n_embd_actual*sizeof(float));
```

**Lines 1182-1189** - Use actual tensor dimension:
```cpp
// OLD:
float * embd_out = embd + n_outputs_prev*n_embd;
GGML_ASSERT((n_outputs_prev + n_outputs)*n_embd <= (int64_t) embd_size);
ggml_backend_tensor_get_async(..., n_outputs*n_embd*sizeof(float));

// NEW:
const uint32_t n_embd_actual = t_embd->ne[0];
float * embd_out = embd + n_outputs_prev*n_embd_actual;
GGML_ASSERT((n_outputs_prev + n_outputs)*n_embd_actual <= (int64_t) embd_size);
ggml_backend_tensor_get_async(..., n_outputs*n_embd_actual*sizeof(float));
```

**Line 1318** - Allocate sufficient buffer:
```cpp
// OLD:
embd_size = has_embd ? n_embd*n_outputs_max : 0;

// NEW:
uint32_t n_embd_buf = n_embd;
if (model.hparams.n_embd_inp() > n_embd) {
    n_embd_buf = model.hparams.n_embd_inp();
}
embd_size = has_embd ? n_embd_buf*n_outputs_max : 0;
```

**Explanation**:
- Text model: `n_embd = 2048`
- Vision projector: `n_embd = 1024` → `n_embd_inp = 8192`
- Tensor `t_embd->ne[0]` = 8192 at runtime
- Problem: Code assumed all embeddings are 2048, but vision produces 8192
- Solution: Use `t_embd->ne[0]` for actual dimension, `n_embd_inp()` for allocation

**Next Steps**:
1. Recompile Ollama with updated patch
2. Test with Qwen3-VL-2B-Instruct split model
3. Verify generation works without tensor read errors
4. Consider submitting patch to llama.cpp upstream

---

**Last Updated**: 2025-11-28 19:10
### Issue 2 (Iteration 2):
Test q2_13split.log - text works, image crashes with:
```
Exception 0xc0000005 (access violation)
PC=0x7fff31bddaeb
```

**Cause**: Found 3 MORE locations using incorrect n_embd:
1. **Line 642** `get_embeddings_ith()`: Calculated pointer offset with n_embd (2048) → returned wrong address → access violation
2. **Line 1388** `output_reorder()`: Swapped only 2048 values instead of 8192 → memory corruption
3. **Line 1922** `state_write_data()`: Truncated embeddings to 2048 during save → data loss

### Final Fix (Iteration 3):

**ALL LOCATIONS FIXED** (5 total):

1. **Line 642** - `get_embeddings_ith()`:
   ```cpp
   // OLD: return embd + j*model.hparams.n_embd;
   // NEW: 
   const uint64_t n_embd_stride = model.hparams.n_embd_inp() > model.hparams.n_embd 
                                  ? model.hparams.n_embd_inp() : model.hparams.n_embd;
   return embd + j*n_embd_stride;
   ```

2. **Line 913** - Already fixed in iter 2

3. **Lines 1182-1189** - Already fixed in iter 2

4. **Line 1318** - Already fixed in iter 2

5. **Lines 1388-1389** - `output_reorder()`:
   ```cpp
   // OLD:
   const uint64_t n_embd = model.hparams.n_embd;
   for (uint64_t k = 0; k < n_embd; k++) {
       std::swap(embd[i0*n_embd + k], embd[i1*n_embd + k]);
   }
   
   // NEW:
   const uint64_t n_embd_actual = model.hparams.n_embd_inp() > model.hparams.n_embd 
                                  ? model.hparams.n_embd_inp() : model.hparams.n_embd;
   for (uint64_t k = 0; k < n_embd_actual; k++) {
       std::swap(embd[i0*n_embd_actual + k], embd[i1*n_embd_actual + k]);
   }
   ```

6. **Line 1922** - `state_write_data()`:
   ```cpp
   // OLD:
   const uint64_t embd_size = std::min(..., n_outputs * model.hparams.n_embd);
   
   // NEW:
   const uint64_t n_embd_effective = model.hparams.n_embd_inp() > model.hparams.n_embd 
                                     ? model.hparams.n_embd_inp() : model.hparams.n_embd;
   const uint64_t embd_size = std::min(..., n_outputs * n_embd_effective);
   ```

**Complete Fix Summary**:
- ✅ Buffer allocation: uses n_embd_inp (8192)
- ✅ Tensor operations: uses t_embd->ne[0] (runtime dimension)
- ✅ Pointer arithmetic: uses n_embd_inp for stride
- ✅ Memory swaps: uses full embedding size
- ✅ Serialization: saves complete embeddings

**Status**: PATCH FULLY UPDATED (iteration 3) - Ready for recompile and testing with images
**Branch**: 14-00 (7 commits ahead of origin)
**Files Changed**: 
- `llama/patches/0032-fix-multimodal-embd-size-calculation.patch` (updated)
- `llama/llama.cpp/src/llama-context.cpp` (patched 3 locations)
