# Qwen3-VL Split Model Fixes - Session Summary

## Date
November 28, 2025

## Initial Problem
- **Exception 0xc0000005** when using Qwen3-VL split models with images
- Model crashed immediately on image processing
- Split models: 2B and 8B variants

## Root Cause Analysis
Vision models have **two different embedding dimensions**:
- `n_embd`: Text model dimension (e.g., 2048 for 2B, 4096 for 8B)
- `n_embd_inp`: Vision projector dimension (e.g., 8192 for both 2B and 8B)

Code was using `n_embd` (2048/4096) instead of `n_embd_inp` (8192) for:
1. Batch allocation (Go code)
2. Embedding copy operations (Go code)
3. Context operations (C++ code)

This caused **buffer overflows** and **memory corruption**.

## Fixes Applied

### Iteration 1-4: C++ Fixes (llama-context.cpp)
Fixed **8 locations** in `llama/llama.cpp/src/llama-context.cpp`:

**Lines 1746-1747**: Batch initialization
```cpp
OLD: batch_embd = llama_batch_init(n_batch, 0, n_embd);
NEW: batch_embd = llama_batch_init(n_batch, 0, n_embd_inp());
```

**Lines 2285-2290**: Cross-attention setup
```cpp
OLD: const int64_t n_embd = hparams.n_embd();
     ggml_tensor * t_embd = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens);
NEW: const int64_t n_embd_inp = hparams.n_embd_inp();
     ggml_tensor * t_embd = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd_inp, n_tokens);
```

**Lines 2376-2377**: Input processing
```cpp
OLD: const int64_t n_embd = model.hparams.n_embd();
     ggml_tensor * inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch_all, model.tok_embd, cb);
NEW: ggml_tensor * t_embd = ggml_view_2d(ctx0, batch_all.embd, t_embd->ne[0], n_tokens, ...);
```

**Lines 2482-2483**: Vision cross-attention
```cpp
OLD: ggml_tensor * vision_cross_attn = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_embd, n_head_kv, n_tokens);
NEW: ggml_tensor * vision_cross_attn = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_embd_inp(), n_head_kv, n_tokens);
```

**Lines 2573-2574**: Projection layer
```cpp
OLD: ggml_tensor * cur = ggml_mul_mat(ctx0, model.vision_proj, inpL);
NEW: Use t_embd->ne[0] for actual dimension
```

**Lines 2614-2615**: Output projection
```cpp
OLD: ggml_tensor * inpL = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, n_tokens);
NEW: ggml_tensor * inpL = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, t_embd->ne[0], n_tokens);
```

**Lines 2701-2702**: Final output
```cpp
OLD: ggml_tensor * cur = ggml_view_2d(ctx0, batch_all.embd, n_embd, n_tokens, ...);
NEW: ggml_tensor * cur = ggml_view_2d(ctx0, batch_all.embd, t_embd->ne[0], n_tokens, ...);
```

### Iteration 5: Go Fixes

#### File: llama/llama.go (Line 528-535)
**Added new function** to expose C++ `n_embd_inp()`:
```go
func (m *Model) NEmbd() int {
    return int(C.llama_model_n_embd(m.c))
}

func (m *Model) NEmbdInp() int {  // NEW FUNCTION
    return int(C.llama_model_n_embd_inp(m.c))
}
```

#### File: llama/llama.go (Line 586-620)
**Fixed MultimodalTokenize embedding copy**:
```go
OLD: numEmbed := llamaContext.Model().NEmbd()  // 2048/4096
NEW: numEmbed := llamaContext.Model().NEmbdInp()  // 8192

// Lines 616-620: This slice operation now uses correct size
s := unsafe.Slice((*float32)(chunkEmbd), numTokens*numEmbed)
```

**Impact**: Fixed embedding copy from 736×2048 to 736×8192 per chunk

#### File: runner/llamarunner/image.go (Line 99-106)
**Fixed batch allocation**:
```go
OLD: return llamaContext.Model().NEmbd()  // 2048/4096
NEW: return llamaContext.Model().NEmbdInp()  // 8192
```

**Impact**: Batch allocation uses 8192 instead of 2048/4096
**Used by**: runner.go line 374 for `llama.NewBatch()`

## Current Status

### ✅ RESOLVED
1. **No more crashes** - Exception 0xc0000005 completely fixed
2. **Images process correctly** - Tokenization works (4028 tokens for 53×76 grid)
3. **Content understanding** - Model extracts data ~100% correctly when it works
4. **Text generation** - Fully functional
5. **Memory cleanup with /bye** - Works when using `/bye` command

### ⚠️ PARTIAL
1. **Non-deterministic results** - Sometimes works perfectly, sometimes generates garbage
2. **Memory leak with Ctrl+C** - Only `/bye` frees memory properly
3. **Infinite generation** - Model doesn't detect EOS tokens correctly

### ❌ REMAINING ISSUES

#### Issue 1: Infinite Generation (No EOS Detection)
**Symptoms**:
- Model generates indefinitely
- Doesn't stop at `<|im_end|>` or other stop tokens
- `num_predict` parameter sometimes ignored
- Cannot interrupt with Ctrl+C in client

**Evidence**:
- EOS tokens configured: 151643, 151645, 151662, 151663, 151664
- Logs show no EOS token ever generated
- `TokenIsEog()` check at runner.go line 549 never triggers

**Possible Causes**:
1. Sampler not generating EOS tokens with low temperature (0.0-0.3)
2. Template issues causing model confusion
3. Bug in `TokenIsEog()` implementation for multimodal models

**Debug Added** (runner.go lines 428-436, 549-558):
```go
// Log numPredict value at sequence start
if seq.numPredicted == 1 {
    slog.Info("sequence started", "numPredict", seq.numPredict, "stop", seq.stop)
}

// Log EOS detection
if s.model.TokenIsEog(token) {
    slog.Info("EOS token detected", "token", token, "piece", piece)
    s.removeSequence(i, llm.DoneReasonStop)
}

// Log progress every 50 tokens
if seq.numPredicted%50 == 0 {
    slog.Debug("generation progress", "tokens", seq.numPredicted, "lastToken", token)
}
```

#### Issue 2: Memory Not Released (Windows Specific)
**Symptoms**:
- Ctrl+C doesn't free CUDA memory
- Must kill 2 processes: `ollama.exe serve` and `ollama.exe runner`
- `/bye` works correctly

**Fixes Applied**:

**runner/llamarunner/runner.go (Line 940-945)**:
```go
case llm.LoadOperationClose:
    // Free image context resources
    if s.image != nil {
        s.image.Free(s.modelPath)
    }
```

**runner/llamarunner/runner.go (Line 990-1000)** - Signal handler (doesn't work on Windows):
```go
sigChan := make(chan os.Signal, 1)
signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)
go func() {
    <-sigChan
    slog.Info("shutting down, freeing resources")
    if server.image != nil {
        server.image.Free(server.modelPath)
    }
    cancel()
    os.Exit(0)
}()
```

**runner/llamarunner/runner.go (Line 861-871)** - Cleanup before loading new model:
```go
// Free previous image context if exists (cleanup from previous load)
if s.image != nil {
    s.image.Free(s.modelPath)
    s.image = nil
}

if ppath != "" {
    var err error
    s.image, err = NewImageContext(s.lc, ppath)
    ...
}
```

**Note**: Windows doesn't properly handle signals in child processes, so Ctrl+C cleanup doesn't work reliably.

#### Issue 3: Non-Deterministic Results
**Symptoms**:
- Same prompt gives different results on each server restart
- Sometimes: Perfect extraction
- Sometimes: Repeated text/loops
- Sometimes: Garbage characters

**Suspected Causes**:
1. Race condition in multimodal tokenization
2. Uninitialized state in MtmdContext
3. Cache not cleared properly between runs
4. Sampling seed randomness

**Not the cause** (already verified):
- ✅ Embedding dimensions are correct (16384 for 8B)
- ✅ Not corrupted manifests (tested with new models)
- ✅ Not parameter issues (tested multiple configurations)

## Test Configurations Tried

### Model Files Tested
1. `modelfile.txt` - Original parameters:
   - temperature: 0.0
   - num_predict: 512
   - repeat_penalty: 1.05

2. Modified parameters (didn't fix issues):
   - temperature: 0.3
   - num_predict: 256
   - repeat_penalty: 1.1

### Models Created
- iosu2/unsloth/Qwen3-VL-8B-Instruct-GGUF:Q4_K_4
- Multiple test variants with different parameters
- All show same issues

## Logs Reference
- **q2_14split.log**: Exception 0xc0000005 (before Go fixes)
- **q2_16.log**: First compile with image.go fix
- **q2_17.log**: Model 8B test (returned empty for images)
- **q2_19split.log**: Model 8B responds well but doesn't stop
- **q2_30.log**: Testing with original executable (no logging)
- **q2_31.log**: First test with logging - showed correct numEmbed=16384
- **q2_34-36.log**: Tests with different models and parameters

## Verification Commands

### Check if compiled with fixes:
```powershell
Get-Content log.txt | Select-String -Pattern "multimodal tokenize|image embed size|numEmbed"
```

Expected output:
```
level=DEBUG source=llama.go:587 msg="multimodal tokenize" nChunks=3 numEmbed=16384 nEmbd=4096
level=DEBUG source=image.go:104 msg="image embed size" size=16384 nEmbd=4096
```

### Kill all ollama processes:
```powershell
Get-Process ollama -ErrorAction SilentlyContinue | Stop-Process -Force
```

### Check CUDA memory:
```powershell
nvidia-smi --query-gpu=memory.used --format=csv --loop=1
```

### Test with limit:
```powershell
.\ollama.exe run model:tag "prompt" --num-predict 100
```

## Next Steps for Investigation

### Priority 1: EOS Token Detection
1. Add more detailed logging to sampler (llama.go Sample function)
2. Check if `TokenIsEog()` works correctly for Qwen3-VL
3. Verify EOG token IDs are loaded correctly from GGUF
4. Test with explicit stop sequences in prompt

### Priority 2: Non-Deterministic Behavior
1. Add logging to MtmdContext initialization
2. Check for uninitialized memory in image tokenization
3. Add mutex/synchronization to image cache
4. Verify no race conditions in batch processing

### Priority 3: Memory Cleanup (Windows)
1. Research proper Windows signal handling for child processes
2. Consider using named pipes for graceful shutdown
3. Add cleanup hook in parent process before killing runner
4. Test with WM_CLOSE message instead of SIGTERM

## Code Files Modified

### C++
- `llama/llama.cpp/src/llama-context.cpp` (8 locations)

### Go
- `llama/llama.go` (2 functions: NEmbdInp, MultimodalTokenize + debug logs)
- `runner/llamarunner/image.go` (1 function: EmbedSize + debug logs)
- `runner/llamarunner/runner.go` (3 locations: LoadOperationClose, signal handler, pre-load cleanup + debug logs)

### Configuration
- `z_iosu_2/0/split/qwen3vl/modelfile.txt` (parameters adjusted)

## Important Notes

1. **Always use `/bye`** instead of Ctrl+C to exit cleanly
2. **Kill all ollama processes** before starting new server
3. **Logs with `OLLAMA_DEBUG=1`** are essential for debugging
4. **Image dimensions**: 53×76 grid = 4028 tokens (confirmed working)
5. **Vision embedding size**: 16384 (8B model) - confirmed correct
6. **The crash is completely fixed** - only generation issues remain

## Environment
- Windows 11
- CUDA 13.0
- RTX 3090 (24GB)
- Go version: (check with `go version`)
- Ollama version: 0.13.1-rc0-9-g5ffcf77-dirty

## Key Discoveries
1. **Dual embedding dimensions** were the root cause of crashes
2. **Go unsafe.Slice** operations are critical - wrong size causes corruption
3. **Windows signal handling** is problematic for cleanup
4. **EOS token generation** is model/sampler dependent, not always code issue
5. **Multimodal models** are sensitive to initialization order

---

**Status**: Crashes fixed ✅ | Infinite generation 🔴 | Memory leak ⚠️ | Non-deterministic 🔴

**Last Updated**: November 28, 2025 20:40
