# Multimodal Hallucination Bug Fix - Qwen3-VL

## Problem Description
The Qwen3-VL model produces **non-deterministic results** when processing images:
- Same image + same prompt = different results between sessions
- Server restart → correct result
- `/bye` and reload (without server restart) → incorrect result (hallucinations)
- Model enters **infinite repetition loops** and doesn't stop generating

## Root Cause Analysis
1. **Image cache** was reusing embeddings with the same pointer across requests
2. **KV cache** found false "matches" (same pointer = equal) and reused stale/corrupt data
3. **No detection** of repetition loops - model could repeat forever

## Files Modified

### 1. runner/llamarunner/image.go
**Change**: Disabled image cache entirely
**Function**: `MultimodalTokenize()` now always regenerates embeddings fresh
**Key code** (around line 114):
```go
slog.Debug("encoding image fresh (cache disabled for consistency)")
// Don't cache for now - see comment above
// c.addImage(hash, chunks)
```
**Reason**: Cached embeddings caused KV cache to falsely match and reuse stale data

### 2. runner/llamarunner/cache.go
**Change**: Clear KV cache when prompt contains embeddings (images)
**Function**: `LoadCacheSlot()` detects `hasEmbeddings` and calls `KvCacheSeqRm(slot.Id, 0, -1)`
**Key code** (around lines 65-108):
```go
// Check if prompt contains any embeddings (images)
hasEmbeddings := false
for _, inp := range prompt {
    if inp.embed != nil {
        hasEmbeddings = true
        break
    }
}
// ... later ...
if hasEmbeddings && numPast > 0 {
    slog.Debug("clearing KV cache for prompt with embeddings", "id", slot.Id, "numPast", numPast)
    c.lc.KvCacheSeqRm(slot.Id, 0, -1)
    numPast = 0
}
```
**Reason**: Prevents reusing KV cache entries that may contain stale computation results

### 3. runner/llamarunner/runner.go
**Change**: Added repetition loop detection
**New fields in `Sequence` struct**:
```go
recentTokens     []int  // circular buffer of recent tokens (256)
recentTokensIdx  int    // current index in circular buffer
repetitionCount  int    // count of detected repetitions
```
**New function**: `detectRepetitionLoop(token int) bool`
- Stores tokens in circular buffer
- Checks for repeating patterns of 8-64 tokens
- Returns true after detecting 3 consecutive repetitions
- Logs warning: "repetition loop detected"

**In generation loop** (after sampling token):
```go
if seq.detectRepetitionLoop(token) {
    slog.Warn("stopping generation due to repetition loop", "numPredicted", seq.numPredicted)
    s.removeSequence(i, llm.DoneReasonStop)
    continue
}
```
**Reason**: Stops infinite repetition loops that the model sometimes enters

## Test Procedure
1. Build: `powershell -ExecutionPolicy Bypass -File C:\IA\tools\ollama\z_iosu_2\scripts\build_windows.ps1 buildCPU buildCuda13 buildOllama`
2. Start server: `.\ollama.exe serve`
3. Run client: `.\ollama.exe run hf.co/unsloth/Qwen3-VL-8B-Instruct-GGUF:Q4_K_M`
4. Test with image: `C:\IA\2.png extrae todos los datos`
5. Verify:
   - No hallucinations (correct data extraction)
   - No infinite repetition loop
   - Ctrl+C works in client

## Test Logs Location
- `z_iosu_2/logs3/q3_02-1.log` - Shows incorrect/mixed data (hallucinations)
- `z_iosu_2/logs3/q3_02-2.log` - Shows infinite repetition loop

## Expected Log Messages (if fixes are active)
- `"encoding image fresh (cache disabled for consistency)"` - image.go
- `"clearing KV cache for prompt with embeddings"` - cache.go (only if numPast > 0)
- `"loading cache slot" ... used=0` - cache.go (should show used=0 for image prompts)
- `"repetition loop detected"` - runner.go (only if loop is detected and stopped)

## If Problems Persist
1. Check if build completed successfully (Exit Code: 0)
2. Verify log messages appear (fixes are compiled in)
3. If still hallucinating: the issue may be deeper in the model or llama.cpp
4. If still looping: adjust detection parameters (patternLen, repetitionCount threshold)

## Related Code Locations
- Image embedding generation: `runner/llamarunner/image.go` → `MultimodalTokenize()`
- KV cache management: `runner/llamarunner/cache.go` → `LoadCacheSlot()`
- Token generation loop: `runner/llamarunner/runner.go` → `processBatch()` around line 595
- Sampling parameters: `runner/llamarunner/runner.go` → line 720 (`SamplingParams`)
