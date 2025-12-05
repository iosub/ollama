# PR Objective: Split Vision Model Support (Qwen3-VL)

## Goal

Enable Ollama to run **split multimodal models** like Qwen3-VL, which come as separate GGUF files (text + vision).

## Dependencies

This PR depends on:
- **PR #12992** (`dhiltgen/ggml_bump`) - Updates llama.cpp with qwen3vl architecture support ✅ MERGED
- **PR #13259** (split-gguf) - MetaGGML and ForeignTensors for loading split GGUF files

### Why PR #13259?

> **Answer**: After investigation, PR #13259 may NOT be strictly required. The `ggml_bump` branch already includes commit `aba15753` which makes split vision models fallback to the **llama.cpp runner** (not the new Ollama engine). 
>
> The llama.cpp runner already supports loading multiple GGUF files via `--model` flag.
>
> **PR #13259 Status**: ❌ **NOT MERGED** - It's still a Draft PR (opened 4 days ago by cvrunmin). It is NOT included in PR #12992 (ggml_bump) nor in origin/main.
>
> **Do we need PR #13259?** ❌ **NO** - Our implementation (branch 14-00) already includes our own MetaGGML/ForeignTensors in `fs/ggml/ggml.go` (+184 lines). We implemented the same functionality independently. Our PR is self-contained.


> **Conclusion**: We can proceed WITHOUT PR #13259 dependency if we use the llama.cpp runner path. Our PR just needs to:
> 1. Ensure split models use llama.cpp runner (already done in ggml_bump)
> 2. Fix embedding dimensions (NEmbdInp)
> 3. Add M-RoPE support in Go bindings
>
> **ACTION PLAN**:
> 1. Apply these 3 changes to `ollama-pr` (based on PR #12992)
> 2. Build and test with Qwen3-VL
> 3. If it works → Create the PR

## Problem Statement

Split vision models (Qwen3-VL, Qwen2-VL) fail in Ollama because:

1. **Wrong embedding dimension**: Ollama uses `n_embd` (2048 for Qwen3-VL-2B) instead of `n_embd_inp` (8192) for vision embeddings
2. **Split file detection**: Vision GGUF file is incorrectly classified as "projector" instead of model part
3. **M-RoPE positions**: Qwen3-VL requires Multi-dimensional Rotary Position Embedding (4 position values per image token: temporal, y, x, unused) but Ollama only sets 1 position per token

## Changes Required

### 1. Embedding Dimension Fix
**Files**: `llama/llama.go`, `runner/llamarunner/image.go`

- Add `NEmbdInp()` function to get vision embedding dimension
- Use `NEmbdInp()` instead of `NEmbd()` for multimodal tokenization

### 2. Split Model Detection  
**File**: `server/create.go`

- Check `split.count` metadata to distinguish split model parts from legacy projectors
- If `split.count > 1`, treat as model file, not projector

### 3. M-RoPE Batch Support
**Files**: `llama/llama.go`, `runner/llamarunner/runner.go`, `runner/llamarunner/image.go`

- Add `NewBatchMRoPE()` for creating batches with 4 position arrays
- Add `AddImageMRoPE()` to set 2D grid positions for image tokens
- Add `UsesMRoPE()` to detect if vision model needs M-RoPE
- Process M-RoPE images atomically (all tokens in one batch)

### 4. Memory Cleanup
**File**: `runner/llamarunner/runner.go`

- Free image context on model reload and close
- Proper cleanup in signal handler

### 5. KV Cache Handling
**File**: `runner/llamarunner/cache.go`

- Clear KV cache when processing prompts with image embeddings
- Prevents stale cache from corrupting vision inference

---

## Testing Strategy

### Setup
- `ollama-pr` repo: Based on PR #12992 (ggml_bump) ✅
- Apply our changes on top
- Build and test

### Files to copy from `ollama` (14-00) to `ollama-pr`

**Core files (18 total modified in 14-00):**
```
discover/runner.go
fs/ggml/ggml.go          # MetaGGML, ForeignTensors (+184 lines)
fs/ggml/gguf.go
llama/llama.go           # NEmbdInp, M-RoPE batch functions
llm/server.go            # Split model loading
ml/backend.go
ml/backend/ggml/ggml.go
model/model.go
runner/llamarunner/cache.go    # KV cache clearing
runner/llamarunner/image.go    # UsesMRoPE, grid dimensions
runner/llamarunner/runner.go   # M-RoPE processing
runner/ollamarunner/runner.go
server/create.go         # Split detection
server/images.go
server/routes.go
server/sched.go
```

**llama.cpp internal (probably from ggml_bump already):**
```
llama/llama.cpp/src/llama.go
llama/llama.cpp/src/models/models.go
```

---

## Current Status

| Component | In ollama-pr | Needed | Status |
|-----------|--------------|--------|--------|
| NEmbdInp() | ✅ | ✅ | Done |
| Split detection | ✅ | ✅ | Done |
| Memory cleanup | ✅ | ✅ | Done |
| M-RoPE batch | ❌ | ✅ | **TODO** |
| M-RoPE processing | ❌ | ✅ | **TODO** |
| UsesMRoPE() | ❌ | ✅ | **TODO** |
| KV cache clear | ❌ | ✅ | **TODO** |

---

## Files to Copy from ollama (14-00)

### High Priority (Required for functionality)
- [ ] `llama/llama.go` - M-RoPE batch functions
- [ ] `runner/llamarunner/runner.go` - M-RoPE processing logic
- [ ] `runner/llamarunner/image.go` - UsesMRoPE(), grid dimensions
- [ ] `runner/llamarunner/cache.go` - KV cache clearing

### Medium Priority (Split model loading)
- [ ] `llm/server.go` - Split model path handling

### Lower Priority (May be in PR #13259)
- [ ] `fs/ggml/ggml.go` - MetaGGML support
- [ ] `server/images.go` - Layer handling

---

## Test Plan

1. Load split Qwen3-VL model: `ollama run hf.co/unsloth/Qwen3-VL-8B-Instruct-GGUF:Q4_K_M`
2. Send image + text prompt
3. Verify model describes image correctly (no hallucinations)
4. Verify memory cleanup with `/bye`

---

## Notes

- Without M-RoPE, images will be encoded with linear positions → model hallucinates
- M-RoPE is critical: it's what makes vision models understand spatial relationships in images
- The C++ llama.cpp already supports M-RoPE; this PR exposes it to Go

