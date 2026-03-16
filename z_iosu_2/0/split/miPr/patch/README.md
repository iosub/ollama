# M-RoPE Vision Model Patches

Patches to fix GGML_ASSERT errors when running vision models with M-RoPE (Multi-modal Rotary Position Embedding) like Qwen3-VL.

## Patches

### 01-llama-context.patch
**File:** `llama/llama.cpp/src/llama-context.cpp`

**Purpose:** Fix embedding buffer size and extraction for multimodal models.

**Changes:**
- `decode()`: Separate `n_embd_inp` (16384) from `n_embd` (4096) for batch initialization
- `decode()` POOLING_TYPE_NONE: Use `t_embd->ne[0]` for actual tensor dimension
- `encode()` POOLING_TYPE_NONE: Use `t_embd->ne[0]` for actual tensor dimension  
- `output_reserve()`: Calculate `embd_size` using `n_embd_inp()` when larger than `n_embd`
- `get_embeddings_ith()`: Use `n_embd_inp()` as stride for multimodal models
- `output_reorder()`: Use `n_embd_inp()` when larger than `n_embd`

### 02-llama-kv-cache.patch
**File:** `llama/llama.cpp/src/llama-kv-cache.cpp`

**Purpose:** Disable context shifting for models with M-RoPE.

**Changes:**
- `get_can_shift()`: Return `hparams.n_pos_per_embd() == 1` to disable shifting for vision models

### 03-llama-kv-cache-iswa.patch
**File:** `llama/llama.cpp/src/llama-kv-cache-iswa.cpp`

**Purpose:** Disable context shifting for ISWA cache variant.

**Changes:**
- `get_can_shift()`: Check if base caches support shifting before allowing shift

### 04-llamarunner-runner.patch
**File:** `runner/llamarunner/runner.go`

**Purpose:** Prevent crash when context is full on models that don't support shifting.

**Changes:**
- Check `KvCacheCanShift()` before attempting shift to gracefully end sequence with `DoneReasonLength`

## How to Apply

```bash
cd ollama
git apply z_iosu_2/0/split/miPr/patch/01-llama-context.patch
git apply z_iosu_2/0/split/miPr/patch/02-llama-kv-cache.patch
git apply z_iosu_2/0/split/miPr/patch/03-llama-kv-cache-iswa.patch
git apply z_iosu_2/0/split/miPr/patch/04-llamarunner-runner.patch
```

## Issues Fixed

1. **GGML_ASSERT seq_add() error** - Context shifting used `seq_add()` which doesn't support `n_pos_per_embd() > 1`
2. **GGML_ASSERT embd_size overflow** - Embedding buffer was allocated with `n_embd=4096` but accessed with `n_embd_inp=16384`
3. **Crash on context full** - Runner attempted reprocessing with large image embeddings that didn't fit

## Model Compatibility

These patches are required for:
- Qwen3-VL (all sizes)
- Qwen2-VL (all sizes)
- Other vision models using M-RoPE with `n_pos_per_embd > 1`
