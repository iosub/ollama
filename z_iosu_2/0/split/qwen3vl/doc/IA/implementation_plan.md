# Implementation Plan - Fix Qwen3-VL Split GGUF Crash

## Goal
Resolve the `0xc0000005` Access Violation crash when running Qwen3-VL split GGUF models in Ollama.

## Problem
The split GGUF model contains `rope.dimension_sections` metadata, enabling MRoPE (Multi-axis RoPE) in `llama.cpp`. Ollama sends linear position IDs (0, 1, 2...) which are incompatible with MRoPE, causing a crash in the RoPE kernel. The non-split model works because it lacks this metadata, forcing standard RoPE.

## Proposed Changes

### 1. Disable MRoPE for Qwen3-VL in `llama.cpp`
Modify `llama-model.cpp` to ignore the `rope.dimension_sections` metadata for `LLM_ARCH_QWEN3VL`. This will force the model to use standard RoPE, matching the behavior of the working non-split model and Ollama's linear position IDs.

#### [MODIFY] [llama-model.cpp](file:///c:/IA/tools/ollama/llama/llama.cpp/src/llama-model.cpp)
- Locate `case LLM_ARCH_QWEN3VL:` in `load_hparams`.
- Comment out or remove `ml.get_key_or_arr(LLM_KV_ROPE_DIMENSION_SECTIONS, ...);`.
- **Update**: Manually setting `rope_sections` to `{head_dim, 0, 0, 0}` was attempted to satisfy assertions, but the crash persisted. The final strategy is to ensure the MRoPE code path is NOT taken at all, likely by ensuring `rope_type` is NOT set to `MROPE`.

### 2. Disable `mmap` for Vision Models (Optional/Safety)
While `mmap` was not the root cause, disabling it for vision models avoids potential memory conflicts during debugging.

#### [MODIFY] [server.go](file:///c:/IA/tools/ollama/llm/server.go)
- In `NewLlamaServer`, ensure `UseMMap` is false if a projector is present.

## Verification Plan

### Automated Tests
- None available for this specific crash.

### Manual Verification
1.  **Build**: Compile Ollama with the changes.
2.  **Run**: `ollama run hf.co/unsloth/Qwen3-VL-8B-Instruct-GGUF:Q4_K_M`
3.  **Test**: Input an image and a prompt (e.g., "Describe this image").
4.  **Success Criteria**: The model generates a response without crashing.
