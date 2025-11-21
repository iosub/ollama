# Patch 0033: Qwen3VL Deepstack Fix

## Summary
Fixes illegal memory access in qwen3vl and qwen3vlmoe models when processing text-only inputs (without image embeddings).

## Problem
The generic `llm_build_qwen3` and `llm_build_qwen3moe` structures used by Ollama for qwen3vl models do not support the deepstack feature required for vision-language models. This causes:
- Illegal memory access when processing images
- Incorrect behavior for multimodal inputs
- Runtime crashes with qwen3vl models

## Solution
Created dedicated `llm_build_qwen3vl` and `llm_build_qwen3vlmoe` structures that:
1. Extract deepstack embeddings when image inputs are present (`ubatch.embd == true`)
2. Split input embeddings into main_embed and deepstack (ds0, ds1, ds2)
3. Add deepstack tensors at specific layers (0, 1, 2) for image inputs
4. Avoid creating unnecessary tensors for text-only inputs

## Changes
- **Added**: `llm_build_qwen3vl` structure with deepstack support
- **Added**: `llm_build_qwen3vlmoe` structure with deepstack support (MoE variant)
- **Modified**: Switch case for `LLM_ARCH_QWEN3VL` to use `llm_build_qwen3vl`
- **Modified**: Switch case for `LLM_ARCH_QWEN3VLMOE` to use `llm_build_qwen3vlmoe`

## Technical Details
### Deepstack Handling
```cpp
// Extract deepstack only when image embeddings present
if (ubatch.embd) {
    const int64_t n_embd_main = n_embd / 4;
    main_embed = ggml_view_2d(ctx0, inpL, n_embd_main, n_tokens, inpL->nb[1], 0);
    ds0 = ggml_view_2d(ctx0, inpL, n_embd_main, n_tokens, inpL->nb[1], n_embd_main * sizeof(float));
    ds1 = ggml_view_2d(ctx0, inpL, n_embd_main, n_tokens, inpL->nb[1], 2 * n_embd_main * sizeof(float));
    ds2 = ggml_view_2d(ctx0, inpL, n_embd_main, n_tokens, inpL->nb[1], 3 * n_embd_main * sizeof(float));
    inpL = main_embed;
}
```

### Layer Addition
```cpp
// Add deepstack at specific layers (only for image inputs)
if (ubatch.embd) {
    switch (il) {
        case 0: cur = ggml_add(ctx0, cur, ds0); break;
        case 1: cur = ggml_add(ctx0, cur, ds1); break;
        case 2: cur = ggml_add(ctx0, cur, ds2); break;
    }
}
```

## Based On
- LETS-BEE fork commit: https://github.com/LETS-BEE/llama.cpp/commit/de0e3d3c3ce4b394746ade9263736c8edb40260e
- Adapted for Ollama's llama.cpp integration using `LLM_ARCH_QWEN3VL` naming

## Impact
- **Fixes**: Illegal memory access for qwen3vl models
- **Enables**: Proper vision-language processing
- **Maintains**: Backward compatibility with text-only inputs

## Testing
To test this patch:
1. Load a qwen3vl model with vision capabilities
2. Process text-only input (should work without crashes)
3. Process image + text input (should properly utilize deepstack)
4. Verify no memory access violations in logs

## Files Modified
- `llama/llama.cpp/src/llama-model.cpp` (+290 lines)

## Date
October 24, 2025

## Author
Adapted from LETS-BEE/llama.cpp for Ollama integration
