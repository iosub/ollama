# Converter Differences: Ollama vs. llama.cpp (Qwen3-VL)

## Overview
We observed that Qwen3-VL models converted by Ollama (non-split) work, while those converted by `llama.cpp` (split) crash. This document details the differences found in the conversion logic.

## Key Findings

### 1. Missing Metadata in Ollama Converter
- **File**: `convert/convert_qwen3vl.go`
- **Issue**: The Ollama converter does **not** explicitly write the `rope.dimension_sections` key to the GGUF KV pairs.
- **Llama.cpp Converter**: The Python script `convert_hf_to_gguf.py` (used for the split model) **does** write this key.

### 2. Impact of `rope.dimension_sections`
- This metadata key defines the configuration for **MRoPE** (Multi-axis Rotary Position Embedding).
- **Presence (Split Model)**: Enables MRoPE logic in `llama.cpp`. Requires complex 3D position IDs (Time, Height, Width).
- **Absence (Non-Split Model)**: Disables MRoPE logic. `llama.cpp` falls back to standard RoPE (Linear).

### 3. Why Ollama Crashes with Split Models
- Ollama's runner (`runner.go`) generates **linear position IDs** for all tokens, including vision tokens.
- When the split model is loaded, `llama.cpp` sees `rope.dimension_sections` and expects MRoPE-compatible positions.
- The mismatch between Linear Positions (Ollama) and MRoPE Logic (llama.cpp) causes the crash (Access Violation in RoPE kernel).
- The non-split model works because the missing metadata forces `llama.cpp` to use standard RoPE, which is compatible with linear positions.

## Code Evidence

### llama.cpp (`llama-arch.cpp`)
```cpp
{ LLM_KV_ROPE_DIMENSION_SECTIONS,       "%s.rope.dimension_sections" },
```

### Ollama (`convert_qwen3vl.go`)
- No reference to `rope.dimension_sections` found in the KV writing logic.
- It inherits from `qwen2` which uses standard RoPE.

## Conclusion
The "bug" is actually a feature gap in Ollama's converter (missing MRoPE metadata) that accidentally makes the model compatible with Ollama's current runner. The "correct" conversion by `llama.cpp` breaks compatibility because Ollama's runner doesn't support MRoPE yet.
