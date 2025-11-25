# Debugging Session Results: Qwen3-VL Split GGUF

## Summary
We investigated a persistent `0xc0000005` Access Violation crash when running Qwen3-VL split GGUF models in Ollama.

## Root Cause
The crash is caused by a mismatch in how **MRoPE (Multi-axis Rotary Position Embedding)** is handled:
1.  **Split Model**: Contains `rope.dimension_sections` metadata, enabling MRoPE in `llama.cpp`.
2.  **Ollama**: Generates **linear position IDs** (0, 1, 2...) for vision tokens.
3.  **Conflict**: `llama.cpp` tries to apply MRoPE logic using linear positions, leading to invalid memory access in the RoPE kernel.

## Attempts & Findings
1.  **Disable mmap**: Did not fix the crash.
2.  **Disable MRoPE Metadata**: Caused an assertion failure because `rope_sections` became empty.
3.  **Manual MRoPE Config**: Setting `rope_sections` to `{head_dim, 0, 0, 0}` avoided the assertion but still resulted in an Access Violation.

## Next Steps
The recommended fix is to modify the **graph construction** logic in `llama.cpp` to explicitly use standard RoPE (linear) instead of MRoPE for Qwen3-VL when running in Ollama (or when `rope_sections` are empty/defaulted). This aligns the behavior with the working non-split model.
