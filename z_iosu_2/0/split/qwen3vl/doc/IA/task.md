# Task Checklist: Qwen3-VL Split GGUF Debugging

- [x] **Initial Investigation**
    - [x] Reproduce `0xc0000005` crash with split GGUF model.
    - [x] Confirm non-split model works.
    - [x] Analyze logs: Crash happens during `llama_decode` for image embeddings.

- [x] **Hypothesis Testing**
    - [x] Test `mmap` hypothesis (Enable/Disable). Result: Not the root cause.
    - [x] Compare Converter logic (Ollama vs llama.cpp). Result: Found missing `rope.dimension_sections` in Ollama.
    - [x] Identify MRoPE mismatch as root cause.

- [ ] **Fix Implementation**
    - [x] Attempt 1: Disable `rope.dimension_sections` loading in `llama-model.cpp`. Result: Assertion failure (empty sections).
    - [x] Attempt 2: Manually set `rope_sections` to `{head_dim, 0, 0, 0}`. Result: Crash persists (Access Violation).
    - [ ] Attempt 3: Modify graph construction to use standard `ggml_rope` instead of `ggml_rope_multi` for Qwen3-VL when sections are empty.

- [ ] **Verification**
    - [ ] Compile with Attempt 3 fix.
    - [ ] Run split model with image.
    - [ ] Verify no crash and correct output.
