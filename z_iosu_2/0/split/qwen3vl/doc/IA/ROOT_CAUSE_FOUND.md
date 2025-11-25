# Root Cause Analysis: Qwen3-VL Split GGUF Crash

## Symptom
- **Error**: `0xc0000005` Access Violation (Segmentation Fault).
- **Location**: `llama_decode` function in `llama.cpp` (called via CGO).
- **Trigger**: Occurs specifically during the **second** `llama_decode` call which processes **image embeddings** (vision tokens). The first call (text prompt) works fine.
- **Context**: Running Qwen3-VL-8B Instruct in **split GGUF** format on Ollama (Windows).

## Investigation Findings

### 1. `mmap` is NOT the primary culprit (but related)
- Initially suspected `mmap` being disabled was the cause.
- Enabling `mmap` for vision models (via `server.go` patch) did **not** fix the crash.
- Disabling `mmap` (via `server.go` patch) confirmed `mmap=false` in logs, but the crash persisted at the exact same point.

### 2. Crash Location: Image Embedding Processing
- Debug logs confirmed the crash happens when `embd_ptr` is populated (image embeddings).
- `n_tokens` = 512 (typical for vision batch).
- `token_ptr` = NULL (correct for embedding batch).
- `embd_ptr` = Valid pointer (0x...).
- **Crash happens inside `C.llama_decode`**.

### 3. The "Split" Factor
- **Non-split model works**: The single-file GGUF converted by Ollama works perfectly.
- **Split model fails**: The multi-file GGUF converted by `llama.cpp` (or unsloth) fails.
- **Key Difference**: The split model has `rope.dimension_sections` metadata (MRoPE), while the non-split model does not.

### 4. MRoPE (Multi-axis RoPE) Mismatch
- **Qwen3-VL uses MRoPE**: Positions are not just linear indices (0, 1, 2...); they are 3D coordinates (time, height, width).
- **Ollama's Behavior**: Ollama constructs the `llama_batch` for vision tokens using **linear position IDs** (0, 1, 2...).
- **Llama.cpp's Behavior**:
    - If `rope.dimension_sections` is present (Split model), `llama.cpp` attempts to apply MRoPE.
    - It tries to map the linear positions provided by Ollama to the MRoPE grid.
    - Since Ollama provides simple integers, and MRoPE expects encoded coordinates (or compatible handling), the RoPE calculation fails (likely accessing out-of-bounds memory in the precomputed RoPE tables).
    - **Result**: `0xc0000005` Access Violation.
    - If `rope.dimension_sections` is MISSING (Non-split model), `llama.cpp` falls back to standard RoPE (or no RoPE for vision), which works fine with linear positions.

## Conclusion
The crash is caused by an **incompatibility between Ollama's position ID generation for vision tokens and `llama.cpp`'s MRoPE implementation** when the model metadata explicitly enables MRoPE.

## Solution Strategy
1.  **Ideal Fix**: Update Ollama to support MRoPE position generation for Qwen3-VL (complex).
2.  **Workaround (Chosen)**: Patch `llama.cpp` to **ignore or disable MRoPE** for this specific architecture when running in Ollama. This forces the split model to behave like the non-split model (using standard RoPE), which is compatible with Ollama's linear positions.
