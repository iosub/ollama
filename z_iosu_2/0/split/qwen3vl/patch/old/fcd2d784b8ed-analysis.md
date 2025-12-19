# Analysis of commit fcd2d784b8edc21f950e520c1093869666408f58

## Overview
This backup snapshot stitches together the Qwen3-VL split investigation work, runner stability fixes, and auxiliary assets used for manual testing (docs, prompts, TestModel tooling). The change count is large (1,228 insertions across 20 files) but clusters around four themes: sampler/runner hardening, GGUF manifest safety, documentation/prompts, and convenience artifacts (e.g., the `llama.go.bak` mirror).

## Notable code changes
- **Runner + server path** (`runner/llamarunner/*.go`, `llm/server.go`, `server/prompt.go`, `server/routes.go`):
  - Adds assistant-preface trimming, defers stop sequences until real text is produced, and ignores up to three leading EOG tokens to prevent blank outputs.
  - Tracks whether a model truly supports KV cache shifting (`KvCacheCanShift`) and disables the optimization when multi-position embeddings are detected.
  - Propagates `image_min_tokens` from the load request into the llama image context and increases logging for rendered prompts and streaming chunks to diagnose truncation.
- **GGUF reader guard** (`fs/gguf/gguf.go`, `api/types.go`): enforces sane string-length bounds when decoding manifests and normalizes type definitions exposed through the API to avoid `/api/show` panics.
- **Llama bindings** (`llama/llama.cpp/*`, `llama/llama.go`): surfaces new KV-cache helpers (`KvCacheSeqCp`, `KvCacheCanShift`, `GetEmbeddingsSeq`, etc.) and introduces `NEmbdInput()` so the runner can reason about projector sizes accurately. `llama.go.bak` captures the pre-change state for quick diffs during experimentation.
- **Docs and prompts** (`Z_Iosu/docs/*.md`, `Z_Iosu/prompts/*.txt`): documents the split investigation narrative (same content you summarized earlier) and adds canned prompt templates (`simple`, `vision`) for regression runs.
- **Test harness** (`TestModel/test.py`): expands the script into a CLI-driven helper that can send base64-encoded images to the local server using either curl-style HTTP or the Ollama CLI, enabling quick validation of invoice prompts.

## Impact / risks
- Runner changes alter early-token handling; any downstream component that expected the literal "assistant" prefix may need to adjust, though stop deferral reduces chances of premature termination.
- The GGUF guard rejects absurd string lengths (>1<<30). Malformed GGUFs will now fail fast instead of crashing the server, but tooling must ensure manifests stay within limits.
- `llama.go.bak` duplicates a large Go file in the repo. Keep an eye on future merges so the backup either stays aligned or is removed once no longer needed.

## Suggested follow-ups
1. Validate the new assistant-preface trimming against multilingual prompts and ensure token accounting still matches parser expectations.
2. Remove the `llama.go.bak` snapshot (or move it to an archive branch) once the investigation completes to avoid confusion.
3. Add automated regression tests that cover compatibility-mode runs with high `image_min_tokens` values to catch truncation regressions earlier.
