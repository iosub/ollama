# PR #13278 - Documentation Index

**PR:** https://github.com/ollama/ollama/pull/13278  
**Feature:** M-RoPE and Split Model Support for Qwen3-VL  
**Date:** November 30, 2025

---

## Main Documents

| Document | Description |
|----------|-------------|
| [DESIGN_RATIONALE.md](DESIGN_RATIONALE.md) | **Technical design decisions** - Options considered, pros/cons, rationale for each choice. Best document for reviewers. |
| [PR_MROPE_SPLIT_MODELS.md](PR_MROPE_SPLIT_MODELS.md) | **Full PR description** - Complete list of changes, bug fixes, files modified. |
| [CHANGES_BY_CATEGORY.md](CHANGES_BY_CATEGORY.md) | **Code changes by category** - Organized view of all code modifications (M-RoPE, Split GGUF, QoL). |

---

## Reference Documents

| Document | Description |
|----------|-------------|
| [doc/FIXES_SUMMARY.md](doc/FIXES_SUMMARY.md) | Bug fix history and solutions applied |
| [doc/SPLIT_MODEL_INVESTIGATION.md](doc/SPLIT_MODEL_INVESTIGATION.md) | Initial investigation notes on split models |
| [doc/PR_13278_COPILOT_REVIEW.md](doc/PR_13278_COPILOT_REVIEW.md) | GitHub Copilot review comments and resolutions |

---

## Patches

| Patch | Description |
|-------|-------------|
| [patch/0032-fix-multimodal-embd-size-calculation.patch](patch/0032-fix-multimodal-embd-size-calculation.patch) | C++ fix for n_embd vs n_embd_inp in llama-context.cpp |

---

## External References

- [PR #12992](https://github.com/ollama/ollama/pull/12992) - Base PR (dhiltgen/ggml_bump) with qwen3vl architecture
- [PR #13278](https://github.com/ollama/ollama/pull/13278) - This implementation
- [llama.cpp mtmd-helper.cpp](https://github.com/ggerganov/llama.cpp/blob/master/tools/mtmd/mtmd-helper.cpp) - M-RoPE reference implementation
- [Qwen2-VL Paper](https://arxiv.org/abs/2409.12191) - M-RoPE description

---

## Quick Summary

**Problem:** Qwen3-VL models hallucinate with images because Ollama uses 1 position per token, but M-RoPE requires 4 positions (temporal, y, x, unused).

**Solution:** 
1. New batch functions for M-RoPE (`NewBatchMRoPE`, `AddImageMRoPE`)
2. Split GGUF support (`MetaGGML`, `ForeignTensors`)
3. Fallback to llama.cpp runner for split models

**Status:** PR #13278 submitted, Copilot review comments addressed.
