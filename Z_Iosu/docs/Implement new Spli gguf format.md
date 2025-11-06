# Implementing Support for the “Spli” GGUF Format

> Working notes for enabling the new GGUF variant shipped with hf.co/unsloth/Qwen3-VL-8B-Instruct-GGUF:Q4_K_M inside the Ollama engine.

---

## 1. High-Level Goals

- Load Qwen3-VL “Spli” GGUF through the **Ollama runner** (new engine) instead of falling back to legacy llama.cpp runner.
- Preserve full multimodal capabilities (vision tokenizer, deepstack embeddings, MoE routers) and structured-output constraints.
- Maintain Windows + Vulkan compatibility and avoid regressions for existing Qwen3-VL builds.

---

## 2. Immediate Diagnostics

1. **Force the new engine**
   - Run with `OLLAMA_NEW_ENGINE=1` to capture the exact failure reason.
   - Command from `dist/windows-amd64`:
     ```powershell
     $env:OLLAMA_NEW_ENGINE='1'
     .\ollama.exe run hf.co/unsloth/Qwen3-VL-8B-Instruct-GGUF:Q4_K_M <<< '/bye'
     ```
   - Collect log entries around `model not yet supported by Ollama engine`.

2. **Dump GGUF metadata**
   - Use the Python gguf tooling on `%USERPROFILE%\.ollama\models\blobs\sha256-108e7ff9…`.
   - Capture both `kv` entries and tensor names to compare against existing `model/models/qwen3vl` expectations.

### Diagnostic Findings (Nov 5, 2025)

- 8B UnsloTH `sha256-108e7ff9…`
   - `GGUF.version = 3`, `general.file_type = 15`, `qwen3vl.n_deepstack_layers = 3`, rope sections `[24, 20, 20, 0]`.
   - Failure: `model not yet supported by Ollama engine, switching to compatibility mode` (`split vision models aren't supported`).
   - Artifacts recorded at `Z_Iosu/docs/sha256-108e7ff9.kv.json` and `Z_Iosu/docs/sha256-108e7ff9.tensors.txt`.
- 2B GGML `sha256-b7802e29…` (hf.co/ggml-org/Qwen3-VL-2B-Instruct-GGUF:Q8_0)
   - `GGUF.version = 3`, `general.file_type = 7`, retains `qwen3vl.n_deepstack_layers = 3` and identical rope sections.
   - Same compatibility fallback with `split vision models aren't supported`.
   - Artifacts recorded at `Z_Iosu/docs/sha256-b7802e29.kv.json` and `Z_Iosu/docs/sha256-b7802e29.tensors.txt`.
- Projector blobs (new "Spli" split vision component)
   - 2B projector `sha256-69066c8f…`: `general.type = mmproj`, `general.architecture = clip`, `clip.projector_type = qwen3vl_merger`, `clip.vision.*` keys supply hidden size 1024, block count 24, deepstack flags, spatial merge size 2, etc. (`Z_Iosu/docs/sha256-69066c8f.kv.json`).
   - 8B projector `sha256-d406d03e…`: similar schema with embedding length 1152, projection_dim 4096, deepstack flags at layers 8/16/24, etc. (`Z_Iosu/docs/sha256-d406d03e.kv.json`).
   - Tensor dumps (`*.tensors.txt`) confirm projector contains the vision encoder/merger weights that the base model blobs no longer embed.

### Findings (2025-11-05)

- Engine fails with `split vision models aren't supported`, confirming loader rejects the unsloth layout before model startup.
- Metadata snapshot stored at:
   - `Z_Iosu/docs/sha256-108e7ff9.kv.json`
   - `Z_Iosu/docs/sha256-108e7ff9.tensors.txt`
- Notable key/value highlights compared to upstream GGUF:
   - `general.file_type = 15` (Spli split tensor format), `general.quantization_version = 2`.
   - Added `qwen3vl.n_deepstack_layers = 3` and rope sections `[24, 20, 20, 0]`.
   - Tokenizer preserves chat template but `tokenizer.ggml.merges` now lists ~150k entries (large array confirmed in dump).
   - Quantization metadata references UnsloTH datasets (`quantize.imatrix.*`).
- Tensor list shows `399` entries; verify whether projector/vision tensors carry new prefixes once cross-checked with stock loader.

---

## 3. Expected Gaps

| Area | Current State | Likely Work |
|------|----------------|-------------|
| **Tokenizer/TextProcessor** | `model.NewTextProcessor` fails → fallback to llama runner | Extend `model/models/qwen3vl` to parse UnsloTH-specific metadata (deepstack dims, router config, projector tensors). |
| **Convert Path** | `convert_qwen3*.go` handles official GGUFs only | Either teach the converter the new keys or ensure loader gracefully accepts third-party GGUFs without converter involvement. |
| **Runner Multimodal Glue** | `runner/ollamarunner` assumes existing vision token layout | Validate deepstack embeddings count / shapes; adjust `PostTokenize` and `Forward` if “Spli” introduces new positions. |
| **MoE Configuration** | Sparse FFN already supported for `qwen3vlmoe` | Confirm UnsloTH naming matches (e.g., `norm_topk_prob`, `expert_count`). Add coverage tests if not. |

---

## 4. Implementation Plan (Draft)

1. **Metadata Mapping**
   - Extend `TextOptions` and `VisionModel` constructors to read new gguf keys (e.g., projector dimensions, deepstack counts).
   - Guard with defaults so existing official models remain unaffected.

2. **Text Processor Wiring**
   - Ensure `model.NewTextProcessor` can instantiate the tokenizer even if auxiliary tensors (e.g., sentencepiece merges) use different prefixes.
   - Add a unit test that loads the tokenizer portion of the UnsloTH blob.

3. **Forward Path Adjustments**
   - Verify `PostTokenize` handles the UnsloTH placeholder sequence (tokenVision/Start/End). Update `positionCache` logic if new offsets are required.
   - Validate `Forward` handles additional deepstack embeddings; update the loop or merging logic if the counts differ.

4. **Runner Validation**
   - Re-run with `OLLAMA_NEW_ENGINE=1` and confirm the Ollama runner stays active.
   - Exercise both `TestModel/test.py` and CLI prompts to ensure deterministic JSON output is maintained.

5. **Regression Coverage**
   - Add conversion/loader tests that compare expected tensor shapes for both official Qwen3-VL and UnsloTH “Spli”.
   - Document any new environment variables or options.

---

## 5. Open Questions

- Does UnsloTH introduce renamed tensors (e.g., `deepstack_visual_embeds.*`) that the backend ignores today?
- Are there GGUF `kv` fields signalling compulsory use of the new engine (e.g., `general.architecture = qwen3vlmoe`)?
- Any additional projector/vision layers not covered by existing Vulkan kernels?

---

## 6. Next Actions

1. ✅ Run diagnostics (`OLLAMA_NEW_ENGINE=1`, gguf dump) and archive results.
2. Diff the metadata against `model/models/qwen3vl/*` to identify missing keys.
   - ✅ Base model blobs expose only `general.*` and `qwen3vl.*` entries—no `vision.*` keys—so the current loader relies on defaults and never hydrates `VisionModel`.
   - ✅ Vision parameters now live in the companion projector GGUF under `clip.vision.*` and `clip.projector_type` instead of the embedded `v.*` hierarchy expected by `VisionModel`.
   - TODO Map projector metadata to `newVisionModel`/`newImageProcessor` fields (head count, patch size, deepstack indexes, mean/std, spatial merge size, projection dims).
3. Draft loader patches and add targeted tests before touching the runner proper.

---

_This document tracks progress while we experiment; update sections as we discover new requirements or blockers._
