# Split GGUF Enablement Plan (v3)

## Scope
- Target models: Qwen3-VL split GGUF checkpoints (2B, 8B) distributed as paired base+projector blobs.
- Platforms: Windows (CPU/Vulkan) with the Go-based Ollama runner.
- Non-split GGUF models must remain unaffected (no regression in tokenizer, loading speed, or VRAM heuristics).

## Step 1 – Allow the Ollama runner to claim split models
### Tasks
1. Remove the hard-coded `errors.New("split vision models aren't supported")` guard in `llm/server.go` and gate the legacy llama.cpp fallback behind a real capability probe (e.g., only fallback if projector tensors cannot be loaded).
2. Ensure `model.NewTextProcessor` is invoked even when `ProjectorPaths` is non-empty so the tokenizer metadata keeps being parsed before we decide which runner to spawn.
3. Treat `projectors` as a first-class input in `NewLlamaServer` by passing the projector list down into the runner request instead of discarding it.
### Acceptance Criteria
- Bringing up `ollama serve` with `hf.co/ggml-org/Qwen3-VL-2B-Instruct-GGUF:Q8_0` never logs “switching to compatibility mode”.
- Non-split models (e.g., `qwen3-vl:8b-instruct-q4_K_M`) follow the exact same code path as before (validated by diffing INFO logs and ensuring runner args stay identical).

## Step 2 – Dual-backend tensor resolution in the model layer
### Tasks
1. Extend `model.Base` to optionally hold a secondary `ml.Backend` created from the projector GGUF; introduce a helper (e.g., `func (m *Base) GetTensor(name string) ml.Tensor`) that queries the main backend first and transparently falls back to the projector backend.
2. Teach `model.New`/`model.NewTextProcessor` to instantiate the projector backend when `ProjectorPaths` is supplied, wiring its lifetime to the model instance (free both backends together).
3. Update `model/models/qwen3vl` to call `m.GetTensor` everywhere we currently hard-wire `Backend().Get`, so `PatchEmbedding`, `VisionPatchMerger`, deepstack mergers, and any optional tensors can come from either file without additional plumbing.
### Acceptance Criteria
- `ensureVisionReady` (or equivalent constructor) locates projector-only tensors such as `v.patch_embed.weight` and `deepstack_merger.*` without panics.
- Loading a non-split checkpoint performs only one backend allocation and shows identical peak RSS.

## Step 3 – Integrate the split-aware vision pipeline
### Tasks
1. Fold the proven changes from `z_iosu_2/tmp/model_vision_working.go` into the real `model/models/qwen3vl/model_vision.go`: fused QKV handling, deferred padding, optional position embeddings, and merger guards for missing weights.
2. Revisit Conv3D dual-weight support (`weight` + `weight.1`) so that temporal_patch_size=3 pipelines match the llama.cpp layout; unit-test by comparing tensor shapes before/after the patch embedding stage (expect `[1152, 5400]`).
3. Wire deepstack outputs and `PostTokenize` adjustments so that multimodal inserts keep their intended positional encodings even when the projector reshapes the grid.
### Acceptance Criteria
- Vision forward pass logs show consistent hidden sizes (no `[384 5400]` intermediate) and match the non-split baseline after padding.
- Multimodal caching works across consecutive requests (no nil-pointer panics inside `VisionPositionEmbedding`).

## Step 4 – Validation & regression coverage
### Tasks
1. Add targeted tests: (a) loader test that constructs a `Model` with both GGUF files and asserts key tensors exist; (b) integration test that tokenizes+generates a short prompt with an image placeholder to ensure `PostTokenize` emits `<|vision_start|>`/`<|vision_end|>` correctly.
2. Capture new reference logs (`OLLAMA_DEBUG=2`) for split vs. non-split runs and document any VRAM/context deltas.
3. Update troubleshooting docs to mention the new requirement (both blobs must live in `.ollama/models/blobs/` and share the `ProjectorPaths` manifest entry).
### Acceptance Criteria
- CI: new tests pass on Windows and Linux runners.
- Manual run of `ollama run hf.co/ggml-org/Qwen3-VL-2B-Instruct-GGUF:Q8_0` returns non-empty JSON completions without prior warm-up prompts.

## Timeline & Ownership
- Step 1: 0.5 day (Server/runner team).
- Step 2: 1 day (Model infra team).
- Step 3: 1.5 days (Vision/modeling team) – depends on Step 2.
- Step 4: 0.5 day shared across QA + docs.

## Risks & Mitigations
- **GPU-specific regressions**: dual backend increases memory footprint. Mitigate by reusing tensor handles and freeing projector backend when unused.
- **Parsing drift**: third-party projector blobs may omit optional tensors. Guard every access (nil-aware layers) and emit WARN logs instead of panicking.
- **Context pressure remains**: if the new engine still overflows at 32k tokens, be ready to tune template/schema defaults once the compatibility fallback is gone.
