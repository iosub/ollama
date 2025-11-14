# Qwen3-VL Split Investigation

## Scope of the Work Completed

- **Sampler hardening (rounds 08–11):**
	- Diagnosed the empty completions after restarts. Added the leading-EOS resampling guard in `runner/llamarunner/runner.go` so the first emitted token can never be an `eog`. 
	- Trimmed the assistant preface before streaming chunks to stop the “Assistant: ” artefact from leaking into responses.
	- Put a defensive check on KV cache shifts to avoid reusing an invalidated cache when the runner wakes up from idle.
	- Validated these changes against the historical log set (`08-.log` through `11v3-.log`) to ensure there were no regressions.

- **GGUF manifest guard:**
	- Investigated the `/api/show` panic and `/api/chat` 500s traced back to malformed string lengths in GGUF headers.
	- Added a bound check inside `fs/gguf/gguf.go:readString` (reject strings above 1<<30 or `math.MaxInt`) and reformatted the file (`gofmt`).
	- Confirmed the panic disappeared on the next server restart and that `ollama show` now handles the split manifest cleanly.

- **Split Modelfile restoration:**
	- Located the loss of parser directives in the split model build; reintroduced `RENDERER qwen3-vl-instruct` and `PARSER qwen3-vl-instruct` alongside the explicit `stop` tokens and `image_min_tokens`.
	- Rebuilt the model as `qwen3-vl-split-fixed`, verified `ollama create` completed, and tested that parser events now fire as expected in `15-split-.log`.

- **Structured extraction experiments:**
	- Ran paired tests against the non-split checkpoint (`registry.ollama.ai/library/qwen3-vl:8b-instruct-q8_0`) and the split pair (`hf.co/unsloth/...:Q4_K_M`).
	- Captured full server traces (`13-.log`, `13-split-.log`, `15-split-.log`, `15-split-14k-.log`) to compare parser output, `runner.num_ctx`, and aggregate payload sizes.
	- Confirmed caching of vision embeddings (4030 token chunks) and observed how retry prompts amplified the token estimate.

- **Context pressure deep-dive (latest session):**
	- Focused on the difference between the 14k and 32k context runs, correlating `content_len` with truncation and identifying the warm-up workaround (send “hola” first).
	- Mapped the compatibility-mode logs to the llama.cpp fallback path to explain why the prompt is flattened and emitted as a single block.

## What We Observed

1. **Compatibility mode is still enabled for split vision models.** The server reports `model not yet supported by Ollama engine, switching to compatibility mode` as soon as the split GGUF pair is loaded. This means Ollama delegates execution to the bundled llama.cpp runner because the native engine only supports non-split checkpoints today.
2. In compatibility mode (llama.cpp fallback), the initial prompt is rendered as a single block (~11k characters) plus the vision embeddings. With `num_ctx=32768` this leaves very little space for generated tokens. The model therefore stops after producing only the JSON skeleton (`content_len` ~ 189) and returns apparently empty values.
3. When the same workflow is limited to ~14k context, the model produces a full JSON object (`content_len=411`) before the parser finalises. This matches the user report: a lightweight turn ("hola") first warms up the cache, then the invoice request succeeds because the prompt stack is shorter.
4. No crashes were seen after the GGUF bounds guard. Parser input/output is consistent across runs, so the failure is not caused by parser misconfiguration.

## Current Hypothesis

The objective blocker is **context pressure inside compatibility mode**. Because the engine cannot natively run split vision checkpoints, it expands the system + schema + retry instructions into a long single prompt, adds ~4k vision tokens, and leaves almost no room for the model to answer. The immediate symptom is the empty JSON output, which appears to be truncation rather than hallucination.

## Recommended Next Steps

1. **Remove compatibility mode** by enabling native split-vision support in the Windows Vulkan build. That will stop the prompt from being flattened and should restore the full context budget.
2. While the engine is still in compatibility mode, **shorten the schema or lower `image_min_tokens`** in the Modelfile so the first turn falls well below 20k tokens.
3. Instrument the request path to log the prompt and output token counts (e.g. around `llama_eval`) so we can confirm when the model starts truncating.

## Quick Reference / Do-Not-Forget

- Compatibility mode == llama.cpp fallback; anything split will go through that path until native support lands.
- The first turn with image + schema already consumes ~32k tokens; always keep a warm-up or prompt pruning strategy handy.
- `image_min_tokens 1024` inflates the minimum image footprint to ~1M pixels—tune it down when testing long prompts.
- Parser emits valid JSON even when the model truncates; low `content_len` (<200) is the red flag to watch.
- Keep `15-split-.log` and `15-split-14k-.log` as baselines: they show the exact before/after behaviour for context pressure.

Until at least one of these mitigation steps is applied, any invoice extraction that starts directly with the full schema + image will continue to "answer in blank" because the response window is exhausted before the model can populate the fields.
