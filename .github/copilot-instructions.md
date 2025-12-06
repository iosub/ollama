# Ollama Copilot Instructions

Ollama is a Go-based LLM inference server that wraps llama.cpp with native GPU acceleration support.

---

## ⚠️ IMPORTANT RULES FOR THE ASSISTANT

### ❌ FORBIDDEN
1. **DO NOT compile automatically** - User compiles manually
2. **DO NOT execute build commands** without explicit permission
3. **DO NOT run `powershell -ExecutionPolicy Bypass -File ... build_windows.ps1`**

### ✅ ALLOWED
1. Prepare commands for the user to execute
2. Analyze compilation or execution logs
3. Create/modify code files, patches, documentation
4. Read files to investigate problems
5. Suggest actions, but DO NOT execute without confirmation

### 📋 TYPICAL WORKFLOW
1. User identifies problem → 2. Assistant investigates and proposes solution → 3. Assistant creates/modifies files → 4. **USER compiles** → 5. **USER runs tests** → 6. Assistant analyzes results

---

## 🎯 CURRENT PROJECT: Qwen3-VL Split Models

### Main Objective
Make **split multimodal** models (Qwen3-VL) work in Ollama. Split models come in 2 separate GGUF files (text + vision).

### Execution Architecture

```
                     ┌──────────────────────────────┐
                     │  Ollama New Engine           │
                     │  (ml/backend/ggml/)          │
                     │  ❌ Does NOT support splits  │
                     └──────────────────────────────┘
                                   │
    If qwen3vl is COMMENTED       │     If qwen3vl is UNCOMMENTED
    in fs/ggml/ggml.go            │     in fs/ggml/ggml.go
                                   │
              ┌────────────────────┴────────────────────┐
              ▼                                         ▼
┌─────────────────────────┐              ┌─────────────────────────┐
│  llama.cpp runner       │              │  New Ollama Engine      │
│  (llm/server.go)        │              │  FAILS with split models│
│  ✅ USE THIS            │              │  ❌ DO NOT USE          │
└─────────────────────────┘              └─────────────────────────┘
```

**IMPORTANT**: Keep `qwen3vl` **COMMENTED** in `fs/ggml/ggml.go` (lines ~276, ~1000) to force the llama.cpp runner path.

### Current Status (Nov 29, 2025)

#### ✅ CONFIRMED WORKING
- **llama.cpp (C++)** - Works perfectly with split models on its own
- **Split GGUF models** - Models are correct and functional
- Crashes from Exception 0xc0000005 - **FIXED**
- Image processing works (correct tokenization)
- Text generation works
- Memory cleanup with `/bye`

#### 🔴 CURRENT ISSUES (Ollama Go Code - Loading Split Models)
1. **Hallucinations**: Model generates nonsense instead of describing the image
   - Image is encoded correctly (4028 tokens for 53x76 patches)
   - Embedding dimensions are correct (16384 = n_embd_inp)
   - KV cache is cleared, image cache disabled
   - But output is repetitive garbage text
2. **Infinite loops**: Model gets stuck in repetition (fixed with loop detection)
3. **Memory leak with Ctrl+C**: Only `/bye` frees memory properly on Windows

**Root Cause Hypothesis**: The issue is in how Ollama processes mixed token/embedding batches, not in llama.cpp itself (which works standalone).

**DISCOVERED BUG (Nov 29, 2025)**: Qwen3-VL uses **M-RoPE** (Multi-dimensional Rotary Position Embedding) which requires 4 position values per image token:
- pos[0]: temporal position
- pos[1]: y position in image grid
- pos[2]: x position in image grid
- pos[3]: unused (0)

But Ollama's `llama.Batch.Add()` only sets 1 position per token (sequential: 0, 1, 2, 3...).
See `llama/llama.cpp/tools/mtmd/mtmd-helper.cpp` function `set_position_mrope_2d()` for correct implementation.

**Files to fix:**
- `llama/llama.go`: `NewBatch()` and `Add()` need M-RoPE support
- `runner/llamarunner/runner.go`: Need to pass image grid dimensions (nx, ny) to batch

### Applied Fixes for Hallucinations

**Modified files:**

1. **`runner/llamarunner/image.go`** (~line 114)
   - Disabled image cache
   - `slog.Debug("encoding image fresh (cache disabled for consistency)")`

2. **`runner/llamarunner/cache.go`** (~lines 65-108)
   - Clear KV cache when prompt contains embeddings (images)
   - `c.lc.KvCacheSeqRm(slot.Id, 0, -1)` when `hasEmbeddings && numPast > 0`

3. **`runner/llamarunner/runner.go`**
   - Repetition loop detection (`detectRepetitionLoop`)
   - Circular buffer of 256 recent tokens
   - Detects repeating patterns of 8-64 tokens

### Key Code Locations

| Problem | File | Lines |
|---------|------|-------|
| Embedding copy (Go) | `llama/llama.go` | 586-620 |
| Batch allocation | `runner/llamarunner/image.go` | 99-106 |
| KV cache | `runner/llamarunner/cache.go` | 65-108 |
| Generation loop | `runner/llamarunner/runner.go` | ~595 |
| Sampling params | `runner/llamarunner/runner.go` | ~720 |

### Embedding Dimensions (Critical)

```
Qwen3-VL-2B: n_embd=2048, n_embd_inp=8192
Qwen3-VL-8B: n_embd=4096, n_embd_inp=16384
```

**Code MUST use `n_embd_inp` (not `n_embd`) for:**
- Batch allocation
- Embedding copy operations
- Buffer sizes

### Patches Applied

| Patch | Description | File Modified |
|-------|-------------|---------------|
| `0032-fix-multimodal-embd-size-calculation.patch` | Fix embedding dimensions in C++ | `llama-context.cpp` |

### Test Logs
- Location: `z_iosu_2/logs3/q3_XX.log`
- Patches: `llama/patches/00XX-*.patch`

### Verify fixes are compiled:
```powershell
Get-Content log.txt | Select-String -Pattern "encoding image fresh|clearing KV cache|repetition loop"
```

---

## General Ollama Architecture

```
┌─────────────┐     ┌─────────────┐     ┌──────────────┐
│   CLI       │────▶│   Server    │────▶│   Scheduler  │
│  cmd/cmd.go │     │ server/     │     │  sched.go    │
└─────────────┘     └─────────────┘     └──────────────┘
                          │                    │
                          ▼                    ▼
                    ┌─────────────┐     ┌──────────────┐
                    │  API Types  │     │   Runner     │
                    │   api/      │     │  llm/ + ml/  │
                    └─────────────┘     └──────────────┘
```

### Key Components
- **`cmd/`** - CLI (Cobra). Entry: `main.go` → `cmd.NewCLI()`
- **`server/`** - HTTP server (Gin) with routes in `routes.go`
- **`api/`** - Public types (`types.go`) and client (`client.go`)
- **`llama/`** - CGO bindings to llama.cpp, with patches in `llama/patches/`
- **`runner/llamarunner/`** - Per-model inference runner
- **`envconfig/`** - Environment variables (`OLLAMA_HOST`, `OLLAMA_MODELS`)

## Build Commands

### Quick Development (No GPU)
```shell
go run . serve          # Start server
go run . run llama3.2   # Run model (separate terminal)
```

### Full Build with GPU (Windows)
```powershell
cmake -B build
cmake --build build --config Release
go run . serve
```

### Build Scripts
- Windows: `scripts/build_windows.ps1` with targets: `buildCPU`, `buildCuda13`, `buildOllama`
- Custom script: `z_iosu_2/scripts/build_windows.ps1`

## llama.cpp Vendoring

Patches in `llama/vendor/` with custom patches:

```shell
make -f Makefile.sync apply-patches  # Apply patches
make -f Makefile.sync format-patches # Generate patches from changes
make -f Makefile.sync sync           # Sync to llama/llama.cpp/
```

## Key Environment Variables
- `OLLAMA_HOST` - Server address (default: `127.0.0.1:11434`)
- `OLLAMA_MODELS` - Model path (default: `~/.ollama/models`)
- `OLLAMA_DEBUG=1` - Enable debug logs (essential for debugging)
- `OLLAMA_NEW_ENGINE` - DO NOT use with split models

## Code Conventions
- **Commits**: `<package>: <description>` (e.g., `llm/backend/mlx: support llama architecture`)
- **CGO**: Run `go clean -cache` if native structures change

## Key Files by Task

| Task | Files |
|------|-------|
| CLI command | `cmd/cmd.go` |
| API endpoint | `server/routes.go`, `api/types.go` |
| Model loading | `server/model.go`, `llm/server.go` |
| Multimodal | `runner/llamarunner/image.go`, `llama/llama.go` |
| KV Cache | `runner/llamarunner/cache.go` |
| Scheduler | `server/sched.go` |

## Split Models Project Documentation
- Investigation: `z_iosu_2/0/split/qwen3vl/doc/SPLIT_MODEL_INVESTIGATION.md`
- Applied fixes: `z_iosu_2/0/split/qwen3vl/doc/FIXES_SUMMARY.md`
- Plan: `z_iosu_2/0/split/qwen3vl/PLAN_SPLIT_MODELS.md`
- Current fix: `z_iosu_2/00Seguir/multimodal_hallucination_fix.md`
