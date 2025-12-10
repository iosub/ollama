# Ollama Development Guide for AI Agents

## Architecture Overview

Ollama is a Go application that provides a local LLM inference server with REST API. The architecture has these key layers:

```
CLI (cmd/) → API Server (server/routes.go) → Scheduler (server/sched.go) → Runner (llm/server.go + runner/)
                                                      ↓
                                           GPU Discovery (discover/)
                                                      ↓
                                        ML Backend (ml/) ← llama.cpp bindings (llama/)
```

### Core Components

- **`server/`** - HTTP server with Gin framework, model scheduling, and image/manifest management
- **`llm/`** - LlamaServer interface bridging to native runners; platform-specific files (`llm_{darwin,linux,windows}.go`)
- **`ml/`** - Backend abstraction layer for tensor operations and device management
- **`model/models/`** - Architecture implementations (llama, gemma, qwen, deepseek, etc.) - each subfolder is a model family
- **`convert/`** - Safetensors/PyTorch to GGUF conversion (`convert_{modelname}.go` pattern)
- **`runner/`** - Native inference runners (`llamarunner/`, `ollamarunner/`)
- **`llama/`** - Vendored llama.cpp with Go bindings via CGO

### Data Flow

1. Requests hit `server/routes.go` endpoints (`/api/generate`, `/api/chat`, `/api/embed`)
2. `server/sched.go` manages model loading/unloading and GPU memory allocation
3. `server/images.go` handles model manifests and layer resolution
4. `llm/server.go` spawns native runner processes communicating via HTTP

## Build & Development

```bash
# Quick start (CPU only, uses CGO)
go run . serve

# Force rebuild native code (when CGO structures change)
go clean -cache && go run . serve

# Full build with GPU acceleration (Windows/Linux)
cmake -B build
cmake --build build --config Release
go run . serve

# ROCm (AMD GPU) on Linux
cmake -B build -G Ninja -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
cmake --build build --config Release
```

### Testing

```bash
# Unit tests
go test ./...

# Integration tests (requires compiled binary at repo root)
go build .
go test -tags=integration ./integration/...

# Integration with model tests (long running)
go test -tags=integration,models -timeout=60m ./integration/...

# Enable experimental synctest for CI parity
GOEXPERIMENT=synctest go test ./...
```

## Code Conventions

### Commit Messages
Format: `<package>: <lowercase description>` continuing "This changes Ollama to..."
```
llm/backend/mlx: support the llama architecture
server: add endpoint for model capabilities
convert: handle qwen3 architecture
```

### Adding New Model Support

1. Create converter: `convert/convert_{modelname}.go` implementing `Converter` interface
2. Create model package: `model/models/{modelname}/model.go` with `Model` struct implementing `ml.Model`
3. Register in `model/models/models.go` via blank import
4. Add architecture mapping in `fs/ggml/ggml.go` if needed

### Environment Variables (from `envconfig/`)

- `OLLAMA_HOST` - Server address (default: `127.0.0.1:11434`)
- `OLLAMA_MODELS` - Models directory (default: `~/.ollama/models`)
- `OLLAMA_KEEP_ALIVE` - Model memory duration (default: `5m`)
- `OLLAMA_DEBUG` - Enable debug logging
- `OLLAMA_ORIGINS` - Allowed CORS origins (comma-separated)

### llama.cpp Vendoring

Patches are managed via `Makefile.sync`. To update vendored code:
```bash
make -f Makefile.sync apply-patches   # Apply existing patches
# Make changes in ./vendor/
make -f Makefile.sync format-patches sync  # Generate new patches
```

## Key Interfaces

```go
// llm/server.go - Runner interface
type LlamaServer interface {
    Load(ctx context.Context, systemInfo ml.SystemInfo, gpus []ml.DeviceInfo, requireFull bool) ([]ml.DeviceID, error)
    Completion(ctx context.Context, req CompletionRequest, fn func(CompletionResponse)) error
    Embedding(ctx context.Context, input string) ([]float32, error)
}

// ml/backend.go - Tensor backend
type Backend interface {
    Load(ctx context.Context, progress func(float32)) error
    Get(name string) Tensor
    NewContext() Context
}
```

## API Patterns

- REST API follows OpenAI-compatible patterns for `/v1/` endpoints
- Native API uses `/api/` prefix with streaming JSON responses
- Model names follow `namespace/model:tag` format parsed by `types/model/`
