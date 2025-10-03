# Ollama Codebase Guide for AI Coding Agents

## Project Overview
Ollama is a **local LLM runtime** built in Go with C/C++ native acceleration libraries. The architecture centers around:
- **Model serving** via HTTP REST API and OpenAI-compatible endpoints
- **GPU acceleration** through CUDA, ROCm, and Metal backends 
- **Model conversion** from HuggingFace/GGUF formats to Ollama's internal format
- **Dynamic resource management** with automatic model loading/unloading

## Core Architecture Components

### 1. Server (`/server/`)
- **`routes.go`**: Main HTTP handlers for `/api/generate`, `/api/chat`, `/v1/chat/completions` endpoints
- **`sched.go`**: Critical scheduler managing model loading, GPU memory allocation, and request queuing
- **`model.go`**: Model metadata, configuration parsing, and lifecycle management
- Key pattern: All requests flow through the scheduler for resource management

### 2. Runner (`/runner/`)
- **`llamarunner/`**: Native C++ integration with llama.cpp backends
- **`ollamarunner/`**: Go wrapper for model execution and memory management
- Library detection at runtime: `./lib/ollama` (Windows), `../lib/ollama` (Linux), `.` (macOS)

### 3. Model Conversion (`/convert/`)
- **Family-specific converters**: `convert_llama.go`, `convert_gemma.go`, etc.
- **Format readers**: `reader_safetensors.go`, `reader_torch.go` for HuggingFace models
- **Tokenizer handling**: `tokenizer_spm.go` for SentencePiece, `tokenizer.go` for generic tokenizers

### 4. GPU Discovery (`/discover/`)
- **Platform-specific detection**: `gpu_windows.go`, `gpu_linux.go`, `gpu_darwin.go`
- **Vendor backends**: `cuda_common.go`, `amd_*.go` files for different GPU types
- **Runtime library loading**: Dynamic detection of CUDA/ROCm installations

## Essential Development Patterns

### Model Name Handling (`/types/model/name.go`)
```go
// Always use model.Name type for parsing and validation
name, err := model.ParseName("llama3.1:8b")
if errors.Is(err, model.ErrUnqualifiedName) {
    // Handle unqualified names (missing registry/namespace)
}
```

### Error Handling Convention
- Use `errors.Is()` and `errors.As()` for error type checking
- Wrap errors with context: `fmt.Errorf("failed to load model: %w", err)`
- API errors use `api.StatusError` with HTTP status codes

### Cross-Platform Build System
- **CMake + Go hybrid**: CMake builds native libraries, Go builds the CLI/server
- **Platform detection**: Build tags and conditional compilation in `*_windows.go`, `*_linux.go`, `*_darwin.go`
- **GPU backend selection**: Runtime detection with fallback to CPU-only mode

### Concurrent Model Management
```go
// Scheduler pattern - one model loading at a time
type Scheduler struct {
    activeLoading llm.LlamaServer  // Prevents concurrent loading
    loaded        map[string]*runnerRef  // Currently loaded models
    pendingReqCh  chan *LlmRequest  // Request queue
}
```

## Key Development Workflows

### Building from Source
```bash
# Development mode (auto-builds native libs)
go run . serve

# Full production build with GPU support
cmake -B build
cmake --build build --config Release
go build -ldflags "-X=github.com/ollama/ollama/version.Version=$VERSION"
```

### Testing Strategy
- **Unit tests**: `go test ./...` with `GOEXPERIMENT=synctest` for Go 1.24+
- **Integration tests**: Located in `/integration/` directory
- **Platform-specific testing**: Each GPU backend has dedicated test files

### Model Development
1. **Add conversion support**: Create `convert_<family>.go` in `/convert/`
2. **Update model detection**: Modify `server/model.go` for new model families
3. **Template integration**: Add chat templates in `/template/` if needed

## Important Implementation Notes

### Memory Management
- **Scheduler controls loading**: Only one model can be loaded at a time to prevent OOM
- **Dynamic unloading**: Models are evicted based on memory pressure and usage patterns
- **GPU memory tracking**: Each backend reports VRAM usage for intelligent scheduling

### API Compatibility  
- **OpenAI compatibility**: `/v1/` endpoints mirror OpenAI API format
- **Streaming responses**: All generation endpoints support server-sent events
- **Model parameter mapping**: Ollama parameters map to underlying model configs

### Configuration Management
- **Environment variables**: `OLLAMA_*` prefix for all configuration
- **Model-specific options**: Stored in `api.Options` struct, validated per model family
- **Runtime feature flags**: `OLLAMA_EXPERIMENT` enables experimental features

When modifying this codebase, always consider cross-platform compatibility, memory constraints, and the async nature of model loading. The scheduler (`sched.go`) is the heart of the system - understand its flow before making changes to model management.