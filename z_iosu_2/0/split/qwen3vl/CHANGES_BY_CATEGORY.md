# Changes Summary by Category

## Category A: M-RoPE Support (Core Fix for Qwen3-VL)

These changes implement the M-RoPE position encoding required by Qwen2-VL and Qwen3-VL.

### llama/llama.go

```go
// NEW: Batch struct additions
type Batch struct {
    // ... existing fields ...
    nPosPerEmbd int        // M-RoPE: 4 positions per token
    mropePos    []C.llama_pos  // Go-managed position array
}

// NEW: Create M-RoPE batch
func NewBatchMRoPE(batchSize int, maxSeq int, embedSize int) (*Batch, error)

// NEW: Add image with 2D positions
func (b *Batch) AddImageMRoPE(embeddings []float32, pos0 int, nx int, ny int, logitsLast bool, seqIds ...int)

// NEW: Check M-RoPE mode
func (b *Batch) IsMRoPE() bool

// NEW: Vision embedding dimension
func (m *Model) NEmbdInp() int

// NEW: Check if model uses M-RoPE
func (c *MtmdContext) UsesMRoPE() bool

// MODIFIED: MtmdChunk to include grid dimensions
type MtmdChunk struct {
    Embed  []float32
    Tokens []int
    Nx int  // NEW
    Ny int  // NEW
}

// MODIFIED: MultimodalTokenize to extract grid dimensions and group embeddings
```

### runner/llamarunner/runner.go

```go
// MODIFIED: input struct
type input struct {
    token   int
    embed   []float32
    imageNx int  // NEW: M-RoPE grid width
    imageNy int  // NEW: M-RoPE grid height
}

// NEW: Token count calculation
func (inp *input) numTokens() int  // returns nx*ny for M-RoPE images

// NEW: Position advancement
func (inp *input) numPos() int     // returns max(nx,ny) for M-RoPE images

// MODIFIED: processBatch() - use M-RoPE batch and mropeBatchReady flag

// MODIFIED: loadModel() - increase batch size to 8192 for multimodal
```

### runner/llamarunner/image.go

```go
// NEW: Check M-RoPE support
func (c *ImageContext) UsesMRoPE() bool

// MODIFIED: BatchSize() - return 8192 for M-RoPE models

// MODIFIED: EmbedSize() - use NEmbdInp() for vision embeddings
```

### runner/llamarunner/cache.go

```go
// MODIFIED: LoadCacheSlot() - clear KV cache when prompt has embeddings

// NEW: inputsEqual() - conservative comparison for embeddings
```

---

## Category B: Split GGUF Support

These changes allow loading models that are split into multiple GGUF files.

### fs/ggml/ggml.go

```go
// NEW: Aggregate multiple GGUF shards
type MetaGGML struct {
    Shards     []GGML
    ShardPaths []string
    Tensors    ForeignTensors
    kv         KV
}

// NEW: Split file info
type GGUFSplitInfo struct {
    No    uint16
    Count uint16
}
func (kv KV) GGUFSplitInfo() *GGUFSplitInfo

// NEW: Track tensors across shards
type ForeignTensor struct {
    *Tensor
    ModelPath          string
    TensorRegionOffset uint64
}
type ForeignTensors []ForeignTensor

// NEW: Build MetaGGML from shards
func MakeMetaGGML(ggmls []GGML, ggmlPaths []string) MetaGGML
func BuildForeignTensors(shards []GGML, shardsPaths []string) (*ForeignTensors, error)
```

### llm/server.go

```go
// MODIFIED: LoadModel() accepts extraModels
func LoadModel(model string, extraModels []string, maxArraySize int, reliefSplitConstrain bool) (*ggml.MetaGGML, error)

// MODIFIED: NewLlamaServer() accepts extraModelPaths
func NewLlamaServer(..., extraModelPaths []string, ...) (LlamaServer, error)

// MODIFIED: StartRunner() passes extra model paths
func StartRunner(..., extraModelPaths []string, ...) (cmd *exec.Cmd, port int, err error)

// NEW: Fallback for split models
if len(extraModelPaths) > 0 {
    err = errors.New("split models not supported in new engine")
}
```

### server/images.go

```go
// MODIFIED: Model struct
type Model struct {
    // ...
    ExtraModelPaths []string  // NEW
}

// MODIFIED: GetModel() - read multiple model layers
```

### server/create.go

```go
// NEW: Validation for split GGUFs
func baseLayerSortNCheckSan(baseLayers *[]*layerGGML) error

// NEW: Broadcast KV to all shards
func broadcastKV(main *ggml.GGML, subs ...*ggml.GGML)
```

---

## Category C: Quality of Life Improvements

### runner/llamarunner/runner.go

```go
// NEW: Repetition loop detection
func (seq *Sequence) detectRepetitionLoop(token int) bool

// NEW: Signal handler for cleanup
sigChan := make(chan os.Signal, 1)
signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)

// MODIFIED: Use pflag instead of flag for --model repetition
```

### llama/llama.go

```go
// NEW: Free context resources
func (c *Context) Free()

// MODIFIED: LoadModelFromFile uses llama_model_load_from_splits
```

### runner/llamarunner/image.go

```go
// NEW: Clear cache functions
func (c *ImageContext) ClearCache()
func (c *ImageContext) ClearCacheUnsafe()

// MODIFIED: Image cache disabled for consistency
```

---

## Key Bug Fixes

### 1. Assertion Failure: `n_tokens_all <= cparams.n_batch`
**Cause**: Context created with n_batch=512, but M-RoPE images have 4000+ tokens
**Fix**: `loadModel()` increases batch to 8192 for multimodal models

### 2. Position Array Corruption
**Cause**: Using `allocSize` as stride instead of `n_tokens`
**Fix**: `AddImageMRoPE()` uses `nTokensFinal` as stride

### 3. Wrong Position Advancement
**Cause**: Advancing position by `nx*ny` instead of `max(nx,ny)`
**Fix**: `numPos()` returns `max(nx,ny)` for M-RoPE images

### 4. Batch Corruption After Image
**Cause**: Adding more tokens after M-RoPE image invalidates position layout
**Fix**: `mropeBatchReady` flag prevents additional inputs

### 5. KV Cache Reuse Issues
**Cause**: Cached KV entries don't match regenerated embeddings
**Fix**: Clear KV cache when prompt contains images
