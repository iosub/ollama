# Plan: Split Vision Model Support in Ollama

## Current Problem

The `Qwen3-Vl-8B-Instruct` model is split across multiple GGUF files:
- Base file with the LLM
- Vision file(s) (vision encoder)

**Current error:** `split vision models aren't supported`
- Occurs in `llm/server.go` line ~177
- Ollama detects `len(projectors) > 0` and incorrectly assumes no support
- Forces compatibility mode (old llama.cpp) → causes crash

**Reality:**
- ✅ llama.cpp **DOES support** split multimodal models
- ✅ Ollama's new engine has support for `qwen3vl` in `model/models/qwen3vl/`
- ❌ Ollama is not correctly passing the file list to llama.cpp

## Split Model Architecture

### No-Split Models (working correctly)
```
model.gguf
├── Metadata (KV pairs)
├── LLM Tensors
└── Vision Encoder Tensors (integrated)
```

### Split Models (causing the error)
```
model-text.gguf           # Base file
├── Shared Metadata
├── general.architecture = "qwen3vl"
└── LLM Tensors

model-vision.gguf         # Projector/Vision encoder
├── v.* tensors (vision encoder)
└── mm.* tensors (multimodal projection)
```

**GGUF Metadata Structure:**
- `split.no` = shard number (0, 1, 2...)
- `split.count` = total number of files
- `split.tensors.count` = tensors in this shard

## How Ollama Detects Split Files - ROOT CAUSE FOUND

### Projector Detection Logic (in `server/create.go` line 744)

**The critical code that causes the problem:**

```go
func parseFromModel(layers []Layer, name string, fn func(*LayerReader) ([]Layer, error)) ([]Layer, error) {
    // ... decode GGUF file ...
    
    mediatype := "application/vnd.ollama.image.model"
    if f.KV().Kind() == "adapter" {
        mediatype = "application/vnd.ollama.image.adapter"
    } else if (f.KV().Uint("block_count") == 0 && f.KV().Uint("vision.block_count") > 0) || 
              f.KV().Kind() == "projector" {
        // ❌ BUG: This logic treats split vision files as legacy projectors
        mediatype = "application/vnd.ollama.image.projector"
    }
    // ...
}
```

**What happens with Qwen3-VL split files:**

1. **File 1:** `model-00001-of-00002.gguf`
   - `block_count` > 0 (has LLM layers)
   - `vision.block_count` = 0 (no vision in this shard)
   - `split.no` = 0, `split.count` = 2
   - **Result:** Classified as "model" ✅

2. **File 2:** `model-00002-of-00002.gguf`
   - `block_count` = 0 (no LLM layers in vision shard)
   - `vision.block_count` > 0 (has vision layers)
   - `split.no` = 1, `split.count` = 2
   - **Result:** Classified as "projector" ❌ **WRONG!**

3. In `server/images.go` line 383:
   ```go
   case "application/vnd.ollama.image.projector":
       model.ProjectorPaths = append(model.ProjectorPaths, filename)
   ```

4. In `llm/server.go` line 191:
   ```go
   if len(projectors) > 0 {
       err = errors.New("split vision models aren't supported")
   }
   ```

### Root Cause Analysis

**The bug:** Ollama's mediatype detection in `create.go` doesn't check for `split.no`/`split.count` metadata.

**Why this fails:**
- Legacy projectors: Separate GGUF file with `general.type = "projector"` (MLP, old Llava)
- Modern split models: Part of unified model, have `split.no`/`split.count` metadata

**Detection criteria comparison:**

| Type | `block_count` | `vision.block_count` | `split.count` | `general.type` |
|------|---------------|---------------------|---------------|----------------|
| **Legacy projector** | 0 | >0 | (empty) | "projector" |
| **Split vision shard** | 0 | >0 | >1 | "model" |
| **Unified model** | >0 | >0 | (empty) | "model" |

**The fix:** Check for `split.count` before classifying as projector.

## Current Code Analysis

### 1. `llm/server.go` (THE MAIN PROBLEM)
```go
// Línea ~166
func NewLlamaServer(..., projectors []string, ...) {
    var llamaModel *llama.Model
    var textProcessor model.TextProcessor
    
    if envconfig.NewEngine() || f.KV().OllamaEngineRequired() {
        if len(projectors) == 0 {
            textProcessor, err = model.NewTextProcessor(modelPath)
        } else {
            // ❌ AQUÍ ESTÁ EL ERROR
            err = errors.New("split vision models aren't supported")
        }
        if err != nil {
            // Fallback a modo compatibilidad → CRASH
            slog.Debug("switching to compatibility mode", ...)
        }
    }
    
    if textProcessor == nil {
        // Usa llama.cpp antiguo
        llamaModel, err = llama.LoadModelFromFile(modelPath, extraModelPaths, ...)
    }
    
    // ...
    
    if len(projectors) > 0 && llamaModel != nil {
        loadRequest.ProjectorPath = projectors[0]  // ⚠️ Solo pasa el primer proyector
    }
}
```

**Identified problems:**
1. Confuses "legacy separate projector" with "modern split vision model"
2. Doesn't pass all split files correctly
3. Forces compatibility mode unnecessarily for `qwen3vl`

### 2. `llama/llama.go` (EXISTING SUPPORT)
```go
func LoadModelFromFile(modelPath string, extraModelPaths []string, params ModelParams) (*Model, error) {
    // ✅ Acepta múltiples archivos via extraModelPaths
    // ✅ llama.cpp los maneja correctamente
}
```

### 3. `fs/ggml/ggml.go` (PR #13259 YA APLICADO)
```go
type MetaGGML struct {
    Shards     []GGML          // ✅ Múltiples archivos GGUF
    ShardPaths []string        // ✅ Paths de cada archivo
    Tensors    ForeignTensors  // ✅ Tensores con path y offset
    kv         KV
}

func MakeMetaGGML(ggmls []GGML, ggmlPaths []string) MetaGGML {
    // ✅ Ordena por split.no automáticamente
    // ✅ Merge de metadata
    // ✅ Crea índice unificado de tensores
}
```

## When Does This Code Run?

### Ollama Model Registration Flow

**You already have GGUF files ready to use.** This code runs when you **register** them with Ollama:

```bash
# Create a Modelfile
cat > Modelfile << EOF
FROM C:/models/Qwen3-VL-8B-00001-of-00002.gguf
FROM C:/models/Qwen3-VL-8B-00002-of-00002.gguf
EOF

# Register with Ollama
ollama create my-qwen3vl -f Modelfile
```

**What happens during `ollama create`:**

1. **Parse Modelfile** → finds two `FROM` lines
2. **For each GGUF file:**
   - Call `parseFromModel()` in `server/create.go`
   - **Read metadata** (not modify file!)
   - Assign mediatype label based on metadata
   - Store in Ollama's registry
3. **Create manifest** linking the files together
4. **Save to blob storage** (files copied, not converted)

**The bug occurs in step 2:** Second file gets wrong mediatype → wrong loading path later.

**No conversion happens** - files are used exactly as they are. We just need to fix the **classification logic**.

## Implementation Plan

### Phase 1: Fix Mediatype Detection (CRITICAL FIX)
**Goal:** Prevent split vision shards from being classified as legacy projectors

**File:** `server/create.go` line ~744

**IMPORTANT:** This is NOT a conversion process! This code runs when:
- You create a Modelfile with `FROM /path/to/model.gguf`
- Ollama imports/registers the model
- It only **classifies** the file type - doesn't modify the GGUF data

**What this code does:**
- Reads GGUF metadata
- Assigns a "mediatype" label (like a tag)
- Determines how to load the file later

**Current code (BUGGY):**
```go
} else if (f.KV().Uint("block_count") == 0 && f.KV().Uint("vision.block_count") > 0) || 
          f.KV().Kind() == "projector" {
    mediatype = "application/vnd.ollama.image.projector"
}
```

**Fixed code:**
```go
} else if f.KV().Kind() == "projector" {
    // Explicit projector type (legacy models)
    mediatype = "application/vnd.ollama.image.projector"
} else if f.KV().Uint("block_count") == 0 && f.KV().Uint("vision.block_count") > 0 {
    // Vision-only file: could be projector OR split shard
    splitCount := f.KV().Uint("split.count")
    if splitCount > 1 {
        // Part of split model - treat as extra model file, not projector
        mediatype = "application/vnd.ollama.image.model"
    } else {
        // Standalone vision model (legacy projector)
        mediatype = "application/vnd.ollama.image.projector"
    }
}
```

**This change ensures:**
- Split vision shards (with `split.count > 1`) → classified as "model"
- Will be added to `ExtraModelPaths` instead of `ProjectorPaths`
- Files passed AS-IS to llama.cpp (no conversion!)
- llama.cpp will automatically load them as part of the split model

### Phase 2: Remove Unnecessary Error Check (OPTIONAL CLEANUP)
**File:** `llm/server.go` line ~191

**Current code:**
```go
if envconfig.NewEngine() || f.KV().OllamaEngineRequired() {
    if len(projectors) == 0 {
        textProcessor, err = model.NewTextProcessor(modelPath)
    } else {
        err = errors.New("split vision models aren't supported")
    }
    // ...
}
```

**After Phase 1 fix, this error will never trigger for split models** because:
- Split vision shards won't be in `projectors[]` anymore
- They'll be in `extraModelPaths[]` instead
- Only true legacy projectors will be in `projectors[]`

**Optional enhancement:**
```go
if envconfig.NewEngine() || f.KV().OllamaEngineRequired() {
    if len(projectors) == 0 {
        textProcessor, err = model.NewTextProcessor(modelPath)
    } else {
        // This will only occur for legacy projectors now
        err = errors.New("legacy vision projectors not supported in new engine")
    }
    // ...
}
```

**Note:** Phase 1 fix alone should resolve the Qwen3-VL issue. Phase 2 is just clarifying the error message.

### Phase 3: Verify ExtraModelPaths Handling (VERIFY EXISTING CODE)
**File:** `server/images.go` lines 370-376

**Current code already handles this correctly:**
```go
switch layer.MediaType {
case "application/vnd.ollama.image.model":
    if !readMainModelFlag {
        model.ModelPath = filename
        model.ParentModel = layer.From
        readMainModelFlag = true
    } else {
        model.ExtraModelPaths = append(model.ExtraModelPaths, filename)  // ✅ Already works!
    }
```

**After Phase 1 fix:**
1. First file (split 0): `block_count > 0` → classified as "model" → becomes `ModelPath`
2. Second file (split 1): `split.count > 1` → classified as "model" → appended to `ExtraModelPaths` ✅
3. In `sched.go` line 417: 
   ```go
   llama, err = s.newServerFn(..., req.model.ModelPath, req.model.ExtraModelPaths, ...)
   ```
4. In `llm/server.go` line 186:
   ```go
   func NewLlamaServer(..., modelPath string, extraModelPaths []string, ...) {
       // extraModelPaths passed to llama.LoadModelFromFile
   }
   ```

**Conclusion:** No changes needed in Phase 3. Existing code already passes `extraModelPaths` correctly to llama.cpp!

### Phase 4: Testing Plan
**Test cases after Phase 1 fix:**

1. **No-split unified model** (current working case)
   - Single `.gguf` file with text + vision
   - `block_count > 0`, `vision.block_count > 0`, no `split.count`
   - Expected: Classified as "model", loads normally ✅

2. **Split vision model** (Qwen3-VL with separate files) **← OUR CASE**
   - File 1: `model-00001-of-00002.gguf` (text)
     - `block_count > 0`, `split.count = 2`, `split.no = 0`
     - Expected: Classified as "model" → `ModelPath` ✅
   - File 2: `model-00002-of-00002.gguf` (vision)
     - `vision.block_count > 0`, `split.count = 2`, `split.no = 1`
     - Expected: Classified as "model" → `ExtraModelPaths` ✅
   - Result: Both files passed to llama.cpp, model loads successfully ✅

3. **Legacy projector model** (old Llava)
   - File 1: `llava-text.gguf`
     - `block_count > 0`, no `split.count`
     - Expected: Classified as "model" → `ModelPath` ✅
   - File 2: `llava-mmproj.gguf`
     - `general.type = "projector"` OR (`vision.block_count > 0` AND no `split.count`)
     - Expected: Classified as "projector" → `ProjectorPaths` ✅
   - Result: Error in new engine (as expected), falls back to compatibility mode ✅

4. **Text-only split model** (multiple shards)
   - Multiple files: `model-00001-of-00004.gguf`, `model-00002-of-00004.gguf`, etc.
   - All have `block_count > 0`, `split.count = 4`
   - Expected: First as "model" → `ModelPath`, rest → `ExtraModelPaths` ✅
   - Result: llama.cpp loads all shards ✅

## Convert Information

### Ollama's `convert` package behavior

**Ollama convert (`convert/convert_qwen3vl.go`):**
```go
func ConvertModel(fsys fs.FS, f *os.File) error {
    // Reads HuggingFace model (safetensors)
    // Processes ALL tensors (text + vision)
    // Writes a SINGLE unified GGUF file
    
    conv := &qwen3VLModel{}
    ts := parseTensors(fsys, replacer)  // Gets ALL tensors
    return writeFile(f, conv.KV(t), conv.Tensors(ts))  // Single file output
}
```

**Result:** One unified `.gguf` file with both LLM and vision tensors integrated.

**Tensor name transformations in `convert_qwen3vl.go`:**
```go
func (m *qwen3VLModel) Replacements() []string {
    return []string{
        "model.language_", "",      // LLM tensors: model.language_layers.0 → blk.0
        "model.visual", "v",        // Vision: model.visual.blocks.0 → v.blk.0
        "patch_embed.proj", "patch_embed",
        "blocks", "blk",
        "attn.qkv", "attn_qkv",
        "attn.proj", "attn_out",
    }
}
```

### External split models (from HuggingFace or other sources)

**Problem:** Some Qwen3-VL distributions on HuggingFace come pre-split:
```
Qwen3-VL-8B-Instruct-Q4_K_M/
├── Qwen3-VL-8B-Instruct-Q4_K_M-00001-of-00002.gguf  # LLM tensors
└── Qwen3-VL-8B-Instruct-Q4_K_M-00002-of-00002.gguf  # Vision tensors
```

**Split GGUF structure (from external conversions):**
```python
# Base file (split 0) - created by external tools
base_file = "model-00001-of-00002.gguf"
writer = gguf.GGUFWriter(base_file)
writer.add_architecture("qwen3vl")
writer.add_uint16("split.no", 0)        # This shard number
writer.add_uint16("split.count", 2)     # Total shards
writer.add_tensor("token_embd.weight", tensor)
writer.add_tensor("blk.0.attn_q.weight", tensor)  # Only LLM tensors
# ... more LLM tensors
writer.write()

# Vision file (split 1)
vision_file = "model-00002-of-00002.gguf"
writer = gguf.GGUFWriter(vision_file)
writer.add_uint16("split.no", 1)
writer.add_uint16("split.count", 2)
writer.add_tensor("v.blk.0.attn.weight", vision_tensor)    # Vision encoder
writer.add_tensor("v.patch_embed.weight", vision_tensor)
# ... more vision tensors
writer.write()
```

**Tensor naming convention (after Ollama conversion):**
- `token_embd.*`, `blk.*.attn_*`, `output.*` → LLM (text model)
- `v.blk.*`, `v.ln.*`, `v.patch_embed.*` → Vision encoder
- `v.deepstack_merger.*` → Multimodal projection for Qwen3-VL

**Key difference:**
- **Ollama convert:** Always creates single unified file
- **External tools:** May create split files with `split.no`/`split.count` metadata
- **Our issue:** Ollama doesn't handle externally-split multimodal models correctly

### How llama.cpp handles split files

**From `llama-model-loader.cpp` lines 515-565:**

```cpp
// Load main file first
uint16_t n_split = 0;
get_key(llm_kv(LLM_KV_SPLIT_COUNT), n_split, false);

if (n_split > 1) {
    // Verify main file is split 0
    uint16_t idx = 0;
    get_key(llm_kv(LLM_KV_SPLIT_NO), idx);
    if (idx != 0) {
        throw std::runtime_error("model must be loaded with the first split");
    }
    
    // Generate split file list: model-00001-of-00002.gguf → model-00002-of-00002.gguf
    splits = llama_get_list_splits(fname, idx, n_split);
    
    // Load each additional split
    for (idx = 1; idx < n_split; idx++) {
        gguf_context_ptr ctx_gguf = gguf_init_from_file(splits[idx]);
        // Merge tensors into weights_map
        // ...
    }
}
```

**Key metadata keys (from `llama-arch.cpp` line 206-208):**
```cpp
{ LLM_KV_SPLIT_NO,            "split.no"            },  // This file's index (0-based)
{ LLM_KV_SPLIT_COUNT,         "split.count"         },  // Total number of files
{ LLM_KV_SPLIT_TENSORS_COUNT, "split.tensors.count" }, // Tensors in this file
```

**llama.cpp split loading algorithm:**
1. Read primary file (must have `split.no = 0`)
2. Parse `split.count` to know total files
3. Auto-generate remaining filenames based on naming pattern
4. Load each split sequentially
5. Merge all tensors into unified weight map
6. Proceed with model initialization

**Critical insight:** llama.cpp **automatically finds and loads** additional splits if:
- Primary file has `split.count > 1`
- Files follow naming convention: `name-00001-of-NNNNN.gguf`
- OR files are explicitly provided via extra model paths

## Next Steps - SIMPLIFIED PLAN

### Priority 1: Critical Fix (Phase 1)
1. [ ] **Modify `server/create.go` line ~744** to check `split.count` before classifying as projector
   - File: `c:\IA\tools\ollama\server\create.go`
   - Lines to change: 744-747
   - Expected impact: Split vision files will go to `ExtraModelPaths` instead of `ProjectorPaths`
   - This single change should fix the Qwen3-VL loading issue

### Priority 2: Optional Improvements
2. [ ] **Update error message in `llm/server.go` line ~194** (optional)
   - Clarify error message: "legacy vision projectors not supported in new engine"
   - No functional impact, just better error reporting

### Priority 3: Testing & Validation
3. [ ] Test with Qwen3-VL split model files
4. [ ] Verify legacy projector models still show appropriate error
5. [ ] Verify unified models continue working
6. [ ] Test text-only split models (if available)

### Summary
**The fix is simpler than originally thought:**
- Only Phase 1 needs implementation
- No changes to `llm/server.go` logic required
- No changes to new engine required
- Existing `ExtraModelPaths` handling already works correctly
- llama.cpp already has full split model support

## References

- PR #13259: Multi-file GGUF support (ya aplicado)
- `llama/llama.cpp/convert_hf_to_gguf.py`: Script de conversión original
- `fs/ggml/ggml.go`: Implementación de MetaGGML
- `llm/server.go`: Entry point que necesita modificación
