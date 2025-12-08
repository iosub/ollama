# PR #13305 Output Reference (llama.cpp Runner)

## What Was PR #13305?

Pull request to llama.cpp that added M-RoPE (multimodal rotary position embeddings) support for Qwen2-VL and Qwen3-VL models.

**Important**: This PR used the **llama.cpp runner**, NOT the new Go runner.

## How It Worked

### Runner Architecture
- **Runner**: llama.cpp C++ executable
- **Communication**: Ollama server → llama.cpp process via API
- **Vision processing**: Done in C++ (llama.cpp/src/models/qwen3vl.cpp)

### What Happened When Processing Images

1. **Model loaded successfully** - Both text and vision weights from split GGUF
2. **Image tokenization** - Converted image to vision embeddings  
3. **Deepstack generation** - Created 3 deepstack projection layers automatically
4. **Concatenation** - Combined embeddings with correct dimensions
5. **LLM forward** - Processed with M-RoPE positions correctly
6. **Output** - **Responded correctly to image content** (extracted data, described image)

### Key Success Indicators from Logs

When it worked correctly (PR #13305 with llama.cpp runner):

```
✅ Model loaded with n_embd_inp = 16384
✅ Vision encoder ready
✅ Image tokenized to embeddings
✅ Deepstack features generated automatically
✅ Correct tensor dimensions throughout
✅ NO assertion failures
✅ Output: Correct Spanish response describing image content
```

## Why It Worked

### Critical Differences vs Current Go Runner

| Aspect | PR #13305 (llama.cpp) | Current (Go runner) |
|--------|----------------------|---------------------|
| **Vision processing** | C++ qwen3vl.cpp | Go model_vision.go |
| **Deepstack** | Automatic in C++ | Must implement manually |
| **FC weights** | Loaded automatically | NOT loading (nil) |
| **Dimensions** | Correct (16384) | Wrong (4608) |
| **Tensor views** | Correct sizes | Assertion failures |
| **Output** | ✅ Works perfectly | ❌ Crashes/hallucinates |

### What llama.cpp Did Automatically

From `llama.cpp/src/models/qwen3vl.cpp`:

1. **Detected deepstack layers**: Read `n_deepstack_layers` from GGUF metadata
2. **Loaded FC weights**: Automatically found and loaded v.deepstack.*.fc* tensors
3. **Generated features**: Called deepstack projections at correct encoder layers
4. **Concatenated**: Combined main + deepstack → 16384 dimensions
5. **Integrated with LLM**: Added features to hidden states at layers 0,1,2

**None of this required manual configuration** - llama.cpp handled it all based on GGUF metadata and model architecture.

## Expected Output Example

**User**: `C:\IA\2p.png extrae todos los datos`

**Model response** (PR #13305 with llama.cpp):
```
Aquí están los datos extraídos de la imagen:

Nombre: Juan Pérez
DNI: 12345678-A
Dirección: Calle Principal 123
Teléfono: 555-1234
...
```

**Current Go runner response**:
- Option A: Chinese hallucination text
- Option B: Empty output  
- Option C: Server crash with assertion error

## Why We Can't Just Use llama.cpp Runner

**Previous situation**: 
- llama.cpp runner worked ✅
- But had other issues (memory leaks, Windows signal handling, etc.)

**Current situation**:
- New Go runner is more stable for most models ✅
- But lacks deepstack support for split Qwen3-VL models ❌

**Goal**: 
Port the deepstack functionality from llama.cpp C++ code to Go runner so split models work correctly.

## Reference for Next Session

The PR #13305 proves that:
1. ✅ Split GGUF files are valid and complete
2. ✅ All required weights exist (including v.deepstack.*.fc*)
3. ✅ The architecture works when implemented correctly
4. ✅ llama.cpp has working reference implementation

**Therefore**: The Go runner implementation is missing/incorrect, not the model files.
