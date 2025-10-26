# Qwen3-VL Patches for llama.cpp

This directory contains the patch files used to implement Qwen3-VL and Qwen3-VL-MoE support in llama.cpp.

## Patch Files

### 1. qwen3vl_01_base_architecture.patch
**Source**: [LETS-BEE/llama.cpp@99719122b](https://github.com/LETS-BEE/llama.cpp/commit/99719122bf16db5db85f0c2d37c059a3aefd3eca)

**Description**: Base architecture implementation for Qwen3-VL models.

**Changes**:
- Added `LLM_ARCH_QWEN3_VL` and `LLM_ARCH_QWEN3_VL_MOE` architectures
- Implemented tensor mappings for both dense and MoE variants
- Added M-RoPE (Multi-axis Rotary Position Embedding) support
  - rope_sections = [24, 20, 20, 0] for spatial awareness
- Implemented load_hparams for model configuration
- Added load_tensors cases for tensor initialization
- Added MRoPE debug logging in print_info

**Key Features**:
- QWEN3_VL: Dense model (same structure as QWEN3)
- QWEN3_VL_MOE: MoE model with 4x n_embd expansion

---

### 2. qwen3vl_02_deepstack_implementation.patch
**Source**: [LETS-BEE/llama.cpp@b913e895a](https://github.com/LETS-BEE/llama.cpp/commit/b913e895a2189b9792da7709b36a36a1aed2feb9)

**Description**: DeepStack fusion architecture for Qwen3-VL-MoE.

**Changes**:
- Implemented `llm_build_qwen3vlmoe` graph building function
- Added 3-channel DeepStack fusion (ds0, ds1, ds2)
- Fusion at layers 0, 1, 2 using learnable merger layers
- Added Q/K normalization before attention
- Implemented DeepStack merger in clip.cpp:
  - Layer normalization (norm_w, norm_b)
  - Two-layer MLP (fc1_w/b, fc2_w/b)
  - GELU activation
- Windows platform support: _setmaxstdio file handle limit increase
- Enhanced batch position array handling

**DeepStack Architecture**:
```
Vision Input → Split 3 channels
  ├─ ds0 → Layer 0 fusion ─┐
  ├─ ds1 → Layer 1 fusion ─┼→ Transformer Layers → Output
  └─ ds2 → Layer 2 fusion ─┘
```

---

### 3. qwen3vl_03_memory_fix.patch
**Source**: [LETS-BEE/llama.cpp@de0e3d3c3](https://github.com/LETS-BEE/llama.cpp/commit/de0e3d3c3ce4b394746ade9263736c8edb40260e)

**Description**: Fixed illegal memory access for text-only inputs.

**Changes**:
- Removed zero-tensor initialization for text-only batches
- Fixed DeepStack fusion to only run when vision embeddings present
- Conditional fusion: `if (ubatch.embd != nullptr)`

**Problem Fixed**:
Previously, the code would create zero tensors for text inputs, causing:
- Illegal memory access attempts
- Crashes on text-only inference
- Performance issues

---

### 4. qwen3vl_04_layer_norm_bias.patch
**Source**: [LETS-BEE/llama.cpp@e45aecb7b](https://github.com/LETS-BEE/llama.cpp/commit/e45aecb7b051d3c0fea968d64aadbeb0b777e4a1)

**Description**: Updated DeepStack merger to use layer normalization with bias.

**Changes**:
- Added `norm_b` (bias) tensor to deepstack_merger
- Changed from RMS normalization to Layer normalization
- Updated `build_norm` to support bias parameter
- Proper loading of all 6 merger tensors:
  - norm_w, norm_b (normalization)
  - fc1_w, fc1_b (first linear layer)
  - fc2_w, fc2_b (second linear layer)

**Improvement**:
Layer norm with bias provides better training stability and accuracy for vision fusion.

---

## Application Order

The patches must be applied in numerical order:
1. Base architecture
2. DeepStack implementation
3. Memory fix
4. Layer norm bias

All patches have been applied to the current codebase.

## Model Support

### Qwen3-VL (Dense)
- **Layers**: 36 (4B model)
- **Architecture**: Standard transformer with vision encoder
- **Embedding**: Full n_embd dimension
- **RoPE**: M-RoPE with 4 sections

### Qwen3-VL-MoE (Mixture of Experts)
- **Layers**: 48 (30B-A3B) or 94 (235B-A22B)
- **Architecture**: MoE transformer with DeepStack fusion
- **Embedding**: n_embd/4 per channel, 4 channels total
- **Experts**: Multiple expert networks per layer
- **DeepStack**: 3-layer vision fusion at early layers

## References

- **Original Branch**: https://github.com/LETS-BEE/llama.cpp/commits/qwen3vl/
- **Discussion**: https://github.com/ggml-org/llama.cpp/issues/16207#issuecomment-3443868720
- **Upstream PR**: https://github.com/ggml-org/llama.cpp/pull/16745
- **Ollama PR**: https://github.com/ollama/ollama/pull/12665

## Usage

These patches enable running Qwen3-VL vision-language models for:
- Multimodal understanding
- Image analysis and description
- OCR (Optical Character Recognition)
- Visual question answering
- Document understanding

## Technical Notes

### M-RoPE Configuration
Multi-axis RoPE sections [24, 20, 20, 0]:
- **Section 0**: Temporal/sequence dimension (24 dims)
- **Section 1**: Image height dimension (20 dims)
- **Section 2**: Image width dimension (20 dims)
- **Section 3**: Unused (0 dims)

This configuration allows the model to maintain spatial awareness of image patches during attention computation.

### DeepStack Fusion
The fusion mechanism combines three processing channels:
1. **Early fusion** (layers 0-2): Vision features are merged using learned transformations
2. **Per-layer processing**: Each channel processes through its own expert layers
3. **Gating**: Attention mechanism selects relevant experts per token

This architecture enables efficient processing of high-resolution images while maintaining model quality.
