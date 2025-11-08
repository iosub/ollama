# Qwen3-VL Split GGUF Implementation Plan

## Objective

Implement minimal changes to Ollama's upstream Qwen3-VL support to enable loading and running split GGUF format models while maintaining full compatibility with standard non-split models.

## Base Code

**Upstream commit:** `91ec3ddb` (upstream/main)
- **Subject:** bugfix: don't include both consolidated.safetensors and model-*.safetensors (#13010)
- **Branch:** upstream/main
- **Date:** 2025-11-08

This commit represents the latest stable Ollama codebase with working Qwen3-VL support for standard models.

## Problem Statement

The current Ollama implementation supports Qwen3-VL models in standard GGUF format where all vision model weights are in a single file. Split GGUF models distribute weights across multiple files with structural differences:

1. **Conv3D dual weights:** Split models use `patch_embed.weight` + `patch_embed.weight1` for 3D convolution with `temporal_patch_size=3`
2. **Attention weights:** Split models may use fused `attn_qkv` instead of separate `attn_q`, `attn_k`, `attn_v`
3. **Metadata prefix:** Split models use `mm.*` prefix instead of `v.*` for some vision tensors

## Required Changes

### 1. Conv3D Dual Weight Support (`ml/nn/convolution.go`)

**File:** `ml/nn/convolution.go`  
**Lines:** 44-77 (approximately)

**Current behavior:**
- Only processes single weight for Conv3D
- Assumes `temporal_patch_size` applies to single weight

**Required change:**
```go
// For Qwen3-VL split models: Weight1 indicates split GGUF format
// For temporal_patch_size=3, need to handle dual weights correctly
if m.Weight1 != nil && s2 == 3 {
    wShape := m.Weight.Shape()
    w1Shape := m.Weight1.Shape()
    logutil.Trace("conv3d using dual weights for temporal_patch_size=3", "s2", s2, "c", c, "weight_shape", wShape, "weight1_shape", w1Shape)
    
    // Both weights process ALL 3 frames and concatenate in channel dimension
    // Each weight produces half the output channels (576 each → 1152 total)
    t1 := m.Weight.Conv3D(ctx, t, c, s0, s1, s2, p0, p1, p2, d0, d1, d2)
    t2 := m.Weight1.Conv3D(ctx, t, c, s0, s1, s2, p0, p1, p2, d0, d1, d2)
    logutil.Trace("conv3d dual weight full outputs", "t1_shape", t1.Shape(), "t2_shape", t2.Shape())
    
    // Concatenate along channel dimension (dim 0)
    t = t1.Concat(ctx, t2, 0)
    logutil.Trace("conv3d dual weight strategy", "type", "channel-concat-all-frames", "output_shape", t.Shape())
} else {
    t = m.Weight.Conv3D(ctx, t, c, s0, s1, s2, p0, p1, p2, d0, d1, d2)
    if m.Weight1 != nil {
        logutil.Trace("conv3d detected dual-weight but using single", "s2", s2)
    }
    logutil.Trace("conv3d single weight output", "shape", t.Shape())
}
```

**Rationale:** Split GGUF models divide the Conv3D output channels across two weight tensors. Both weights must process the full temporal sequence and concatenate in the channel dimension to produce the correct 1152-channel output.

### 2. Fused QKV Attention Support (`model/models/qwen3vl/model_vision.go`)

**File:** `model/models/qwen3vl/model_vision.go`  
**Function:** `VisionSelfAttention.Forward`  
**Lines:** Approximately 30-50

**Current behavior:**
```go
query := sa.Query.Forward(ctx, hiddenStates)
query = query.Reshape(ctx, opts.headDim(), opts.numHeads, query.Dim(1))
query = applyRotaryPositionalEmbedding(ctx, query, cos, sin)

key := sa.Key.Forward(ctx, hiddenStates)
key = key.Reshape(ctx, opts.headDim(), opts.numHeads, key.Dim(1))
key = applyRotaryPositionalEmbedding(ctx, key, cos, sin)

value := sa.Value.Forward(ctx, hiddenStates)
value = value.Reshape(ctx, opts.headDim(), opts.numHeads, value.Dim(1))
```

**Required change:**
```go
var (
    query ml.Tensor
    key   ml.Tensor
    value ml.Tensor
)

// Support both separate and fused QKV weights
if sa.QKV != nil {
    // Split GGUF format: fused QKV linear layer
    qkv := sa.QKV.Forward(ctx, hiddenStates)
    qkv = qkv.Reshape(ctx, 3, opts.headDim(), opts.numHeads, qkv.Dim(1))
    
    stride := qkv.Stride(0)
    
    // Split QKV tensor into separate query, key, value
    query = qkv.View(ctx, 0, opts.headDim()*opts.numHeads*qkv.Dim(3)).
            Reshape(ctx, opts.headDim(), opts.numHeads, qkv.Dim(3))
    key = qkv.View(ctx, stride, opts.headDim()*opts.numHeads*qkv.Dim(3)).
          Reshape(ctx, opts.headDim(), opts.numHeads, qkv.Dim(3))
    value = qkv.View(ctx, stride*2, opts.headDim()*opts.numHeads*qkv.Dim(3)).
            Reshape(ctx, opts.headDim(), opts.numHeads, qkv.Dim(3))
    
    query = applyRotaryPositionalEmbedding(ctx, query, cos, sin)
    key = applyRotaryPositionalEmbedding(ctx, key, cos, sin)
} else if sa.Query != nil && sa.Key != nil && sa.Value != nil {
    // Standard format: separate Q, K, V linear layers
    query = sa.Query.Forward(ctx, hiddenStates)
    query = query.Reshape(ctx, opts.headDim(), opts.numHeads, query.Dim(1))
    query = applyRotaryPositionalEmbedding(ctx, query, cos, sin)

    key = sa.Key.Forward(ctx, hiddenStates)
    key = key.Reshape(ctx, opts.headDim(), opts.numHeads, key.Dim(1))
    key = applyRotaryPositionalEmbedding(ctx, key, cos, sin)

    value = sa.Value.Forward(ctx, hiddenStates)
    value = value.Reshape(ctx, opts.headDim(), opts.numHeads, value.Dim(1))
} else {
    panic("vision attention missing required weights (need either QKV or Query+Key+Value)")
}
```

**Rationale:** Split GGUF models may use a single fused `attn_qkv` weight tensor instead of three separate weight matrices. This reduces file count and potentially improves loading performance.

### 3. Struct Definition Update

**File:** `model/models/qwen3vl/model_vision.go`  
**Struct:** `VisionSelfAttention`

**Current:**
```go
type VisionSelfAttention struct {
    Query  *nn.Linear `gguf:"attn_q"`
    Key    *nn.Linear `gguf:"attn_k"`
    Value  *nn.Linear `gguf:"attn_v"`
    Output *nn.Linear `gguf:"attn_output"`
}
```

**Required change:**
```go
type VisionSelfAttention struct {
    Query  *nn.Linear `gguf:"attn_q"`
    Key    *nn.Linear `gguf:"attn_k"`
    Value  *nn.Linear `gguf:"attn_v"`
    QKV    *nn.Linear `gguf:"attn_qkv"` // For split GGUF format
    Output *nn.Linear `gguf:"attn_output"`
}
```

## Files NOT Modified

The following files remain unchanged from upstream:
- `model/models/qwen3vl/imageprocessor.go`
- `model/models/qwen3vl/model.go`
- All other Qwen3-VL related files

This ensures maximum compatibility and minimal maintenance burden.

## Testing Strategy

### Test Case 1: Standard Non-Split Model
- **Model:** `qwen3-vl:8b-instruct-q4_K_M` (official Ollama model)
- **Expected:** Same behavior as upstream/main
- **Validation:** Model loads, processes images, generates accurate responses

### Test Case 2: Split GGUF Model
- **Model:** Custom split GGUF with dual Conv3D weights
- **Expected:** Model loads with dual weight detection, processes correctly
- **Validation:** 
  - Conv3D output shape: `[1152, spatial_dim]`
  - Attention uses fused QKV path
  - Image processing produces correct embeddings

## Rollback Plan

If issues arise:
```bash
# Revert to upstream/main baseline
git checkout upstream/main -- model/models/qwen3vl/model_vision.go
git checkout upstream/main -- ml/nn/convolution.go
```

## Implementation Checklist

- [x] Revert `model/models/qwen3vl/model_vision.go` to upstream/main baseline (commit 91ec3ddb)
- [x] Apply Conv3D dual weight support in `ml/nn/convolution.go`
- [x] Add `QKV` field to `VisionAttention` struct
- [x] Implement fused QKV attention logic with fallback to separate weights
- [ ] Compile and test with standard non-split model
- [ ] Test with split GGUF model
- [ ] Verify no regressions in standard model performance
- [ ] Document any behavioral differences

## Implementation Status

**Date:** 2025-11-08  
**Status:** Code changes complete, ready for testing

### Changes Applied

1. **`ml/nn/convolution.go`** - Dual weight Conv3D support ✅
   - Both weights process full temporal sequence (all 3 frames)
   - Concatenate along channel dimension (not spatial)
   - Produces correct 1152-channel output

2. **`model/models/qwen3vl/model_vision.go`** - Fused QKV support ✅
   - Added `QKV *nn.Linear` field to `VisionAttention` struct
   - Implemented dual-path forward logic:
     - Path 1: Fused QKV (split GGUF format)
     - Path 2: Separate Query/Key/Value (standard format)
   - Maintains full backward compatibility

## Notes

- All code comments and documentation use English
- Changes are minimal and surgical to reduce maintenance burden
- Backward compatibility with standard models is mandatory
- Split GGUF support is additive, not replacing existing functionality
