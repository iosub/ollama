package qwen3vl

import (
	"fmt"
	"iter"
	"log/slog"
	"math"
	"slices"

	"github.com/ollama/ollama/fs"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn"
)

type VisionAttention struct {
	Query  *nn.Linear `gguf:"attn_q"`   // Unified model: separate Q
	Key    *nn.Linear `gguf:"attn_k"`   // Unified model: separate K
	Value  *nn.Linear `gguf:"attn_v"`   // Unified model: separate V
	QKV    *nn.Linear `gguf:"attn_qkv"` // Split model: combined QKV
	Output *nn.Linear `gguf:"attn_out"`
}

func rotateHalf(ctx ml.Context, t ml.Tensor) ml.Tensor {
	x1 := t.Slice(ctx, 0, 0, t.Dim(0)/2, 1)
	x2 := t.Slice(ctx, 0, t.Dim(0)/2, t.Dim(0), 1).Contiguous(ctx)
	return x2.Scale(ctx, -1).Concat(ctx, x1, 0)
}

func applyRotaryPositionalEmbedding(ctx ml.Context, t, cos, sin ml.Tensor) ml.Tensor {
	return t.Mul(ctx, cos).Add(ctx, rotateHalf(ctx, t).Mul(ctx, sin))
}

func (sa *VisionAttention) Forward(ctx ml.Context, hiddenStates, cos, sin ml.Tensor, opts VisionOptions) ml.Tensor {
	var query, key, value ml.Tensor

	if sa.QKV != nil {
		// Split model: combined QKV tensor - split into Q, K, V
		qkv := sa.QKV.Forward(ctx, hiddenStates)
		// qkv shape is [hiddenSize*3, seqLen] - split along first dimension
		hiddenSize := opts.hiddenSize
		seqLen := qkv.Dim(1)
		query = qkv.Slice(ctx, 0, 0, hiddenSize, 1)
		key = qkv.Slice(ctx, 0, hiddenSize, hiddenSize*2, 1)
		value = qkv.Slice(ctx, 0, hiddenSize*2, hiddenSize*3, 1)
		// Use Contiguous(ctx, shape...) to avoid view_src chain - this calls ggml_cont_Nd
		// which creates a truly independent tensor without view_src issues
		query = query.Contiguous(ctx, opts.headDim(), opts.numHeads, seqLen)
		key = key.Contiguous(ctx, opts.headDim(), opts.numHeads, seqLen)
		value = value.Contiguous(ctx, opts.headDim(), opts.numHeads, seqLen)
	} else {
		// Unified model: separate Q, K, V tensors - use normal Reshape
		query = sa.Query.Forward(ctx, hiddenStates)
		key = sa.Key.Forward(ctx, hiddenStates)
		value = sa.Value.Forward(ctx, hiddenStates)
		query = query.Reshape(ctx, opts.headDim(), opts.numHeads, query.Dim(1))
		key = key.Reshape(ctx, opts.headDim(), opts.numHeads, key.Dim(1))
		value = value.Reshape(ctx, opts.headDim(), opts.numHeads, value.Dim(1))
	}

	query = applyRotaryPositionalEmbedding(ctx, query, cos, sin)
	key = applyRotaryPositionalEmbedding(ctx, key, cos, sin)

	attention := nn.Attention(ctx, query, key, value, math.Pow(float64(opts.headDim()), -0.5), nil)
	attention = attention.Reshape(ctx, opts.hiddenSize, attention.Dim(2))
	return sa.Output.Forward(ctx, attention)
}

type VisionMLP struct {
	FC1 *nn.Linear `gguf:"linear_fc1"`
	FC2 *nn.Linear `gguf:"linear_fc2"`
}

func (mlp *VisionMLP) Forward(ctx ml.Context, hiddenStates ml.Tensor, opts VisionOptions) ml.Tensor {
	if mlp.FC1 == nil || mlp.FC1.Weight == nil {
		panic("VisionMLP.FC1 is nil - alias 'v.blk.*.mlp.linear_fc1' not found. Check if ffn_up→linear_fc1 alias was created.")
	}
	if mlp.FC2 == nil || mlp.FC2.Weight == nil {
		panic("VisionMLP.FC2 is nil - alias 'v.blk.*.mlp.linear_fc2' not found. Check if ffn_down→linear_fc2 alias was created.")
	}

	fc1Out := mlp.FC1.Forward(ctx, hiddenStates)
	activated := fc1Out.GELU(ctx)
	return mlp.FC2.Forward(ctx, activated)
}

type VisionEncoderLayer struct {
	Norm1     *nn.LayerNorm `gguf:"norm1"`
	Attention *VisionAttention
	Norm2     *nn.LayerNorm `gguf:"norm2"`
	MLP       *VisionMLP    `gguf:"mlp"`
}

func (e *VisionEncoderLayer) Forward(ctx ml.Context, hiddenStates, cos, sin ml.Tensor, opts VisionOptions) ml.Tensor {
	residual := hiddenStates
	hiddenStates = e.Norm1.Forward(ctx, hiddenStates, opts.eps)
	hiddenStates = e.Attention.Forward(ctx, hiddenStates, cos, sin, opts)
	hiddenStates = hiddenStates.Add(ctx, residual)

	residual = hiddenStates
	hiddenStates = e.Norm2.Forward(ctx, hiddenStates, opts.eps)
	hiddenStates = e.MLP.Forward(ctx, hiddenStates, opts)
	return hiddenStates.Add(ctx, residual)
}

type VisionOptions struct {
	hiddenSize,
	numHeads,
	patchSize,
	numChannels,
	spatialMergeSize,
	temporalPatchSize,
	gridPerSide int

	eps,
	ropeTheta float32

	// isSplitArchitecture indicates split model variant uses different patch embedding
	// Split: [16,16,3,1152] per-channel 2D conv with multiple weight tensors
	// Unified: [16,16,2,3456] 3D conv with merged channels
	isSplitArchitecture bool

	deepstackVisualIndexes []int32
	mropeSections          []int
}

func (o VisionOptions) headDim() int {
	return o.hiddenSize / o.numHeads
}

type VisionPatchMerger struct {
	Norm *nn.LayerNorm `gguf:"norm"`
	FC1  *nn.Linear    `gguf:"linear_fc1"`
	FC2  *nn.Linear    `gguf:"linear_fc2"`
}

func (m *VisionPatchMerger) Forward(ctx ml.Context, visionOutputs ml.Tensor, postshuffleNorm bool, opts VisionOptions) ml.Tensor {
	hiddenSize := opts.hiddenSize * opts.spatialMergeSize * opts.spatialMergeSize
	if postshuffleNorm {
		visionOutputs = visionOutputs.Reshape(ctx, hiddenSize, -1)
	}

	// Norm is required
	if m.Norm != nil {
		visionOutputs = m.Norm.Forward(ctx, visionOutputs, opts.eps)
	}
	visionOutputs = visionOutputs.Reshape(ctx, hiddenSize, -1)

	// FC1/FC2 may be nil for split models (they use separate mm.* projector)
	if m.FC1 != nil && m.FC2 != nil {
		return m.FC2.Forward(ctx, m.FC1.Forward(ctx, visionOutputs).GELU(ctx))
	}

	// Split model: just return normalized output, projection happens elsewhere
	return visionOutputs
}

type VisionPositionEmbedding struct {
	PositionEmbedding *nn.Embedding `gguf:"position_embed"` // Aliased from position_embd for split models
}

func makeSlice2D[T int32 | float32](n0, n1 int) iter.Seq[[]T] {
	return func(yield func([]T) bool) {
		for range n0 {
			if !yield(make([]T, n1)) {
				return
			}
		}
	}
}

func (m *VisionPositionEmbedding) Forward(ctx ml.Context, hiddenStates ml.Tensor, grid *Grid, opts VisionOptions) ml.Tensor {
	// Unified models don't have explicit position embedding tensor - they use RoPE in attention layers
	// Split models (unsloth) have v.position_embd tensor that needs to be looked up
	if m == nil || m.PositionEmbedding == nil || m.PositionEmbedding.Weight == nil {
		// No position embedding tensor - return unchanged (unified model uses RoPE in positions())
		return hiddenStates
	}

	// Use nearest-neighbor position embedding lookup for split models
	// Avoids CPU/GPU Mul issue by using only Rows (handles CPU indices -> GPU result)
	stepHeight := float32(opts.gridPerSide-1) / float32(grid.Height-1)
	stepWidth := float32(opts.gridPerSide-1) / float32(grid.Width-1)

	// Compute nearest position indices (round instead of bilinear interpolation)
	indices := make([]int32, grid.Height*grid.Width)

	i := 0
	for h := range grid.Height {
		for w := range grid.Width {
			// Use nearest neighbor (round) instead of bilinear interpolation
			y := int32(float32(h)*stepHeight + 0.5)
			x := int32(float32(w)*stepWidth + 0.5)

			// Clamp to valid range
			if y >= int32(opts.gridPerSide) {
				y = int32(opts.gridPerSide) - 1
			}
			if x >= int32(opts.gridPerSide) {
				x = int32(opts.gridPerSide) - 1
			}

			indices[i] = y*int32(opts.gridPerSide) + x
			i++
		}
	}

	n := hiddenStates.Dim(0) // hidden size

	// Use Rows to look up embeddings - this handles CPU indices -> GPU result properly
	idx := ctx.Input().FromInts(indices, grid.Height*grid.Width)
	positionEmbeds := m.PositionEmbedding.Weight.Rows(ctx, idx)
	// Use Contiguous(ctx, shape...) to avoid view_src chain - this calls ggml_cont_Nd
	// which creates a truly independent tensor without view_src issues
	positionEmbeds = positionEmbeds.Contiguous(ctx, -1, grid.Width/opts.spatialMergeSize, opts.spatialMergeSize, grid.Height/opts.spatialMergeSize)
	positionEmbeds = positionEmbeds.Permute(ctx, 0, 2, 1, 3).Contiguous(ctx, n, -1)
	return hiddenStates.Add(ctx, positionEmbeds)
}

type VisionModel struct {
	PatchEmbedding    *nn.Conv3D `gguf:"patch_embed"` // Unified model uses 3D conv
	PatchEmbedding2D  *nn.Conv2D // Split model uses 2D conv (not auto-populated, set from PatchEmbedding when detected)
	PositionEmbedding *VisionPositionEmbedding
	Layers            []VisionEncoderLayer `gguf:"blk"`
	PatchMerger       *VisionPatchMerger   `gguf:"merger"`
	DeepstackMerger   []*VisionPatchMerger `gguf:"deepstack_merger"`

	VisionOptions
}

func (m *VisionModel) positions(ctx ml.Context, grid *Grid) (_, _ ml.Tensor) {
	// Create indices on CPU first
	cpuIndices := ctx.Input().FromInts(slices.Collect(func(yield func(int32) bool) {
		for y := range grid.Height {
			for x := range grid.Width {
				if !yield(int32(y)) {
					return
				}
				if !yield(int32(x)) {
					return
				}
			}
		}
	}), grid.Width*grid.Height*2)

	// Copy indices to GPU layer context for split model compatibility
	gpuIndicesDest := ctx.Layer(0).Zeros(ml.DTypeI32, grid.Width*grid.Height*2)
	indices := cpuIndices.Copy(ctx, gpuIndicesDest)
	// Use Contiguous(shape) to avoid view_src issues - this reshapes and creates independent tensor
	indices = indices.Contiguous(ctx, -1, grid.Width/m.spatialMergeSize, m.spatialMergeSize, grid.Height/m.spatialMergeSize)
	indices = indices.Permute(ctx, 0, 2, 1, 3)
	// Final reshape to 1D also uses Contiguous(shape) to avoid view_src
	indices = indices.Contiguous(ctx, -1)

	halfDim := m.headDim() / 2
	maxGrid := max(grid.Height, grid.Width)

	// Create frequency table
	// freqDim = halfDim/2 because indices contain BOTH y and x coordinates (2x factor)
	// The reshape later will combine them back to produce output matching seqLen
	freqDim := halfDim / 2
	freqCount := maxGrid * freqDim

	ropeTheta := float64(m.ropeTheta)
	freqData := make([]float32, freqCount)
	for i := range maxGrid {
		for j := range freqDim {
			freqData[i*freqDim+j] = float32(float64(i) / math.Pow(ropeTheta, float64(j*2)/float64(halfDim)))
		}
	}

	// Create frequencies tensor on input context (CPU) first
	cpuFrequencies := ctx.Input().FromFloats(freqData, freqDim, maxGrid)

	// For split models, explicitly copy to GPU layer context to avoid scheduler issues
	// Create a GPU destination tensor and copy the CPU data to it
	gpuDest := ctx.Layer(0).Zeros(ml.DTypeF32, freqDim, maxGrid)
	frequencies := cpuFrequencies.Copy(ctx, gpuDest)

	embeds := frequencies.Rows(ctx, indices)
	// Use Contiguous(ctx, shape...) to avoid view_src chain - this calls ggml_cont_Nd
	// which creates a truly independent tensor without view_src issues
	embeds = embeds.Contiguous(ctx, halfDim, 1, -1)
	embeds = embeds.Concat(ctx, embeds, 0)
	return embeds.Cos(ctx), embeds.Sin(ctx)
}

// Forward computes the vision model for an input tensor
func (m *VisionModel) Forward(ctx ml.Context, pixelValues ml.Tensor, grid *Grid) (ml.Tensor, []ml.Tensor) {
	var hiddenStates ml.Tensor

	if m.isSplitArchitecture {
		// Split architecture: weight was originally [16,16,3,1152] but LoadSecondary reshapes it
		// to [768, 1152] at load time to avoid view_src assertion failures in GGML.
		// Input [768, numPatches] is already patchDim,seqLen format from ImageProcessor.

		// Verify PatchEmbedding exists
		if m.PatchEmbedding == nil || m.PatchEmbedding.Weight == nil {
			panic("VisionModel.PatchEmbedding.Weight is nil - split model patch embedding not loaded correctly")
		}

		patchDim := m.numChannels * m.patchSize * m.patchSize // Should be 768
		numPatches := pixelValues.Dim(1)

		// Log shapes for debugging
		weightShape := m.PatchEmbedding.Weight.Shape()
		pixelShape := pixelValues.Shape()
		slog.Info("Split patch embedding forward",
			"weight_shape", weightShape, "pixel_shape", pixelShape,
			"expected_patchDim", patchDim, "hiddenSize", m.hiddenSize, "numPatches", numPatches)

		// Verify shapes before Mulmat to give clear error message
		// For Mulmat(a, b): requires a.ne[0] == b.ne[0]
		// Weight should be [patchDim, hiddenSize], pixelValues should be [patchDim, numPatches]
		if len(weightShape) >= 2 && len(pixelShape) >= 1 {
			if weightShape[0] != pixelShape[0] {
				slog.Error("Shape mismatch for patch embedding Mulmat",
					"weight_ne0", weightShape[0], "pixel_ne0", pixelShape[0],
					"need", "weight.ne[0] == pixel.ne[0]")
				panic(fmt.Sprintf("Patch embedding shape mismatch: weight[0]=%d != pixel[0]=%d", weightShape[0], pixelShape[0]))
			}
		}

		hiddenStates = m.PatchEmbedding.Weight.Mulmat(ctx, pixelValues)
		// Mulmat output is [hiddenSize, numPatches] - already correct shape, no Reshape needed

		if m.PatchEmbedding.Bias != nil {
			hiddenStates = hiddenStates.Add(ctx, m.PatchEmbedding.Bias)
		}
	} else {
		// Unified architecture: conv kernel is [kH, kW, temporal, channels*hidden] - use Conv3D
		pixelValues = pixelValues.Reshape(ctx, m.patchSize, m.patchSize, m.temporalPatchSize, -1)
		hiddenStates = m.PatchEmbedding.Forward(ctx, pixelValues, m.numChannels, m.patchSize, m.patchSize, m.temporalPatchSize, 0, 0, 0, 1, 1, 1)
	}

	hiddenStates = m.PositionEmbedding.Forward(ctx, hiddenStates, grid, m.VisionOptions)

	cos, sin := m.positions(ctx, grid)

	// Verify first layer exists before processing
	if len(m.Layers) == 0 {
		panic("VisionModel.Layers is empty - no vision encoder layers loaded")
	}
	// VisionEncoderLayer is a struct (not pointer), check MLP field directly
	if m.Layers[0].MLP == nil {
		slog.Error("First layer MLP is nil - aliases may not be working",
			"layer0_mlp", m.Layers[0].MLP,
			"layer0_norm1", m.Layers[0].Norm1,
			"layer0_norm2", m.Layers[0].Norm2)
		panic("VisionEncoderLayer.MLP is nil - MLP tensor aliases (ffn_up→linear_fc1) not working")
	}

	deepstackStates := make([]ml.Tensor, len(m.deepstackVisualIndexes))
	for i, layer := range m.Layers {
		hiddenStates = layer.Forward(ctx, hiddenStates, cos, sin, m.VisionOptions)
		if i := slices.Index(m.deepstackVisualIndexes, int32(i)); i >= 0 && m.DeepstackMerger[i] != nil {
			deepstackStates[i] = m.DeepstackMerger[i].Forward(ctx, hiddenStates, true, m.VisionOptions)
		}
	}

	// PatchMerger may be nil for split models
	if m.PatchMerger != nil {
		hiddenStates = m.PatchMerger.Forward(ctx, hiddenStates, false, m.VisionOptions)
	}
	return hiddenStates, deepstackStates
}

// newVisionModel creates a new instance of the Qwen vision model
func newVisionModel(c fs.Config) *VisionModel {
	deepstackVisualIndexes := c.Ints("vision.deepstack_visual_indexes")
	model := &VisionModel{
		Layers:          make([]VisionEncoderLayer, c.Uint("vision.block_count", 32)),
		DeepstackMerger: make([]*VisionPatchMerger, len(deepstackVisualIndexes)),
		VisionOptions: VisionOptions{
			hiddenSize:        int(c.Uint("vision.embedding_length", 1280)),
			numHeads:          int(c.Uint("vision.attention.head_count", 16)),
			patchSize:         int(c.Uint("vision.patch_size", 14)),
			numChannels:       int(c.Uint("vision.num_channels", 3)),
			eps:               c.Float("vision.attention.layer_norm_epsilon", 1e-6),
			ropeTheta:         c.Float("vision.rope.freq_base", 10000.0),
			spatialMergeSize:  int(c.Uint("vision.spatial_merge_size", 2)),
			temporalPatchSize: int(c.Uint("vision.temporal_patch_size", 2)),
			gridPerSide:       int(math.Sqrt(float64(c.Uint("vision.num_positional_embeddings", 2304)))),
			mropeSections: slices.Collect(func(yield func(int) bool) {
				for _, section := range c.Ints("mrope_sections", []int32{24, 20, 20}) {
					if !yield(int(section)) {
						return
					}
				}
			}),
			deepstackVisualIndexes: deepstackVisualIndexes,
		},
	}

	return model
}

// InferOptionsFromTensors updates VisionOptions by inferring dimensions from actual tensor shapes.
// This is used when config values are incorrect or missing (e.g., split GGUF models).
func (m *VisionModel) InferOptionsFromTensors() {
	// Infer hiddenSize from a layer norm bias tensor shape [hiddenSize]
	if len(m.Layers) > 0 && m.Layers[0].Norm1 != nil && m.Layers[0].Norm1.Bias != nil {
		dims := m.Layers[0].Norm1.Bias.Shape()
		if len(dims) > 0 && dims[0] > 0 {
			m.hiddenSize = int(dims[0])
		}
	}

	// Detect split model architecture from PatchEmbedding shape
	// Unified: [16, 16, 2, 3456] = [kH, kW, temporal, channels*hidden] - 3D conv
	// Split:   [16, 16, 3, 1152] = [kH, kW, channels, hidden] - effectively 2D conv

	// Check if Conv2D (split model) is available first
	if m.PatchEmbedding2D != nil && m.PatchEmbedding2D.Weight != nil {
		m.isSplitArchitecture = true
		m.temporalPatchSize = 1
		dims := m.PatchEmbedding2D.Weight.Shape()
		if len(dims) >= 4 {
			kH, kW := int(dims[0]), int(dims[1])
			if kH == kW && kH > 0 {
				m.patchSize = kH
			}
		}
	} else if m.PatchEmbedding != nil && m.PatchEmbedding.Weight != nil {
		dims := m.PatchEmbedding.Weight.Shape()
		slog.Info("InferOptionsFromTensors checking PatchEmbedding", "shape", dims, "len", len(dims))

		if len(dims) == 2 {
			// 2D weight [patchDim, hiddenSize] - this is from load-time reshape of split GGUF
			// Shape [768, 1152] means split architecture with patchDim=768, hiddenSize=1152
			patchDim, hiddenSize := int(dims[0]), int(dims[1])
			if hiddenSize == m.hiddenSize && patchDim > 0 {
				m.isSplitArchitecture = true
				m.temporalPatchSize = 1
				// Infer patchSize from patchDim = numChannels * patchSize * patchSize
				// 768 = 3 * 16 * 16, so patchSize = sqrt(patchDim / numChannels)
				patchArea := patchDim / m.numChannels
				for ps := 1; ps <= 64; ps++ {
					if ps*ps == patchArea {
						m.patchSize = ps
						break
					}
				}
				slog.Info("Detected split architecture from 2D weight",
					"patchDim", patchDim, "hiddenSize", hiddenSize, "patchSize", m.patchSize)
			}
		} else if len(dims) >= 4 {
			kH, kW, dim2, dim3 := int(dims[0]), int(dims[1]), int(dims[2]), int(dims[3])

			// Set patchSize from kernel dimensions
			if kH == kW && kH > 0 {
				m.patchSize = kH
			}

			// Detect architecture variant:
			// Unified: dim3 = numChannels * hiddenSize (e.g., 3456 = 3 * 1152)
			// Split: dim3 = hiddenSize (e.g., 1152), dim2 = numChannels (e.g., 3)
			if dim3 == m.hiddenSize && dim2 == m.numChannels {
				// Split architecture: [kH, kW, channels, hiddenSize]
				m.isSplitArchitecture = true
				m.temporalPatchSize = 1 // No temporal dimension in split
				// Create Conv2D using the same weight/bias tensors
				m.PatchEmbedding2D = &nn.Conv2D{
					Weight: m.PatchEmbedding.Weight,
					Bias:   m.PatchEmbedding.Bias,
				}
			} else if dim3 == m.numChannels*m.hiddenSize {
				// Unified architecture: [kH, kW, temporal, channels*hiddenSize]
				m.isSplitArchitecture = false
				m.temporalPatchSize = dim2
			}
		}
	}

	// Verify numHeads is compatible with hiddenSize
	// For Qwen3VL vision, numHeads=16 with headDim=72 (1152/16=72)
	// Only override if current config produces invalid headDim
	if m.hiddenSize > 0 && m.numHeads > 0 {
		headDim := m.hiddenSize / m.numHeads
		// If headDim is not a reasonable value (64, 72, 80, 96, 128), try to fix
		validHeadDims := map[int]bool{64: true, 72: true, 80: true, 96: true, 128: true}
		if !validHeadDims[headDim] || m.hiddenSize%m.numHeads != 0 {
			// Try common numHeads values for this hiddenSize
			for _, tryHeads := range []int{16, 12, 8, 20, 24} {
				if m.hiddenSize%tryHeads == 0 {
					tryHeadDim := m.hiddenSize / tryHeads
					if validHeadDims[tryHeadDim] {
						m.numHeads = tryHeads
						break
					}
				}
			}
		}
	}

	// Count actual populated layers
	populatedCount := 0
	for _, layer := range m.Layers {
		if layer.Norm1 != nil {
			populatedCount++
		}
	}
	if populatedCount > 0 && populatedCount != len(m.Layers) {
		// Resize layers array to match actual populated count
		m.Layers = m.Layers[:populatedCount]
	}
}
