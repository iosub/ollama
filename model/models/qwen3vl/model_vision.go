package qwen3vl

import (
	"iter"
	"math"
	"slices"

	"github.com/ollama/ollama/fs"
	"github.com/ollama/ollama/logutil"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn"
)

type VisionAttention struct {
	Query  *nn.Linear `gguf:"attn_q"`
	Key    *nn.Linear `gguf:"attn_k"`
	Value  *nn.Linear `gguf:"attn_v"`
	QKV    *nn.Linear `gguf:"attn_qkv"` // For split GGUF format
	Output *nn.Linear `gguf:"attn_out"`
}

func rotateHalf(ctx ml.Context, t ml.Tensor) ml.Tensor {
	x1 := t.View(ctx, 0, t.Dim(0)/2, t.Stride(1), t.Dim(1), t.Stride(2), t.Dim(2), t.Stride(3), t.Dim(3))
	x2 := t.View(ctx, t.Stride(0)*t.Dim(0)/2, t.Dim(0)/2, t.Stride(1), t.Dim(1), t.Stride(2), t.Dim(2), t.Stride(3), t.Dim(3)).Contiguous(ctx)
	return x2.Scale(ctx, -1).Concat(ctx, x1, 0)
}

func applyRotaryPositionalEmbedding(ctx ml.Context, t, cos, sin ml.Tensor) ml.Tensor {
	return t.Mul(ctx, cos).Add(ctx, rotateHalf(ctx, t).Mul(ctx, sin))
}

func (sa *VisionAttention) Forward(ctx ml.Context, hiddenStates, cos, sin ml.Tensor, opts VisionOptions) ml.Tensor {
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

	attention := nn.Attention(ctx, query, key, value, math.Pow(float64(opts.headDim()), -0.5), nil)
	attention = attention.Reshape(ctx, opts.hiddenSize, attention.Dim(2))
	return sa.Output.Forward(ctx, attention)
}

type VisionMLP struct {
	FC1 *nn.Linear `gguf:"linear_fc1"`
	FC2 *nn.Linear `gguf:"linear_fc2"`
}

func (mlp *VisionMLP) Forward(ctx ml.Context, hiddenStates ml.Tensor, opts VisionOptions) ml.Tensor {
	return mlp.FC2.Forward(ctx, mlp.FC1.Forward(ctx, hiddenStates).GELU(ctx))
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

	deepstackVisualIndexes []int32
	mropeSections          []int
	
	// For split GGUF: config hiddenSize for merger calculations even when actual tensors differ
	configHiddenSize int
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
	// Use configHiddenSize for split GGUF merger calculations (weights expect original dimensions)
	mergerHiddenSize := opts.hiddenSize
	if opts.configHiddenSize > 0 {
		mergerHiddenSize = opts.configHiddenSize
	}
	hiddenSize := mergerHiddenSize * opts.spatialMergeSize * opts.spatialMergeSize
	if postshuffleNorm {
		visionOutputs = visionOutputs.Reshape(ctx, hiddenSize, -1)
	}

	visionOutputs = m.Norm.Forward(ctx, visionOutputs, opts.eps)
	visionOutputs = visionOutputs.Reshape(ctx, hiddenSize, -1)
	return m.FC2.Forward(ctx, m.FC1.Forward(ctx, visionOutputs).GELU(ctx))
}

type VisionPositionEmbedding struct {
	PositionEmbedding *nn.Embedding `gguf:"pos_embed"`
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
	indexSlice := slices.Collect(makeSlice2D[int32](4, grid.Height*grid.Width))
	weightSlice := slices.Collect(makeSlice2D[float32](4, grid.Height*grid.Width))

	stepHeight := float32(opts.gridPerSide-1) / float32(grid.Height-1)
	stepWidth := float32(opts.gridPerSide-1) / float32(grid.Width-1)

	var i int
	for h := range grid.Height {
		for w := range grid.Width {
			y, x := float32(h)*stepHeight, float32(w)*stepWidth

			floorY, floorX := int32(y), int32(x)
			ceilY, ceilX := min(floorY+1, int32(opts.gridPerSide-1)), min(floorX+1, int32(opts.gridPerSide-1))

			indexSlice[0][i] = floorY*int32(opts.gridPerSide) + floorX
			indexSlice[1][i] = floorY*int32(opts.gridPerSide) + ceilX
			indexSlice[2][i] = ceilY*int32(opts.gridPerSide) + floorX
			indexSlice[3][i] = ceilY*int32(opts.gridPerSide) + ceilX

			weightSlice[0][i] = (1 - (y - float32(floorY))) * (1 - (x - float32(floorX)))
			weightSlice[1][i] = (1 - (y - float32(floorY))) * (x - float32(floorX))
			weightSlice[2][i] = (y - float32(floorY)) * (1 - (x - float32(floorX)))
			weightSlice[3][i] = (y - float32(floorY)) * (x - float32(floorX))

			i++
		}
	}

	indices := ctx.Input().FromInts(slices.Concat(indexSlice...), grid.Height*grid.Width*4)
	weights := ctx.Input().FromFloats(slices.Concat(weightSlice...), 1, grid.Height*grid.Width*4)

	n := hiddenStates.Dim(0)
	positionEmbeds := m.PositionEmbedding.Forward(ctx, indices)
	positionEmbeds = positionEmbeds.Mul(ctx, weights)
	
	// For split GGUF: if positionEmbeds has more channels than hiddenStates, use only first n channels
	posEmbedDim := positionEmbeds.Dim(0)
	if posEmbedDim > n {
		// Use View to get first n channels: [1152, ...] -> [768, ...]
		positionEmbeds = positionEmbeds.View(ctx, 0, n, positionEmbeds.Stride(1), positionEmbeds.Dim(1))
		logutil.Trace("position embedding truncated for split GGUF", "original", posEmbedDim, "used", n, "hidden", hiddenStates.Shape())
	}
	
	positionEmbeds = positionEmbeds.Reshape(ctx, n, -1, 4)

	positionEmbeds = positionEmbeds.View(ctx, 0, n, positionEmbeds.Stride(1), grid.Height*grid.Width).
		Add(ctx, positionEmbeds.View(ctx, 1*positionEmbeds.Stride(2), n, positionEmbeds.Stride(1), grid.Height*grid.Width)).
		Add(ctx, positionEmbeds.View(ctx, 2*positionEmbeds.Stride(2), n, positionEmbeds.Stride(1), grid.Height*grid.Width)).
		Add(ctx, positionEmbeds.View(ctx, 3*positionEmbeds.Stride(2), n, positionEmbeds.Stride(1), grid.Height*grid.Width))

	positionEmbeds = positionEmbeds.Reshape(ctx, -1, grid.Width/opts.spatialMergeSize, opts.spatialMergeSize, grid.Height/opts.spatialMergeSize)
	positionEmbeds = positionEmbeds.Permute(ctx, 0, 2, 1, 3).Contiguous(ctx, n, -1)
	return hiddenStates.Add(ctx, positionEmbeds)
}

type VisionModel struct {
	PatchEmbedding    *nn.Conv3D `gguf:"patch_embed"`
	PositionEmbedding *VisionPositionEmbedding
	Layers            []VisionEncoderLayer `gguf:"blk"`
	PatchMerger       *VisionPatchMerger   `gguf:"merger"`
	DeepstackMerger   []*VisionPatchMerger `gguf:"deepstack_merger"`

	VisionOptions
}

func (m *VisionModel) positions(ctx ml.Context, grid *Grid) (_, _ ml.Tensor) {
	indices := ctx.Input().FromInts(slices.Collect(func(yield func(int32) bool) {
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

	indices = indices.Reshape(ctx, -1, grid.Width/m.spatialMergeSize, m.spatialMergeSize, grid.Height/m.spatialMergeSize)
	indices = indices.Permute(ctx, 0, 2, 1, 3).Contiguous(ctx)
	indices = indices.Reshape(ctx, -1)

	halfDim := m.headDim() / 2
	maxGrid := max(grid.Height, grid.Width)
	frequencies := ctx.Input().FromFloats(slices.Collect(func(yield func(float32) bool) {
		ropeTheta := float64(m.ropeTheta)
		for i := range maxGrid {
			for j := range halfDim / 2 {
				if !yield(float32(i) / float32(math.Pow(ropeTheta, float64(j*2)/float64(halfDim)))) {
					return
				}
			}
		}
	}), halfDim/2, maxGrid)

	embeds := frequencies.Rows(ctx, indices)
	embeds = embeds.Reshape(ctx, halfDim, 1, -1)
	embeds = embeds.Concat(ctx, embeds, 0)
	return embeds.Cos(ctx), embeds.Sin(ctx)
}

// Forward computes the vision model for an input tensor
func (m *VisionModel) Forward(ctx ml.Context, pixelValues ml.Tensor, grid *Grid) (ml.Tensor, []ml.Tensor) {
	pixelValues = pixelValues.Reshape(ctx, m.patchSize, m.patchSize, m.temporalPatchSize, -1)
	hiddenStates := m.PatchEmbedding.Forward(ctx, pixelValues, m.numChannels, m.patchSize, m.patchSize, m.temporalPatchSize, 0, 0, 0, 1, 1, 1)
	
	// For split GGUF: adjust hiddenSize if PatchEmbedding produced fewer channels
	actualHiddenSize := hiddenStates.Dim(0)
	opts := m.VisionOptions
	if actualHiddenSize != opts.hiddenSize {
		logutil.Trace("split GGUF dimension mismatch", "config", opts.hiddenSize, "actual", actualHiddenSize)
		// Save original hiddenSize for merger calculations
		opts.configHiddenSize = opts.hiddenSize  // 1152
		
		// Apply position embedding with truncated dimensions
		posOpts := opts
		posOpts.hiddenSize = actualHiddenSize  // 768 for position embedding
		hiddenStates = m.PositionEmbedding.Forward(ctx, hiddenStates, grid, posOpts)
		
		// Pad hiddenStates from [768, spatial] to [1152, spatial] for vision layers
		// Vision layer weights expect full hiddenSize=1152
		spatialDim := hiddenStates.Dim(1)
		paddingSize := opts.hiddenSize - actualHiddenSize  // 1152 - 768 = 384
		padding := ctx.Input().Zeros(hiddenStates.DType(), paddingSize, spatialDim)
		hiddenStates = hiddenStates.Concat(ctx, padding, 0)  // [768+384, spatial] = [1152, spatial]
		logutil.Trace("padded hiddenStates for split GGUF", "from", actualHiddenSize, "to", opts.hiddenSize, "shape", hiddenStates.Shape())
	} else {
		hiddenStates = m.PositionEmbedding.Forward(ctx, hiddenStates, grid, opts)
	}

	cos, sin := m.positions(ctx, grid)

	deepstackStates := make([]ml.Tensor, len(m.deepstackVisualIndexes))
	for i, layer := range m.Layers {
		hiddenStates = layer.Forward(ctx, hiddenStates, cos, sin, opts)
		if i := slices.Index(m.deepstackVisualIndexes, int32(i)); i >= 0 {
			deepstackStates[i] = m.DeepstackMerger[i].Forward(ctx, hiddenStates, true, opts)
		}
	}

	hiddenStates = m.PatchMerger.Forward(ctx, hiddenStates, false, opts)
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
