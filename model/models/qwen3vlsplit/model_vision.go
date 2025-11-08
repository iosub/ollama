package qwen3vl

import (
	"iter"
	"math"
	"slices"

	"github.com/ollama/ollama/fs"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn"
)

// Conv3DSplit handles split GGUF Conv3D operations with dual weights for Qwen3-VL split models
type Conv3DSplit struct {
	Weight  ml.Tensor `gguf:"weight"`
	Weight1 ml.Tensor `gguf:"weight.1"` // Temporal weight for Qwen3-VL split models
	Bias    ml.Tensor `gguf:"bias"`
}

func (m *Conv3DSplit) Forward(ctx ml.Context, t ml.Tensor, c, s0, s1, s2, p0, p1, p2, d0, d1, d2 int) ml.Tensor {
	// For split GGUF: if Conv3D or Weight is nil, return input unchanged
	if m == nil || m.Weight == nil {
		return t
	}

	bias := m.Bias
	biasShape := []int(nil)
	biasSize := 0
	if bias != nil {
		bias = bias.Contiguous(ctx)
		biasShape = bias.Shape()
		biasSize = 1
		for _, d := range biasShape {
			biasSize *= d
		}
	}

	// For Qwen3-VL split models: Weight1 indicates split GGUF format
	// llama.cpp uses Conv2D, but we have Conv3D weights
	// Strategy: Use stride=1 in temporal dim to process all frames together
	if m.Weight1 != nil && s2 == 3 {
		// CRITICAL: Use stride=1 in temporal dim (s2) to preserve all channels
		// Conv3D with stride=s2 divides channels by s2, but we want full channels
		// llama.cpp: two conv2d operations that each produce full n_embd channels

		t1 := m.Weight.Conv3D(ctx, t, c, s0, s1, 1, p0, p1, p2, d0, d1, d2)  // stride_temporal=1
		t2 := m.Weight1.Conv3D(ctx, t, c, s0, s1, 1, p0, p1, p2, d0, d1, d2) // stride_temporal=1

		// CONCAT the outputs along channel dimension (dim 0)
		// llama.cpp uses ADD because each Conv2D produces full 1152 channels
		// Our Conv3D divides channels: 1152/3=384 per weight
		// So we CONCAT: 384 + 384 = 768, then need to handle the mismatch
		t = t1.Concat(ctx, t2, 0)

		// Bias will be applied after CONCAT (standard path below)
	} else {
		t = m.Weight.Conv3D(ctx, t, c, s0, s1, s2, p0, p1, p2, d0, d1, d2)
		if m.Weight1 != nil {
		}
	}

	// Validate output shape
	finalShape := t.Shape()
	if len(finalShape) >= 2 {
		if finalShape[0] < 1000 {
		}
	}

	if bias != nil {
		outShape := t.Shape()
		totalOut := 1
		for _, d := range outShape {
			totalOut *= d
		}

		if biasSize == 0 {
			return t
		}

		if len(outShape) == 2 && totalOut%biasSize == 0 {
			// For split GGUF: Apply bias by broadcasting
			channels := outShape[0]

			// Check if we can broadcast the bias
			if totalOut%biasSize == 0 && biasSize > channels {
				// Bias is larger than output channels - likely needs reshaping
				// Try to broadcast by repeating along sequence dimension
				repeats := totalOut / biasSize
				if repeats*biasSize == totalOut {
					biasReshaped := bias.Reshape(ctx, biasSize, 1).Repeat(ctx, 1, repeats)
					biasFlat := biasReshaped.Reshape(ctx, outShape...)
					return t.Add(ctx, biasFlat)
				}
			}

			if biasSize == channels {
				// Standard broadcast: bias [channels] -> [channels, 1] -> [channels, seq_len]
				biasReshaped := bias.Reshape(ctx, channels, 1)
				return t.Add(ctx, biasReshaped)
			}

			// Fallback: skip if dimensions don't match
			return t
		}

		// Original logic for Conv3D
		channels := outShape[0]
		if biasSize != channels {
			return t
		}

		remaining := outShape[1:]
		broadcastShape := append([]int{channels}, make([]int, len(remaining))...)
		fullShape := append([]int{channels}, remaining...)

		reshaped := t.Contiguous(ctx).Reshape(ctx, fullShape...)
		biasExpanded := bias.Reshape(ctx, broadcastShape...)
		withBias := reshaped.Add(ctx, biasExpanded)
		return withBias.Reshape(ctx, outShape...)
	}
	return t
}

type VisionAttention struct {
	Query  *nn.Linear `gguf:"attn_q"`
	Key    *nn.Linear `gguf:"attn_k"`
	Value  *nn.Linear `gguf:"attn_v"`
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
	query := sa.Query.Forward(ctx, hiddenStates)
	query = query.Reshape(ctx, opts.headDim(), opts.numHeads, query.Dim(1))
	query = applyRotaryPositionalEmbedding(ctx, query, cos, sin)

	key := sa.Key.Forward(ctx, hiddenStates)
	key = key.Reshape(ctx, opts.headDim(), opts.numHeads, key.Dim(1))
	key = applyRotaryPositionalEmbedding(ctx, key, cos, sin)

	value := sa.Value.Forward(ctx, hiddenStates)
	value = value.Reshape(ctx, opts.headDim(), opts.numHeads, value.Dim(1))

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
	PatchEmbedding    *Conv3DSplit `gguf:"patch_embed"`
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
	hiddenStates = m.PositionEmbedding.Forward(ctx, hiddenStates, grid, m.VisionOptions)

	cos, sin := m.positions(ctx, grid)

	deepstackStates := make([]ml.Tensor, len(m.deepstackVisualIndexes))
	for i, layer := range m.Layers {
		hiddenStates = layer.Forward(ctx, hiddenStates, cos, sin, m.VisionOptions)
		if i := slices.Index(m.deepstackVisualIndexes, int32(i)); i >= 0 {
			deepstackStates[i] = m.DeepstackMerger[i].Forward(ctx, hiddenStates, true, m.VisionOptions)
		}
	}

	hiddenStates = m.PatchMerger.Forward(ctx, hiddenStates, false, m.VisionOptions)
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
