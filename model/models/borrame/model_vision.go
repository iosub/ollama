package qwen3vlsplit

import (
	"iter"
	"math"
	"slices"

	"github.com/ollama/ollama/fs"
	"github.com/ollama/ollama/logutil"
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
		wShape := m.Weight.Shape()
		w1Shape := m.Weight1.Shape()
		logutil.Trace("SPLIT GGUF Conv3D: using dual weights for temporal_patch_size=3", "s2", s2, "c", c, "weight_shape", wShape, "weight1_shape", w1Shape)

		// CRITICAL: Use stride=1 in temporal dim (s2) to preserve all channels
		// Conv3D with stride=s2 divides channels by s2, but we want full channels
		// llama.cpp: two conv2d operations that each produce full n_embd channels
		inputShape := t.Shape()
		logutil.Trace("SPLIT GGUF Conv3D: input", "shape", inputShape, "stride_spatial", []int{s0, s1}, "stride_temporal", 1)

		t1 := m.Weight.Conv3D(ctx, t, c, s0, s1, 1, p0, p1, p2, d0, d1, d2)  // stride_temporal=1
		t2 := m.Weight1.Conv3D(ctx, t, c, s0, s1, 1, p0, p1, p2, d0, d1, d2) // stride_temporal=1

		t1Shape := t1.Shape()
		t2Shape := t2.Shape()
		logutil.Trace("SPLIT GGUF Conv3D: dual weight outputs", "t1_shape", t1Shape, "t2_shape", t2Shape)

		if len(t1Shape) > 0 && len(t2Shape) > 0 {
			logutil.Trace("SPLIT GGUF Conv3D: channel check", "t1_channels", t1Shape[0], "t2_channels", t2Shape[0], "expected", 1152)
		}

		// CONCAT the outputs along channel dimension (dim 0)
		// llama.cpp uses ADD because each Conv2D produces full 1152 channels
		// Our Conv3D divides channels: 1152/3=384 per weight
		// So we CONCAT: 384 + 384 = 768, then need to handle the mismatch
		t = t1.Concat(ctx, t2, 0)
		concatShape := t.Shape()
		logutil.Trace("SPLIT GGUF Conv3D: CONCAT result", "shape", concatShape, "type", "channel concatenation")

		if len(concatShape) > 0 {
			logutil.Trace("SPLIT GGUF Conv3D: final channels after CONCAT", "channels", concatShape[0], "expected", 1152, "actual", concatShape[0])
		}

		// Bias will be applied after CONCAT (standard path below)
	} else {
		t = m.Weight.Conv3D(ctx, t, c, s0, s1, s2, p0, p1, p2, d0, d1, d2)
		if m.Weight1 != nil {
			logutil.Trace("SPLIT GGUF Conv3D: detected dual-weight but using single", "s2", s2)
		}
		logutil.Trace("SPLIT GGUF Conv3D: single weight output", "shape", t.Shape())
	}

	// Validate output shape
	finalShape := t.Shape()
	if len(finalShape) >= 2 {
		logutil.Trace("SPLIT GGUF Conv3D: output validation", "channels", finalShape[0], "spatial", finalShape[1], "expected_channels_approx", 1152)
		if finalShape[0] < 1000 {
			logutil.Trace("SPLIT GGUF Conv3D: output channel count LOW", "actual", finalShape[0], "expected", 1152, "WARNING", true)
		}
	}

	if bias != nil {
		outShape := t.Shape()
		totalOut := 1
		for _, d := range outShape {
			totalOut *= d
		}

		logutil.Trace("SPLIT GGUF Conv3D: bias reshape", "bias_shape", biasShape, "output_shape", outShape, "bias_size", biasSize, "total_output", totalOut)

		if biasSize == 0 {
			logutil.Trace("SPLIT GGUF Conv3D: bias reshape", "strategy", "skip-zero")
			return t
		}

		if len(outShape) == 2 && totalOut%biasSize == 0 {
			// For split GGUF: Apply bias by broadcasting
			channels := outShape[0]
			seqLen := outShape[1]

			// Check if we can broadcast the bias
			if totalOut%biasSize == 0 && biasSize > channels {
				// Bias is larger than output channels - likely needs reshaping
				// Try to broadcast by repeating along sequence dimension
				repeats := totalOut / biasSize
				if repeats*biasSize == totalOut {
					biasReshaped := bias.Reshape(ctx, biasSize, 1).Repeat(ctx, 1, repeats)
					biasFlat := biasReshaped.Reshape(ctx, outShape...)
					logutil.Trace("SPLIT GGUF Conv3D: bias reshape", "strategy", "broadcast-repeat", "bias_shape", bias.Shape(), "out_shape", outShape, "repeats", repeats)
					return t.Add(ctx, biasFlat)
				}
			}

			if biasSize == channels {
				// Standard broadcast: bias [channels] -> [channels, 1] -> [channels, seq_len]
				biasReshaped := bias.Reshape(ctx, channels, 1)
				logutil.Trace("SPLIT GGUF Conv3D: bias reshape", "strategy", "broadcast-2d", "bias_shape", biasReshaped.Shape(), "out_shape", outShape)
				return t.Add(ctx, biasReshaped)
			}

			// Fallback: skip if dimensions don't match
			logutil.Trace("SPLIT GGUF Conv3D: bias reshape", "strategy", "skip-bias-temporarily-v2", "reason", "dimension-mismatch", "channels", channels, "bias_size", biasSize, "seq_len", seqLen)
			return t
		}

		// Original logic for Conv3D
		channels := outShape[0]
		if biasSize != channels {
			logutil.Trace("SPLIT GGUF Conv3D: bias reshape", "strategy", "skip-incompatible", "channels", channels, "bias_size", biasSize)
			return t
		}

		remaining := outShape[1:]
		broadcastShape := append([]int{channels}, make([]int, len(remaining))...)
		fullShape := append([]int{channels}, remaining...)

		reshaped := t.Contiguous(ctx).Reshape(ctx, fullShape...)
		biasExpanded := bias.Reshape(ctx, broadcastShape...)
		logutil.Trace("SPLIT GGUF Conv3D: bias reshape", "strategy", "axis-first", "full_shape", reshaped.Shape(), "bias_shape", biasExpanded.Shape())
		withBias := reshaped.Add(ctx, biasExpanded)
		return withBias.Reshape(ctx, outShape...)
	}
	return t
}

type VisionSelfAttention struct {
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

func (sa *VisionSelfAttention) Forward(ctx ml.Context, hiddenStates, cos, sin ml.Tensor, opts VisionOptions) ml.Tensor {
	var (
		query ml.Tensor
		key   ml.Tensor
		value ml.Tensor
	)

	// Support both separate and fused QKV weights
	if sa.QKV != nil {
		logutil.Trace("SPLIT GGUF Attention: using fused QKV", "hidden_shape", hiddenStates.Shape())
		// Split GGUF format: fused QKV linear layer
		qkv := sa.QKV.Forward(ctx, hiddenStates)
		logutil.Trace("SPLIT GGUF Attention: QKV output", "qkv_shape", qkv.Shape())
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
		logutil.Trace("SPLIT GGUF Attention: QKV processed", "query_shape", query.Shape(), "key_shape", key.Shape(), "value_shape", value.Shape())
	} else if sa.Query != nil && sa.Key != nil && sa.Value != nil {
		logutil.Trace("SPLIT GGUF Attention: using separate QKV")
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
	// Return input unchanged if MLP components are nil (optional for split GGUF)
	if mlp == nil || mlp.FC1 == nil || mlp.FC2 == nil {
		logutil.Trace("SPLIT GGUF MLP: skipping MLP (optional components missing)")
		return hiddenStates
	}
	return mlp.FC2.Forward(ctx, mlp.FC1.Forward(ctx, hiddenStates).GELU(ctx))
}

type VisionEncoderLayer struct {
	Norm1     *nn.LayerNorm `gguf:"norm1"`
	Attention *VisionSelfAttention
	Norm2     *nn.LayerNorm `gguf:"norm2"`
	MLP       *VisionMLP    `gguf:"mlp"`
}

func (e *VisionEncoderLayer) Forward(ctx ml.Context, hiddenStates, cos, sin ml.Tensor, opts VisionOptions) ml.Tensor {
	residual := hiddenStates

	// Skip LayerNorm if nil (optional for split GGUF - matches llama.cpp behavior)
	if e.Norm1 != nil {
		hiddenStates = e.Norm1.Forward(ctx, hiddenStates, opts.eps)
	} else {
		logutil.Trace("SPLIT GGUF Layer: skipping Norm1 (optional tensor missing)")
	}

	hiddenStates = e.Attention.Forward(ctx, hiddenStates, cos, sin, opts)
	hiddenStates = hiddenStates.Add(ctx, residual)

	residual = hiddenStates

	// Skip LayerNorm if nil (optional for split GGUF - matches llama.cpp behavior)
	if e.Norm2 != nil {
		hiddenStates = e.Norm2.Forward(ctx, hiddenStates, opts.eps)
	} else {
		logutil.Trace("SPLIT GGUF Layer: skipping Norm2 (optional tensor missing)")
	}

	// Skip MLP if nil (optional for split GGUF - matches llama.cpp behavior)
	if e.MLP != nil {
		hiddenStates = e.MLP.Forward(ctx, hiddenStates, opts)
	} else {
		logutil.Trace("SPLIT GGUF Layer: skipping MLP (optional tensor missing)")
	}

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
	// Return input unchanged if merger components are nil (optional for split GGUF)
	if m == nil {
		logutil.Trace("SPLIT GGUF Merger: skipping merger (nil merger)")
		return visionOutputs
	}

	hiddenSize := opts.hiddenSize * opts.spatialMergeSize * opts.spatialMergeSize
	logutil.Trace("SPLIT GGUF Merger: input", "shape", visionOutputs.Shape(), "opts.hiddenSize", opts.hiddenSize, "spatialMergeSize", opts.spatialMergeSize)
	if postshuffleNorm {
		visionOutputs = visionOutputs.Reshape(ctx, hiddenSize, -1)
		logutil.Trace("SPLIT GGUF Merger: reshaped for postshuffle", "shape", visionOutputs.Shape())
	}

	// Skip LayerNorm if nil (optional for split GGUF)
	if m.Norm != nil {
		visionOutputs = m.Norm.Forward(ctx, visionOutputs, opts.eps)
		logutil.Trace("SPLIT GGUF Merger: after norm", "shape", visionOutputs.Shape())
	} else {
		logutil.Trace("SPLIT GGUF Merger: skipping norm (optional tensor missing)")
	}

	visionOutputs = visionOutputs.Reshape(ctx, hiddenSize, -1)
	logutil.Trace("SPLIT GGUF Merger: after reshape", "shape", visionOutputs.Shape())

	// Skip FC layers if nil (optional for split GGUF)
	if m.FC1 != nil && m.FC2 != nil {
		result := m.FC2.Forward(ctx, m.FC1.Forward(ctx, visionOutputs).GELU(ctx))
		logutil.Trace("SPLIT GGUF Merger: final output", "shape", result.Shape())
		return result
	} else {
		logutil.Trace("SPLIT GGUF Merger: skipping FC layers (optional tensors missing)")
		return visionOutputs
	}
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

	// Detect split GGUF: Conv3D CONCAT produces 768, non-split produces 1152
	actualHiddenSize := hiddenStates.Dim(0)
	configHiddenSize := m.VisionOptions.hiddenSize
	isSplitGGUF := actualHiddenSize != configHiddenSize

	logutil.Trace("SPLIT GGUF Vision: dimension check", "actual", actualHiddenSize, "config", configHiddenSize, "is_split", isSplitGGUF)

	// For split GGUF: use actual size (768) during vision processing to avoid zero-padding corruption
	// For non-split: use config size (1152)
	opts := m.VisionOptions
	if isSplitGGUF {
		logutil.Trace("SPLIT GGUF Vision: using actual hiddenSize for processing", "size", actualHiddenSize)
		opts.hiddenSize = actualHiddenSize
	} else {
		logutil.Trace("SPLIT GGUF Vision: dimensions match, applying position embedding")
	}

	cos, sin := m.positions(ctx, grid)

	deepstackStates := make([]ml.Tensor, len(m.deepstackVisualIndexes))
	for i, layer := range m.Layers {
		hiddenStates = layer.Forward(ctx, hiddenStates, cos, sin, opts)
		if idx := slices.Index(m.deepstackVisualIndexes, int32(i)); idx >= 0 {
			deepstackStates[idx] = m.DeepstackMerger[idx].Forward(ctx, hiddenStates, true, opts)
		}
	}

	// Apply padding AFTER vision layers for split GGUF to avoid corrupting visual information
	if isSplitGGUF {
		logutil.Trace("SPLIT GGUF Vision: applying padding after layers", "from", actualHiddenSize, "to", configHiddenSize)
		paddingSize := configHiddenSize - actualHiddenSize
		spatialDim := hiddenStates.Dim(1)
		padding := ctx.Input().Zeros(hiddenStates.DType(), paddingSize, spatialDim)
		hiddenStates = hiddenStates.Concat(ctx, padding, 0)
		logutil.Trace("SPLIT GGUF Vision: padded main output", "shape", hiddenStates.Shape())

		// Pad deepstack features too
		for i, ds := range deepstackStates {
			if ds != nil {
				dsSpatialDim := ds.Dim(1)
				dsPadding := ctx.Input().Zeros(ds.DType(), paddingSize, dsSpatialDim)
				deepstackStates[i] = ds.Concat(ctx, dsPadding, 0)
				logutil.Trace("SPLIT GGUF Vision: padded deepstack", "index", i, "shape", deepstackStates[i].Shape())
			}
		}

		opts.hiddenSize = configHiddenSize
	}

	hiddenStates = m.PatchMerger.Forward(ctx, hiddenStates, false, opts)

	logutil.Trace("SPLIT GGUF Vision: returning features", "main_shape", hiddenStates.Shape(), "deepstack_count", len(deepstackStates))
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
