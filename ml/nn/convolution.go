package nn

import (
	"github.com/ollama/ollama/logutil"
	"github.com/ollama/ollama/ml"
)

// Bias broadcast helpers ensure Conv layers can add channel biases even when
// ggml exposes tensors in unexpected shapes after convolution.

type Conv2D struct {
	Weight ml.Tensor `gguf:"weight"`
	Bias   ml.Tensor `gguf:"bias"`
}

func (m *Conv2D) Forward(ctx ml.Context, t ml.Tensor, s0, s1, p0, p1, d0, d1 int) ml.Tensor {
	t = m.Weight.Conv2D(ctx, t, s0, s1, p0, p1, d0, d1)
	if m.Bias != nil {
		// Bias shape is (out_channels,) while t shape is (width, height, out_channels, batch)
		t = t.Add(ctx, m.Bias.Reshape(ctx, 1, 1, -1))
	}
	return t
}

type Conv3D struct {
	Weight  ml.Tensor `gguf:"weight"`
	Weight1 ml.Tensor `gguf:"weight.1"` // Temporal weight for Qwen3-VL split models
	Bias    ml.Tensor `gguf:"bias"`
}

func (m *Conv3D) Forward(ctx ml.Context, t ml.Tensor, c, s0, s1, s2, p0, p1, p2, d0, d1, d2 int) ml.Tensor {
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
		logutil.Trace("conv3d using dual weights for temporal_patch_size=3", "s2", s2, "c", c, "weight_shape", wShape, "weight1_shape", w1Shape)
		
		// CRITICAL: Use stride=1 in temporal dim (s2) to preserve all channels
		// Conv3D with stride=s2 divides channels by s2, but we want full channels
		// llama.cpp: two conv2d operations that each produce full n_embd channels
		inputShape := t.Shape()
		logutil.Trace("SPLIT GGUF Conv3D: input", "shape", inputShape, "stride_spatial", []int{s0, s1}, "stride_temporal", 1)
		
		t1 := m.Weight.Conv3D(ctx, t, c, s0, s1, 1, p0, p1, p2, d0, d1, d2)  // stride_temporal=1
		t2 := m.Weight1.Conv3D(ctx, t, c, s0, s1, 1, p0, p1, p2, d0, d1, d2)  // stride_temporal=1
		
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
			logutil.Trace("conv3d detected dual-weight but using single", "s2", s2)
		}
		logutil.Trace("conv3d single weight output", "shape", t.Shape())
	}
	
	// Validate output shape
	finalShape := t.Shape()
	if len(finalShape) >= 2 {
		logutil.Trace("conv3d output validation", "channels", finalShape[0], "spatial", finalShape[1], "expected_channels_approx", 1152)
		if finalShape[0] < 1000 {
			logutil.Trace("conv3d output channel count LOW", "actual", finalShape[0], "expected", 1152, "WARNING", true)
		}
	}
	
	if bias != nil {
		outShape := t.Shape()
		totalOut := 1
		for _, d := range outShape {
			totalOut *= d
		}

		logutil.Trace("conv3d bias reshape", "bias_shape", biasShape, "output_shape", outShape, "bias_size", biasSize, "total_output", totalOut)

		if biasSize == 0 {
			logutil.Trace("conv3d bias reshape", "strategy", "skip-zero")
			return t
		}

		if len(outShape) < 2 {
			logutil.Trace("conv3d bias reshape", "strategy", "scalar-or-1d")
			return t.Add(ctx, bias.Reshape(ctx, outShape...))
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
			if repeats * biasSize == totalOut {
				biasReshaped := bias.Reshape(ctx, biasSize, 1).Repeat(ctx, 1, repeats)
				biasFlat := biasReshaped.Reshape(ctx, outShape...)
				logutil.Trace("conv3d bias reshape", "strategy", "broadcast-repeat", "bias_shape", bias.Shape(), "out_shape", outShape, "repeats", repeats)
				return t.Add(ctx, biasFlat)
			}
		}
		
		if biasSize == channels {
			// Standard broadcast: bias [channels] -> [channels, 1] -> [channels, seq_len]
			biasReshaped := bias.Reshape(ctx, channels, 1)
			logutil.Trace("conv3d bias reshape", "strategy", "broadcast-2d", "bias_shape", biasReshaped.Shape(), "out_shape", outShape)
			return t.Add(ctx, biasReshaped)
		}
		
		// Fallback: skip if dimensions don't match
		logutil.Trace("conv3d bias reshape", "strategy", "skip-bias-temporarily-v2", "reason", "dimension-mismatch", "channels", channels, "bias_size", biasSize, "seq_len", seqLen)
		return t
	}
	
	// Original logic for Conv3D
	channels := outShape[0]
	if biasSize != channels {
		logutil.Trace("conv3d bias reshape", "strategy", "skip-incompatible", "channels", channels, "bias_size", biasSize)
		return t
	}

	remaining := outShape[1:]
	broadcastShape := append([]int{channels}, make([]int, len(remaining))...)
	fullShape := append([]int{channels}, remaining...)

	reshaped := t.Contiguous(ctx).Reshape(ctx, fullShape...)
	biasExpanded := bias.Reshape(ctx, broadcastShape...)
	logutil.Trace("conv3d bias reshape", "strategy", "axis-first", "full_shape", reshaped.Shape(), "bias_shape", biasExpanded.Shape())
	withBias := reshaped.Add(ctx, biasExpanded)
	return withBias.Reshape(ctx, outShape...)
}
	return t
}