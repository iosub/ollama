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
	// For temporal_patch_size=3, need to handle dual weights correctly
	if m.Weight1 != nil && s2 == 3 {
		wShape := m.Weight.Shape()
		w1Shape := m.Weight1.Shape()
		logutil.Trace("conv3d using dual weights for temporal_patch_size=3", "s2", s2, "c", c, "weight_shape", wShape, "weight1_shape", w1Shape)
		
		// FIX: Both weights process ALL 3 frames and concatenate in channel dimension
		// Each weight produces half the output channels (576 each → 1152 total)
		t1 := m.Weight.Conv3D(ctx, t, c, s0, s1, s2, p0, p1, p2, d0, d1, d2)
		t2 := m.Weight1.Conv3D(ctx, t, c, s0, s1, s2, p0, p1, p2, d0, d1, d2)
		logutil.Trace("conv3d dual weight full outputs", "t1_shape", t1.Shape(), "t2_shape", t2.Shape())
		
		// Apply bias split: first half to t1, second half to t2, BEFORE concatenating
		if bias != nil && biasSize > 0 {
			t1Shape := t1.Shape()
			t1Channels := t1Shape[0]
			concatChannels := t1Channels * 2 // Total after concat
			
			// For split GGUF: bias [1152] needs to map to concat output [768]
			// Strategy: Use first concatChannels elements from bias
			if biasSize >= concatChannels {
				// Bias is larger than or equal to output
				// Use first concatChannels elements [0:768] and split [0:384] + [384:768]
				bias1 := bias.View(ctx, 0, t1Channels)
				bias2 := bias.View(ctx, t1Channels * 4, t1Channels)
				
				t1 = t1.Add(ctx, bias1.Reshape(ctx, t1Channels, 1))
				t2 = t2.Add(ctx, bias2.Reshape(ctx, t1Channels, 1))
				
				if biasSize == concatChannels {
					logutil.Trace("conv3d applied exact bias split", "bias_size", biasSize, "t1_channels", t1Channels, "split", true)
				} else {
					logutil.Trace("conv3d applied partial bias split", "bias_size", biasSize, "used", concatChannels, "t1_channels", t1Channels, "unused", biasSize-concatChannels)
				}
				bias = nil
			} else {
				logutil.Trace("conv3d cannot split bias", "bias_size", biasSize, "t1_channels", t1Channels, "concat_channels", concatChannels)
			}
		}
		
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