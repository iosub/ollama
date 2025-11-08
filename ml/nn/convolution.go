package nn

import "github.com/ollama/ollama/ml"

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
	Weight1 ml.Tensor `gguf:"weight.1"` // For split GGUF temporal_patch_size=3
	Bias    ml.Tensor `gguf:"bias"`
}

func (m *Conv3D) Forward(ctx ml.Context, t ml.Tensor, c, s0, s1, s2, p0, p1, p2, d0, d1, d2 int) ml.Tensor {
	// For split GGUF: Weight1 indicates temporal_patch_size=3 with dual weights
	// Each weight processes all frames with stride=1 and outputs are concatenated
	// Non-split models (Weight1==nil) use original single-weight path
	if m.Weight1 != nil && s2 == 3 {
		t1 := m.Weight.Conv3D(ctx, t, c, s0, s1, 1, p0, p1, p2, d0, d1, d2)
		t2 := m.Weight1.Conv3D(ctx, t, c, s0, s1, 1, p0, p1, p2, d0, d1, d2)
		t = t1.Concat(ctx, t2, 0)
	} else {
		t = m.Weight.Conv3D(ctx, t, c, s0, s1, s2, p0, p1, p2, d0, d1, d2)
	}
	
	if m.Bias != nil {
		t = t.Add(ctx, m.Bias)
	}
	return t
}
