package nn

import "github.com/ollama/ollama/ml"

type Linear struct {
	Weight ml.Tensor `gguf:"weight"`
	Bias   ml.Tensor `gguf:"bias"`
}

func (m *Linear) Forward(ctx ml.Context, t ml.Tensor) ml.Tensor {
	// For split GGUF: if Linear is nil or Weight is nil, return input unchanged
	if m == nil || m.Weight == nil {
		return t
	}
	t = m.Weight.Mulmat(ctx, t)
	if m.Bias != nil {
		t = t.Add(ctx, m.Bias)
	}

	return t
}

type LinearBatch struct {
	Weight ml.Tensor `gguf:"weight"`
	Bias   ml.Tensor `gguf:"bias"`
}

func (m *LinearBatch) Forward(ctx ml.Context, t, indices ml.Tensor) ml.Tensor {
	t = m.Weight.MulmatID(ctx, t, indices)
	if m.Bias != nil {
		t = t.AddID(ctx, m.Bias, indices)
	}

	return t
}
