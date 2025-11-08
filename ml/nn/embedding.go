package nn

import "github.com/ollama/ollama/ml"

type Embedding struct {
	Weight ml.Tensor `gguf:"weight"`
}

func (m *Embedding) Forward(ctx ml.Context, hiddenState ml.Tensor) ml.Tensor {
	// For split GGUF: if Embedding is nil or Weight is nil, return input unchanged
	if m == nil || m.Weight == nil {
		return hiddenState
	}
	return m.Weight.Rows(ctx, hiddenState)
}
