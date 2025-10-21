package qwen3vl

import (
	"bytes"
	"image"
	"slices"

	"github.com/ollama/ollama/fs"
	"github.com/ollama/ollama/kvcache"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/model"
	"github.com/ollama/ollama/model/input"
)

type Model struct {
	model.Base
	model.TextProcessor

	*TextModel
	*VisionModel `gguf:"v"`

	ImageProcessor
}

func (m *Model) EncodeMultimodal(ctx ml.Context, multimodalData []byte) ([]input.Multimodal, error) {
	if len(m.VisionModel.Layers) == 0 {
		return nil, model.ErrNoVisionModel
	}

	img, _, err := image.Decode(bytes.NewReader(multimodalData))
	if err != nil {
		return nil, err
	}

	pixelValues, grid, err := m.ProcessImage(ctx, img)
	if err != nil {
		return nil, err
	}

	// Calculate tensor dimensions
	visionOutputs, deepstackVisualEmbeds := m.VisionModel.Forward(ctx, pixelValues, grid)
	mm := []input.Multimodal{{Tensor: visionOutputs}}
	for i := range deepstackVisualEmbeds {
		mm = append(mm, input.Multimodal{Tensor: deepstackVisualEmbeds[i]})
	}

	return mm, nil
}

var (
	inputTokenImagePad    = &input.Input{Token: 151655}
	inputTokenVisionStart = &input.Input{Token: 151652}
	inputTokenVisionEnd   = &input.Input{Token: 151653}
)

// PostTokenize arranges Qwen 3 VL's inputs for the forward pass
func (m *Model) PostTokenize(inputs []*input.Input) ([]*input.Input, error) {
	return slices.Collect(func(yield func(*input.Input) bool) {
		for i := range inputs {
			s := []*input.Input{inputs[i]}
			if inputs[i].Multimodal != nil {
				t := inputs[i].Multimodal[0].Tensor
				s = slices.Repeat([]*input.Input{inputTokenImagePad}, t.Dim(1)+1+1)
				s[0] = inputTokenVisionStart
				s[len(s)-1] = inputTokenVisionEnd
				s[1] = &input.Input{
					Token:          inputTokenImagePad.Token,
					Multimodal:     inputs[i].Multimodal,
					MultimodalHash: inputs[i].MultimodalHash,
					SameBatch:      t.Dim(1),
				}
			}

			for _, e := range s {
				if !yield(e) {
					return
				}
			}
		}
	}), nil
}

func (m *Model) Forward(ctx ml.Context, batch input.Batch) (ml.Tensor, error) {
	hiddenStates := m.TextModel.TokenEmbedding.Forward(ctx, batch.Inputs).Duplicate(ctx)

	var deepstackVisualEmbeds []ml.Tensor
	for _, mi := range batch.Multimodal {
		visionOutputs := mi.Multimodal[0].Tensor
		for _, mm := range mi.Multimodal[1:] {
			deepstackVisualEmbeds = append(deepstackVisualEmbeds, mm.Tensor)
		}
		ctx.Forward(visionOutputs.Copy(ctx, hiddenStates.View(ctx, mi.Index*hiddenStates.Stride(1), visionOutputs.Dim(0)*visionOutputs.Dim(1))))
	}

	positionIDs := make([]int32, len(batch.Positions)*4)
	for i, id := range batch.Positions {
		positionIDs[i+len(batch.Positions)*0] = id
		positionIDs[i+len(batch.Positions)*1] = id
		positionIDs[i+len(batch.Positions)*2] = id
		positionIDs[i+len(batch.Positions)*3] = 0
	}

	positions := ctx.Input().FromIntSlice(positionIDs, len(positionIDs))
	for i, layer := range m.TextModel.Layers {
		if m.Cache != nil {
			m.Cache.SetLayer(i)
		}

		var outputs ml.Tensor
		if i == len(m.TextModel.Layers)-1 {
			outputs = batch.Outputs
		}

		hiddenStates = layer.Forward(ctx, hiddenStates, positions, outputs, m.TextModel.Cache, m.TextModel.Options)
		if j := slices.Index(m.deepstackVisualIndexes, int32(i)); len(deepstackVisualEmbeds) > 0 && j >= 0 {
			visualEmbeds := deepstackVisualEmbeds[j].Pad(ctx, 0, hiddenStates.Dim(1)-deepstackVisualEmbeds[j].Dim(1), 0, 0)
			hiddenStates = hiddenStates.Add(ctx, visualEmbeds)
		}
	}

	hiddenStates = m.OutputNorm.Forward(ctx, hiddenStates, 1e-06)
	return m.Output.Forward(ctx, hiddenStates), nil
}

func New(c fs.Config) (model.Model, error) {
	m := Model{
		TextProcessor: model.NewBytePairEncoding(
			&model.Vocabulary{
				Values: c.Strings("tokenizer.ggml.tokens"),
				Types:  c.Ints("tokenizer.ggml.token_type"),
				Merges: c.Strings("tokenizer.ggml.merges"),
				AddBOS: c.Bool("tokenizer.ggml.add_bos_token", false),
				BOS:    []int32{int32(c.Uint("tokenizer.ggml.bos_token_id"))},
				AddEOS: c.Bool("tokenizer.ggml.add_eos_token", false),
				EOS: append(
					[]int32{int32(c.Uint("tokenizer.ggml.eos_token_id"))},
					c.Ints("tokenizer.ggml.eos_token_ids")...,
				),
			},
			`(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`,
		),
		TextModel:      newTextModel(c),
		VisionModel:    newVisionModel(c),
		ImageProcessor: newImageProcessor(c),
	}

	m.Cache = kvcache.NewCausalCache(m.Shift)
	return &m, nil
}

func init() {
	model.Register("qwen3vl", New)
	model.Register("qwen3vlmoe", New)
}
