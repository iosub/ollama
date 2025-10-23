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
	mm := []input.Multimodal{{Tensor: visionOutputs, Data: grid}}
	for i := range deepstackVisualEmbeds {
		mm = append(mm, input.Multimodal{Tensor: deepstackVisualEmbeds[i]})
	}

	return mm, nil
}

var (
	tokenVision      int32 = 151655
	tokenVisionStart int32 = 151652
	tokenVisionEnd   int32 = 151653
)

var positions []int32

type modelInput struct {
	*input.Input
	position int32
}

// PostTokenize arranges Qwen 3 VL's inputs for the forward pass
func (m *Model) PostTokenize(inputs []*input.Input) ([]*input.Input, error) {
	positions = make([]int32, 0)
	var result []*input.Input
	
	for i := range inputs {
		if mm := inputs[i].Multimodal; mm != nil {
			t := mm[0].Tensor
			
			// Add vision start token
			result = append(result, &input.Input{Token: tokenVisionStart})
			positions = append(positions, int32(i))
			
			// Add vision tokens
			for j := 0; j < t.Dim(1); j++ {
				result = append(result, &input.Input{Token: tokenVision})
				positions = append(positions, int32(i+1))
			}
			
			// Add vision end token
			result = append(result, &input.Input{Token: tokenVisionEnd})
			positions = append(positions, int32(i+1))
		} else {
			result = append(result, inputs[i])
			positions = append(positions, int32(i))
		}
	}
	
	return result, nil
}

func (m *Model) Forward(ctx ml.Context, batch input.Batch) (ml.Tensor, error) {
	// Create position IDs array
	positionIDs := make([]int32, len(batch.Positions)*4)
	for i, id := range batch.Positions {
		var actualID int32
		if id < int32(len(positions)) {
			actualID = positions[id]
		} else if len(positions) > 0 {
			actualID = id - int32(len(positions)) + positions[len(positions)-1] + 1
		} else {
			actualID = id
		}
		
		positionIDs[i+len(batch.Positions)*0] = actualID
		positionIDs[i+len(batch.Positions)*1] = actualID
		positionIDs[i+len(batch.Positions)*2] = actualID
		positionIDs[i+len(batch.Positions)*3] = 0
	}

	hiddenStates := m.TextModel.TokenEmbedding.Forward(ctx, batch.Inputs).Duplicate(ctx)

	var deepstackVisualEmbeds []ml.Tensor
	for _, mi := range batch.Multimodal {
		visionOutputs := mi.Multimodal[0].Tensor
		ctx.Forward(visionOutputs.Copy(ctx, hiddenStates.View(ctx, mi.Index*hiddenStates.Stride(1), visionOutputs.Dim(0)*visionOutputs.Dim(1))))

		if grid, ok := mi.Multimodal[0].Data.(*Grid); ok {
			for i := 0; i < visionOutputs.Dim(1); i++ {
				w := grid.Width / m.VisionModel.spatialMergeSize
				positionIDs[mi.Index+i+len(batch.Positions)*1] += int32(i / w)
				positionIDs[mi.Index+i+len(batch.Positions)*2] += int32(i % w)
			}
		}

		deepstackVisualEmbeds = make([]ml.Tensor, len(mi.Multimodal[1:]))
		for i, mm := range mi.Multimodal[1:] {
			deepstackVisualEmbeds[i] = ctx.Input().Zeros(mm.Tensor.DType(), hiddenStates.Shape()...)
			ctx.Forward(mm.Tensor.Copy(ctx, deepstackVisualEmbeds[i].View(ctx, mi.Index*deepstackVisualEmbeds[i].Stride(1), mm.Tensor.Dim(0)*mm.Tensor.Dim(1))))
		}
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
