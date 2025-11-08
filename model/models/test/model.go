package qwen3vl

import (
	"bytes"
	"fmt"
	"image"
	"log/slog"
	"slices"

	"github.com/ollama/ollama/fs"
	"github.com/ollama/ollama/kvcache"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn"
	"github.com/ollama/ollama/model"
	"github.com/ollama/ollama/model/input"
)

type Model struct {
	model.Base
	model.TextProcessor

	*TextModel
	*VisionModel `gguf:"v"`

	ImageProcessor

	positionCache []int32
	visionReady   bool
}

func (m *Model) ensureVisionReady() error {
	slog.Debug("ensureVisionReady invoked", "visionReady", m.visionReady)

	if m.visionReady {
		return nil
	}

	if m.VisionModel == nil {
		slog.Debug("ensureVisionReady missing VisionModel")
		return model.ErrNoVisionModel
	}

	backend := m.Backend()
	if backend == nil {
		slog.Debug("ensureVisionReady missing backend")
		return model.ErrNoVisionModel
	}

	vm := m.VisionModel

	if vm.PatchEmbedding == nil {
		vm.PatchEmbedding = &nn.Conv3D{}
	}

	if vm.PatchEmbedding.Weight == nil {
		vm.PatchEmbedding.Weight = backend.Get("v.patch_embed.weight")
		if vm.PatchEmbedding.Weight == nil {
			vm.PatchEmbedding.Weight = backend.Get("v.patch_embd.weight")
		}
	}

	if vm.PatchEmbedding.Weight == nil {
		slog.Debug("ensureVisionReady missing patch embedding weight")
		return model.ErrNoVisionModel
	}
	if vm.PatchEmbedding.Weight1 == nil {
		vm.PatchEmbedding.Weight1 = backend.Get("v.patch_embed.weight.1")
		if vm.PatchEmbedding.Weight1 == nil {
			vm.PatchEmbedding.Weight1 = backend.Get("v.patch_embd.weight.1")
		}
	}

	// Deduce temporal_patch_size from weight shape: [patchSize, patchSize, temporalPatchSize, channels]
	// Weight.1 presence indicates split GGUF format but doesn't change the deduced value
	weightShape := vm.PatchEmbedding.Weight.Shape()
	if len(weightShape) == 4 && weightShape[2] > 0 {
		deducedTemporalPatchSize := weightShape[2]
		if vm.PatchEmbedding.Weight1 != nil {
			slog.Debug("Detected dual-weight PatchEmbedding (split GGUF format)",
				"weight_shape", vm.PatchEmbedding.Weight.Shape(),
				"weight1_shape", vm.PatchEmbedding.Weight1.Shape(),
				"temporal_patch_size", deducedTemporalPatchSize)
		} else {
			slog.Debug("Deduced temporal_patch_size from PatchEmbedding.Weight shape",
				"temporal_patch_size", deducedTemporalPatchSize,
				"weight_shape", weightShape)
		}
		vm.temporalPatchSize = deducedTemporalPatchSize
	}

	if vm.PatchEmbedding.Bias == nil {
		vm.PatchEmbedding.Bias = backend.Get("v.patch_embed.bias")
		if vm.PatchEmbedding.Bias == nil {
			vm.PatchEmbedding.Bias = backend.Get("v.patch_embd.bias")
		}
	}

	if vm.PositionEmbedding == nil {
		vm.PositionEmbedding = &VisionPositionEmbedding{}
	}
	if vm.PositionEmbedding.PositionEmbedding == nil {
		pos := backend.Get("v.pos_embed.weight")
		if pos == nil {
			pos = backend.Get("v.position_embd.weight")
		}
		if pos == nil {
			slog.Debug("ensureVisionReady missing position embedding")
			return model.ErrNoVisionModel
		}
		vm.PositionEmbedding.PositionEmbedding = &nn.Embedding{Weight: pos}
	}

	if vm.PatchMerger == nil {
		vm.PatchMerger = &VisionPatchMerger{}
	}
	ensureMainMerger := func() error {
		if vm.PatchMerger.Norm == nil {
			vm.PatchMerger.Norm = &nn.LayerNorm{}
		}
		if vm.PatchMerger.Norm.Weight == nil {
			vm.PatchMerger.Norm.Weight = backend.Get("v.merger.norm.weight")
			if vm.PatchMerger.Norm.Weight == nil {
				vm.PatchMerger.Norm.Weight = backend.Get("v.post_ln.weight")
				if vm.PatchMerger.Norm.Weight != nil {
					slog.Debug("Found PatchMerger.Norm.Weight in split GGUF format", "tensor", "v.post_ln.weight")
				}
			}
		}
		if vm.PatchMerger.Norm.Bias == nil {
			vm.PatchMerger.Norm.Bias = backend.Get("v.merger.norm.bias")
			if vm.PatchMerger.Norm.Bias == nil {
				vm.PatchMerger.Norm.Bias = backend.Get("v.post_ln.bias")
				if vm.PatchMerger.Norm.Bias != nil {
					slog.Debug("Found PatchMerger.Norm.Bias in split GGUF format", "tensor", "v.post_ln.bias")
				}
			}
		}
		if vm.PatchMerger.Norm.Weight == nil {
			// PatchMerger is optional in split GGUF models
			slog.Debug("PatchMerger.Norm.Weight not found - trying split GGUF format (may be in mm.* tensors)")
		} else {
			slog.Debug("PatchMerger.Norm loaded successfully")
		}

		if vm.PatchMerger.FC1 == nil {
			vm.PatchMerger.FC1 = &nn.Linear{}
		}
		if vm.PatchMerger.FC1.Weight == nil {
			vm.PatchMerger.FC1.Weight = backend.Get("v.merger.linear_fc1.weight")
			if vm.PatchMerger.FC1.Weight == nil {
				vm.PatchMerger.FC1.Weight = backend.Get("mm.0.weight")
				if vm.PatchMerger.FC1.Weight != nil {
					slog.Debug("Found PatchMerger.FC1.Weight in split GGUF format", "tensor", "mm.0.weight")
				}
			}
		}
		if vm.PatchMerger.FC1.Bias == nil {
			vm.PatchMerger.FC1.Bias = backend.Get("v.merger.linear_fc1.bias")
			if vm.PatchMerger.FC1.Bias == nil {
				vm.PatchMerger.FC1.Bias = backend.Get("mm.0.bias")
				if vm.PatchMerger.FC1.Bias != nil {
					slog.Debug("Found PatchMerger.FC1.Bias in split GGUF format", "tensor", "mm.0.bias")
				}
			}
		}
		if vm.PatchMerger.FC1.Weight == nil {
			slog.Debug("PatchMerger.FC1.Weight not found after trying v.merger.* and mm.0.*")
		} else {
			slog.Debug("PatchMerger.FC1 loaded successfully")
		}

		if vm.PatchMerger.FC2 == nil {
			vm.PatchMerger.FC2 = &nn.Linear{}
		}
		if vm.PatchMerger.FC2.Weight == nil {
			vm.PatchMerger.FC2.Weight = backend.Get("v.merger.linear_fc2.weight")
			if vm.PatchMerger.FC2.Weight == nil {
				vm.PatchMerger.FC2.Weight = backend.Get("mm.2.weight")
				if vm.PatchMerger.FC2.Weight != nil {
					slog.Debug("Found PatchMerger.FC2.Weight in split GGUF format", "tensor", "mm.2.weight")
				}
			}
		}
		if vm.PatchMerger.FC2.Bias == nil {
			vm.PatchMerger.FC2.Bias = backend.Get("v.merger.linear_fc2.bias")
			if vm.PatchMerger.FC2.Bias == nil {
				vm.PatchMerger.FC2.Bias = backend.Get("mm.2.bias")
				if vm.PatchMerger.FC2.Bias != nil {
					slog.Debug("Found PatchMerger.FC2.Bias in split GGUF format", "tensor", "mm.2.bias")
				}
			}
		}
		if vm.PatchMerger.FC2.Weight == nil {
			slog.Debug("PatchMerger.FC2.Weight not found after trying v.merger.* and mm.2.*")
		} else {
			slog.Debug("PatchMerger.FC2 loaded successfully")
		}

		return nil
	}

	if err := ensureMainMerger(); err != nil {
		return err
	}

	if len(vm.deepstackVisualIndexes) == len(vm.DeepstackMerger) {
		for i := range vm.DeepstackMerger {
			if vm.DeepstackMerger[i] == nil {
				vm.DeepstackMerger[i] = &VisionPatchMerger{}
			}

			prefix := fmt.Sprintf("v.deepstack.%d", vm.deepstackVisualIndexes[i])
			merger := vm.DeepstackMerger[i]

			if merger.Norm == nil {
				merger.Norm = &nn.LayerNorm{}
			}
			if merger.Norm.Weight == nil {
				merger.Norm.Weight = backend.Get(prefix + ".norm.weight")
			}
			if merger.Norm.Bias == nil {
				merger.Norm.Bias = backend.Get(prefix + ".norm.bias")
			}

			if merger.FC1 == nil {
				merger.FC1 = &nn.Linear{}
			}
			if merger.FC1.Weight == nil {
				merger.FC1.Weight = backend.Get(prefix + ".fc1.weight")
			}
			if merger.FC1.Bias == nil {
				merger.FC1.Bias = backend.Get(prefix + ".fc1.bias")
			}

			if merger.FC2 == nil {
				merger.FC2 = &nn.Linear{}
			}
			if merger.FC2.Weight == nil {
				merger.FC2.Weight = backend.Get(prefix + ".fc2.weight")
			}
			if merger.FC2.Bias == nil {
				merger.FC2.Bias = backend.Get(prefix + ".fc2.bias")
			}

			if merger.Norm.Weight == nil || merger.FC1.Weight == nil || merger.FC2.Weight == nil {
				slog.Debug("ensureVisionReady deepstack merger incomplete", "index", vm.deepstackVisualIndexes[i])
				return model.ErrNoVisionModel
			}
		}
	}

	m.visionReady = true
	slog.Debug("ensureVisionReady completed", "deepstack_mergers", len(vm.DeepstackMerger))
	return nil
}

func (m *Model) EncodeMultimodal(ctx ml.Context, multimodalData []byte) ([]input.Multimodal, error) {
	slog.Debug("EncodeMultimodal invoked", "data_bytes", len(multimodalData))
	if len(m.VisionModel.Layers) == 0 {
		slog.Debug("EncodeMultimodal no vision layers")
		return nil, model.ErrNoVisionModel
	}

	if err := m.ensureVisionReady(); err != nil {
		slog.Debug("EncodeMultimodal ensureVisionReady failed", "error", err)
		return nil, err
	}

	img, _, err := image.Decode(bytes.NewReader(multimodalData))
	if err != nil {
		slog.Debug("EncodeMultimodal image decode failed", "error", err)
		return nil, err
	}

	pixelValues, grid, err := m.ProcessImage(ctx, img)
	if err != nil {
		slog.Debug("EncodeMultimodal process image failed", "error", err)
		return nil, err
	}

	// Calculate tensor dimensions
	visionOutputs, deepstackVisualEmbeds := m.VisionModel.Forward(ctx, pixelValues, grid)
	slog.Debug("EncodeMultimodal vision forward complete", "patches", visionOutputs.Dim(1), "deepstack", len(deepstackVisualEmbeds))
	mm := []input.Multimodal{{Tensor: visionOutputs, Data: grid}}
	for i := range deepstackVisualEmbeds {
		mm = append(mm, input.Multimodal{Tensor: deepstackVisualEmbeds[i]})
	}

	slog.Debug("EncodeMultimodal returning embeddings", "modalities", len(mm))
	return mm, nil
}

var (
	tokenVision      int32 = 151655
	tokenVisionStart int32 = 151652
	tokenVisionEnd   int32 = 151653
)

type modelInput struct {
	*input.Input
	position int32
}

// PostTokenize arranges Qwen 3 VL's inputs for the forward pass
func (m *Model) PostTokenize(inputs []*input.Input) ([]*input.Input, error) {
	slog.Debug("PostTokenize invoked", "inputs", len(inputs))
	m.positionCache = m.positionCache[:0]
	return slices.Collect(func(yield func(*input.Input) bool) {
		for i := range inputs {
			s := []modelInput{{Input: inputs[i]}}
			if mm := inputs[i].Multimodal; mm != nil {
				slog.Debug("PostTokenize processing multimodal entry", "index", i, "tensors", len(mm))
				t := mm[0].Tensor
				s = slices.Repeat([]modelInput{
					{
						position: int32(i + 1),
						Input:    &input.Input{Token: tokenVision},
					},
				}, t.Dim(1)+1+1)

				s[0] = modelInput{
					Input:    &input.Input{Token: tokenVisionStart},
					position: int32(i),
				}

				s[len(s)-1] = modelInput{
					Input:    &input.Input{Token: tokenVisionEnd},
					position: int32(i + mm[0].Data.(*Grid).Width/m.spatialMergeSize + 1),
				}

				s[1] = modelInput{
					Input: &input.Input{
						Token:          tokenVision,
						Multimodal:     inputs[i].Multimodal,
						MultimodalHash: inputs[i].MultimodalHash,
						SameBatch:      t.Dim(1),
					},
					position: int32(i + 1),
				}
			}

			for _, e := range s {
				position := e.position
				if position == 0 && len(m.positionCache) > 0 {
					position = m.positionCache[len(m.positionCache)-1] + 1
				}

				m.positionCache = append(m.positionCache, position)
				if !yield(e.Input) {
					return
				}
			}
		}
	}), nil
}

func (m *Model) Forward(ctx ml.Context, batch input.Batch) (ml.Tensor, error) {
	// ggml mrope requires 4 positions per token: [time, height, width, extra]
	positionSlice := slices.Collect(makeSlice2D[int32](4, len(batch.Positions)))
	for i, id := range batch.Positions {
		if id < int32(len(m.positionCache)) {
			id = m.positionCache[id]
		} else if len(m.positionCache) > 0 {
			id = id - int32(len(m.positionCache)) + m.positionCache[len(m.positionCache)-1] + 1
		}

		positionSlice[0][i] = id
		positionSlice[1][i] = id
		positionSlice[2][i] = id
		// positionSlice[3] is intentionally left as zeros
	}

	hiddenStates := m.TextModel.TokenEmbedding.Forward(ctx, batch.Inputs).Duplicate(ctx)

	var deepstackVisualEmbeds []ml.Tensor
	for _, mi := range batch.Multimodal {
		visionOutputs := mi.Multimodal[0].Tensor
		embedDim := visionOutputs.Dim(0)
		patchCount := visionOutputs.Dim(1)
		hiddenShape := hiddenStates.Shape()
		hiddenStride0 := hiddenStates.Stride(0)
		hiddenStride1 := hiddenStates.Stride(1)
		slog.Debug("Vision embedding copy", "index", mi.Index, "hidden_rows", hiddenShape[0], "hidden_cols", hiddenShape[1], "stride0", hiddenStride0, "stride1", hiddenStride1, "vision_rows", embedDim, "vision_cols", patchCount)
		startColumn := mi.Index + 1 // skip <vision_start>
		offset := startColumn * hiddenStride1
		targetView := hiddenStates.View(ctx, offset, embedDim, hiddenStride1, patchCount)
		targetView = targetView.Contiguous(ctx)
		ctx.Forward(targetView.Set(ctx, visionOutputs, 0))

		if grid, ok := mi.Multimodal[0].Data.(*Grid); ok {
			patchesPerRow := grid.Width / m.spatialMergeSize
			if patchesPerRow > 0 {
				tokenBase := startColumn
				maxPatches := visionOutputs.Dim(1)
				for patch := 0; patch < maxPatches; patch++ {
					tokenIdx := tokenBase + patch
					if tokenIdx >= len(positionSlice[0]) {
						break
					}
					row := int32(patch / patchesPerRow)
					col := int32(patch % patchesPerRow)
					positionSlice[1][tokenIdx] += row
					positionSlice[2][tokenIdx] += col
					if patch == 0 {
						slog.Debug("Vision position assignment", "token_index", tokenIdx, "row", row, "col", col, "patches_per_row", patchesPerRow, "patch_count", maxPatches)
					}
				}
			}
		}

		deepstackVisualEmbeds = make([]ml.Tensor, len(mi.Multimodal[1:]))
		for i, mm := range mi.Multimodal[1:] {
			deepstackVisualEmbeds[i] = ctx.Input().Zeros(mm.Tensor.DType(), hiddenStates.Shape()...)
			embedDim := mm.Tensor.Dim(0)
			patchCount := mm.Tensor.Dim(1)
			startColumn := mi.Index + 1
			offset := startColumn * deepstackVisualEmbeds[i].Stride(1)
			targetView := deepstackVisualEmbeds[i].View(ctx, offset, embedDim, deepstackVisualEmbeds[i].Stride(1), patchCount)
			targetView = targetView.Contiguous(ctx)
			ctx.Forward(targetView.Set(ctx, mm.Tensor, 0))
		}
	}

	positions := ctx.Input().FromInts(slices.Concat(positionSlice...), len(positionSlice[0])*len(positionSlice))
	for i, layer := range m.TextModel.Layers {
		if m.Cache != nil {
			m.Cache.SetLayer(i)
		}

		var outputs ml.Tensor
		if i == len(m.TextModel.Layers)-1 {
			outputs = batch.Outputs
		}

		hiddenStates = layer.Forward(ctx, hiddenStates, positions, outputs, m.Cache, m.Options)
		if i < len(deepstackVisualEmbeds) {
			hiddenStates = hiddenStates.Add(ctx, deepstackVisualEmbeds[i])
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

	m.Cache = kvcache.NewCausalCache(func(ctx ml.Context, layer int, key, positions ml.Tensor) (ml.Tensor, error) {
		m.positionCache = nil
		positions = positions.Repeat(ctx, 1, 4).Reshape(ctx, -1)
		return m.Options.applyRotaryPositionalEmbedding(ctx, key, positions), nil
	})
	return &m, nil
}

func init() {
	model.Register("qwen3vl", New)
	model.Register("qwen3vlmoe", New)
}
