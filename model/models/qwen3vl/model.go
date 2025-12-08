package qwen3vl

import (
	"bytes"
	"context"
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

	// Split vision model support
	visionReady   bool       // true if vision encoder is ready (either loaded from main file or separate file)
	visionPath    string     // path to separate vision GGUF file (empty if embedded in main file)
	visionBackend ml.Backend // backend for vision model when loaded separately
}

// HasProjector returns true if the model has a vision projector (vision capability)
// HasProjector checks if the vision encoder is actually loaded with tensors.
// For split models, this returns false until the vision GGUF is loaded.
func (m *Model) HasProjector() bool {
	// Check if a critical vision tensor is actually populated
	// Unified models use Conv3D (PatchEmbedding), split models use Conv2D (PatchEmbedding2D)
	return m.VisionModel != nil && (m.VisionModel.PatchEmbedding != nil || m.VisionModel.PatchEmbedding2D != nil)
}

// ensureVisionReady loads the vision encoder if it hasn't been loaded yet.
// For split GGUF models, this loads vision tensor data from the separate file
// into the main backend's pre-allocated tensors using LoadSecondary.
// For unified models, this just marks vision as ready.
func (m *Model) ensureVisionReady() error {
	if m.visionReady {
		return nil
	}

	slog.Debug("ensureVisionReady", "hasProjector", m.HasProjector(), "visionPath", m.visionPath)

	// If vision layers are already populated (unified model), mark as ready
	if m.HasProjector() {
		// Infer correct vision dimensions from actual tensor shapes (fixes incorrect config defaults)
		m.VisionModel.InferOptionsFromTensors()

		// MANUAL LOADING: DeepstackMerger FC weights
		// vision_bridge cannot populate FC weights because array indices (0,1,2) don't match GGUF layer IDs (8,16,24)
		// vision_bridge uses strconv.Itoa(j) which builds "v.deepstack.0.fc1" but GGUF has "v.deepstack.8.fc1.weight"
		if m.VisionModel.DeepstackMerger != nil && len(m.VisionModel.DeepstackMerger) >= 3 {
			layerIDs := []int{8, 16, 24}
			for idx, layerID := range layerIDs {
				prefix := fmt.Sprintf("v.deepstack.%d", layerID)

				// Try to find FC1 weight tensor
				fc1WeightName := prefix + ".fc1.weight"
				if fc1Weight := m.Backend().Get(fc1WeightName); fc1Weight != nil {
					fc1BiasName := prefix + ".fc1.bias"
					fc1Bias := m.Backend().Get(fc1BiasName) // May be nil
					m.VisionModel.DeepstackMerger[idx].FC1 = &nn.Linear{
						Weight: fc1Weight,
						Bias:   fc1Bias,
					}
					slog.Info("Manually loaded DeepstackMerger FC1", "layer", layerID, "idx", idx, "shape", fc1Weight.Shape())
				} else {
					slog.Debug("DeepstackMerger FC1 weight not found (expected for non-deepstack models)", "layer", layerID, "name", fc1WeightName)
				}

				// Try to find FC2 weight tensor
				fc2WeightName := prefix + ".fc2.weight"
				if fc2Weight := m.Backend().Get(fc2WeightName); fc2Weight != nil {
					fc2BiasName := prefix + ".fc2.bias"
					fc2Bias := m.Backend().Get(fc2BiasName) // May be nil
					m.VisionModel.DeepstackMerger[idx].FC2 = &nn.Linear{
						Weight: fc2Weight,
						Bias:   fc2Bias,
					}
					slog.Info("Manually loaded DeepstackMerger FC2", "layer", layerID, "idx", idx, "shape", fc2Weight.Shape())
				} else {
					slog.Debug("DeepstackMerger FC2 weight not found (expected for non-deepstack models)", "layer", layerID, "name", fc2WeightName)
				}
			}
		}

		// Sync temporalPatchSize from VisionModel to ImageProcessor (split model may have different value)
		m.ImageProcessor.temporalPatchSize = m.VisionModel.temporalPatchSize
		m.ImageProcessor.storagePatchSize = m.VisionModel.storagePatchSize
		slog.Debug("Vision ready", "hiddenSize", m.VisionModel.hiddenSize, "numHeads", m.VisionModel.numHeads, "layers", len(m.VisionModel.Layers), "isSplitArchitecture", m.VisionModel.isSplitArchitecture, "temporalPatchSize", m.VisionModel.temporalPatchSize)
		m.visionReady = true
		return nil
	}

	// If no vision path specified, vision is not available
	if m.visionPath == "" {
		return model.ErrNoVisionModel
	}

	// CRITICAL: Register tensor name aliases BEFORE LoadSecondary!
	// Aliases must be active when tensors are loaded so they map to correct struct fields.
	// Split models (e.g., unsloth) use different naming conventions than the Go struct.
	slog.Info("Registering split model tensor aliases BEFORE load")

	// Embedding tensors: patch_embd → patch_embed, position_embd → position_embed
	m.Backend().RegisterTensorAlias("v.patch_embed", "v.patch_embd")
	m.Backend().RegisterTensorAlias("v.position_embed", "v.position_embd")

	// Layer norm tensors: ln1 → norm1, ln2 → norm2
	// Split GGUF has v.blk.0.ln1.weight, model expects v.blk.0.norm1.weight
	m.Backend().RegisterTensorAlias("v.blk.*.norm1", "v.blk.*.ln1")
	m.Backend().RegisterTensorAlias("v.blk.*.norm2", "v.blk.*.ln2")

	// MLP tensors: ffn_up → mlp.linear_fc1, ffn_down → mlp.linear_fc2
	// Split GGUF has v.blk.0.ffn_up.weight, model expects v.blk.0.mlp.linear_fc1.weight
	m.Backend().RegisterTensorAlias("v.blk.*.mlp.linear_fc1", "v.blk.*.ffn_up")
	m.Backend().RegisterTensorAlias("v.blk.*.mlp.linear_fc2", "v.blk.*.ffn_down")

	// Deepstack merger tensors: deepstack → deepstack_merger
	m.Backend().RegisterTensorAlias("v.deepstack_merger", "v.deepstack")

	slog.Info("Split model tensor aliases registered, now loading secondary GGUF")

	// Load vision tensor data from separate GGUF file into main backend
	// LoadSecondary creates tensors that don't exist and loads data into them
	slog.Info("Loading split vision model into main backend", "path", m.visionPath)

	err := m.Backend().LoadSecondary(context.Background(), m.visionPath, nil)
	if err != nil {
		slog.Error("Failed to load vision model from secondary GGUF", "error", err)
		return err
	}

	slog.Info("Split vision model loaded from GGUF, re-populating struct")

	// IMPORTANT: After LoadSecondary, tensors exist in backend but VisionModel struct
	// fields haven't been bound to them. Re-populate the VisionModel field.
	// The "v" tag corresponds to `gguf:"v"` on the VisionModel field.
	slog.Debug("Re-populating VisionModel struct after LoadSecondary")
	if m.VisionModel == nil {
		m.VisionModel = newVisionModel(m.Backend().Config())
	}

	// CRITICAL: vision_bridge needs pre-initialized DeepstackMerger array
	// For Qwen3-VL Split, we know there are 3 deepstack layers (8, 16, 24)
	// This matches llama.cpp's n_deepstack_layers = 3
	nDeepstack := 3
	slog.Info("Pre-initializing DeepstackMerger array for Split GGUF", "n_deep stack", nDeepstack)
	m.VisionModel.DeepstackMerger = make([]*VisionPatchMerger, nDeepstack)
	for i := range m.VisionModel.DeepstackMerger {
		m.VisionModel.DeepstackMerger[i] = &VisionPatchMerger{}
	}

	model.RepopulateField(m.Base, m.VisionModel, "v")

	// MANUAL LOADING: DeepstackMerger FC weights
	// vision_bridge cannot populate FC weights because array indices (0,1,2) don't match GGUF layer IDs (8,16,24)
	// vision_bridge uses strconv.Itoa(j) which builds "v.deepstack.0.fc1" but GGUF has "v.deepstack.8.fc1.weight"
	layerIDs := []int{8, 16, 24}
	for idx, layerID := range layerIDs {
		prefix := fmt.Sprintf("v.deepstack.%d", layerID)

		// Try to find FC1 weight tensor
		fc1WeightName := prefix + ".fc1.weight"
		if fc1Weight := m.Backend().Get(fc1WeightName); fc1Weight != nil {
			fc1BiasName := prefix + ".fc1.bias"
			fc1Bias := m.Backend().Get(fc1BiasName) // May be nil
			m.VisionModel.DeepstackMerger[idx].FC1 = &nn.Linear{
				Weight: fc1Weight,
				Bias:   fc1Bias,
			}
			slog.Info("Manually loaded DeepstackMerger FC1", "layer", layerID, "idx", idx, "shape", fc1Weight.Shape())
		} else {
			slog.Warn("DeepstackMerger FC1 weight not found", "layer", layerID, "name", fc1WeightName)
		}

		// Try to find FC2 weight tensor
		fc2WeightName := prefix + ".fc2.weight"
		if fc2Weight := m.Backend().Get(fc2WeightName); fc2Weight != nil {
			fc2BiasName := prefix + ".fc2.bias"
			fc2Bias := m.Backend().Get(fc2BiasName) // May be nil
			m.VisionModel.DeepstackMerger[idx].FC2 = &nn.Linear{
				Weight: fc2Weight,
				Bias:   fc2Bias,
			}
			slog.Info("Manually loaded DeepstackMerger FC2", "layer", layerID, "idx", idx, "shape", fc2Weight.Shape())
		} else {
			slog.Warn("DeepstackMerger FC2 weight not found", "layer", layerID, "name", fc2WeightName)
		}
	}

	// Infer correct vision dimensions from actual tensor shapes
	m.VisionModel.InferOptionsFromTensors()

	// Verify that the vision model is now ready
	if !m.HasProjector() {
		slog.Error("Vision tensors still not populated after LoadSecondary and RepopulateField",
			"patchEmbedding", m.VisionModel.PatchEmbedding,
			"layers", len(m.VisionModel.Layers))
		return model.ErrNoVisionModel
	}

	m.visionReady = true
	slog.Info("Split vision model loaded", "layers", len(m.VisionModel.Layers), "hiddenSize", m.VisionModel.hiddenSize)
	return nil
}

func (m *Model) EncodeMultimodal(ctx ml.Context, multimodalData []byte) ([]input.Multimodal, error) {
	// Lazy load vision encoder if needed (supports split GGUF models)
	if err := m.ensureVisionReady(); err != nil {
		return nil, err
	}

	if !m.HasProjector() {
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

	// Qwen3-VL requires concatenating main + deepstack embeddings into single expanded tensor
	// Format: [main (n_embd) | deepstack_0 (n_embd) | deepstack_1 (n_embd) | deepstack_2 (n_embd)]
	// This creates n_embd_inp = n_embd * (1 + n_deepstack_layers)
	if len(deepstackVisualEmbeds) > 0 {
		// Concatenate along the feature dimension (dim=1)
		allEmbeds := []ml.Tensor{visionOutputs}
		allEmbeds = append(allEmbeds, deepstackVisualEmbeds...)

		// Concatenated shape: [n_tokens, n_embd * (1 + n_deepstack)]
		concatenated := allEmbeds[0].Concat(ctx, allEmbeds[1], 1)
		for i := 2; i < len(allEmbeds); i++ {
			concatenated = concatenated.Concat(ctx, allEmbeds[i], 1)
		}
		slog.Info("Concatenated vision + deepstack embeddings",
			"main_shape", visionOutputs.Shape(),
			"n_deepstack", len(deepstackVisualEmbeds),
			"concatenated_shape", concatenated.Shape())

		return []input.Multimodal{{Tensor: concatenated, Data: grid}}, nil
	}

	// No deepstack - return main vision output only
	return []input.Multimodal{{Tensor: visionOutputs, Data: grid}}, nil
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
	m.positionCache = m.positionCache[:0]
	return slices.Collect(func(yield func(*input.Input) bool) {
		for i := range inputs {
			s := []modelInput{{Input: inputs[i]}}
			if mm := inputs[i].Multimodal; mm != nil {
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

		// Check if embeddings are concatenated (n_embd_inp format)
		nEmbdFull := visionOutputs.Dim(1)
		nEmbd := m.TextModel.Options.hiddenSize
		if nEmbdFull > nEmbd && nEmbdFull%nEmbd == 0 {
			// Split concatenated embeddings: [main | deepstack_0 | deepstack_1 | ...]
			nDeepstackLayers := nEmbdFull/nEmbd - 1
			slog.Info("Detected concatenated vision embeddings - splitting",
				"full_dim", nEmbdFull, "n_embd", nEmbd, "n_deepstack", nDeepstackLayers)

			// Extract main vision (first n_embd columns)
			mainVision := visionOutputs.View(ctx, 0, visionOutputs.Dim(0)*nEmbd).Reshape(ctx, nEmbd, visionOutputs.Dim(0))

			// Extract deepstack features
			if nDeepstackLayers > 0 && deepstackVisualEmbeds == nil {
				deepstackVisualEmbeds = make([]ml.Tensor, nDeepstackLayers)
				for i := 0; i < nDeepstackLayers; i++ {
					offset := (i + 1) * nEmbd * visionOutputs.Dim(0)
					deepstackVisualEmbeds[i] = visionOutputs.View(ctx, offset, visionOutputs.Dim(0)*nEmbd).Reshape(ctx, nEmbd, visionOutputs.Dim(0))
				}
			}
			visionOutputs = mainVision
		}

		// Copy main vision embeddings into hiddenStates
		ctx.Forward(visionOutputs.Copy(ctx, hiddenStates.View(ctx, mi.Index*hiddenStates.Stride(1), visionOutputs.Dim(0)*visionOutputs.Dim(1))))

		if grid, ok := mi.Multimodal[0].Data.(*Grid); ok {
			w := grid.Width / m.spatialMergeSize
			h := grid.Height / m.spatialMergeSize
			// Get the temporal base position (same for all image tokens)
			temporalBase := positionSlice[0][mi.Index]
			slog.Info("M-RoPE image position adjustment",
				"grid.Width", grid.Width, "grid.Height", grid.Height,
				"spatialMergeSize", m.spatialMergeSize, "w", w, "h", h,
				"mi.Index", mi.Index, "numImageTokens", visionOutputs.Dim(1),
				"temporalBase", temporalBase)

			// For M-RoPE, image tokens need proper 2D position encoding:
			// pos[0] = temporal (constant for all image tokens)
			// pos[1] = temporal + row (y coordinate)
			// pos[2] = temporal + column (x coordinate)
			for i := range visionOutputs.Dim(1) {
				row := int32(i / w)
				col := int32(i % w)
				// pos[0] stays as temporal base (already set from positionCache)
				positionSlice[1][mi.Index+i] = temporalBase + row
				positionSlice[2][mi.Index+i] = temporalBase + col
			}
			// Log sample positions after adjustment
			if visionOutputs.Dim(1) >= 3 {
				slog.Info("M-RoPE sample positions after adjustment",
					"pos[0][0,1,2]", []int32{positionSlice[0][mi.Index], positionSlice[0][mi.Index+1], positionSlice[0][mi.Index+2]},
					"pos[1][0,1,2]", []int32{positionSlice[1][mi.Index], positionSlice[1][mi.Index+1], positionSlice[1][mi.Index+2]},
					"pos[2][0,1,2]", []int32{positionSlice[2][mi.Index], positionSlice[2][mi.Index+1], positionSlice[2][mi.Index+2]})
			}
		}

		deepstackVisualEmbeds = make([]ml.Tensor, len(mi.Multimodal[1:]))
		for i, mm := range mi.Multimodal[1:] {
			deepstackVisualEmbeds[i] = ctx.Input().Zeros(mm.Tensor.DType(), hiddenStates.Shape()...)
			ctx.Forward(mm.Tensor.Copy(ctx, deepstackVisualEmbeds[i].View(ctx, mi.Index*deepstackVisualEmbeds[i].Stride(1), mm.Tensor.Dim(0)*mm.Tensor.Dim(1))))
		}
	}

	slog.Info("M-RoPE total positions",
		"numPositions", len(positionSlice[0]),
		"mropeSections", m.Options.mropeSections,
		"samplePos0_first3", positionSlice[0][:min(3, len(positionSlice[0]))],
		"samplePos1_first3", positionSlice[1][:min(3, len(positionSlice[1]))])

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

// SetVisionPath sets the path to a separate vision GGUF file for split models.
// This should be called before any image processing if the vision model is
// stored in a separate file from the language model.
func (m *Model) SetVisionPath(path string) {
	m.visionPath = path
	m.visionReady = false // Reset ready flag to trigger re-loading
}

// Close cleans up resources used by the model, including the vision backend
// if it was loaded from a separate file.
func (m *Model) Close() {
	if m.visionBackend != nil {
		m.visionBackend.Close()
		m.visionBackend = nil
	}
}

func init() {
	model.Register("qwen3vl", New)
	model.Register("qwen3vlmoe", New)
}
