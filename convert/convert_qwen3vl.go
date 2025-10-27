package convert

import (
	"cmp"
	"encoding/json"
	"io"
	"io/fs"
	"slices"
	"strings"

	"github.com/ollama/ollama/fs/ggml"
)

type qwen3VLModel struct {
	qwen3Model `json:"text_config"`

	VisionModel struct {
		Depth                  uint32  `json:"depth"`
		HiddenSize             uint32  `json:"hidden_size"`
		NumHeads               uint32  `json:"num_heads"`
		InChannels             uint32  `json:"in_channels"`
		PatchSize              uint32  `json:"patch_size"`
		SpatialMergeSize       uint32  `json:"spatial_merge_size"`
		WindowSize             uint32  `json:"window_size"`
		RMSNormEps             float32 `json:"layer_norm_epsilon"`
		RopeTheta              float32 `json:"rope_theta"`
		TemporalPatchSize      uint32  `json:"temporal_patch_size"`
		DeepstackVisualIndexes []int32 `json:"deepstack_visual_indexes"`
		HiddenAct              string  `json:"hidden_act"`

		Size struct {
			ShortestEdge uint32 `json:"shortest_edge"`
			LongestEdge  uint32 `json:"longest_edge"`
		} `json:"size"`

		ImageMean []float32 `json:"image_mean"`
		ImageStd  []float32 `json:"image_std"`
	} `json:"vision_config"`
}

func (m *qwen3VLModel) parseMore(fsys fs.FS) error {
	bts, err := fs.ReadFile(fsys, "preprocessor_config.json")
	if err != nil {
		return err
	}

	return json.Unmarshal(bts, &m.VisionModel)
}

func (m *qwen3VLModel) KV(t *Tokenizer) ggml.KV {
	kv := m.qwen3Model.KV(t)

	arch := "qwen3vl"
	if m.NumExperts > 0 {
		arch += "moe"
	}
	// override architecture
	kv["general.architecture"] = arch

	kv["vision.block_count"] = cmp.Or(m.VisionModel.Depth, 32)
	kv["vision.embedding_length"] = m.VisionModel.HiddenSize
	kv["vision.attention.head_count"] = cmp.Or(m.VisionModel.NumHeads, 16)
	kv["vision.num_channels"] = m.VisionModel.InChannels
	kv["vision.patch_size"] = cmp.Or(m.VisionModel.PatchSize, 14)
	kv["vision.spatial_merge_size"] = cmp.Or(m.VisionModel.SpatialMergeSize, 2)
	kv["vision.attention.layer_norm_epsilon"] = cmp.Or(m.VisionModel.RMSNormEps, 1e-6)
	kv["vision.rope.freq_base"] = cmp.Or(m.VisionModel.RopeTheta, 1e4)
	kv["vision.temporal_patch_size"] = cmp.Or(m.VisionModel.TemporalPatchSize, 2)
	
	// Add deepstack visual indexes support
	if len(m.VisionModel.DeepstackVisualIndexes) > 0 {
		kv["vision.deepstack_visual_indexes"] = m.VisionModel.DeepstackVisualIndexes
	}

	// Set projector type to QWEN3VL for improved vision handling
	kv["clip.projector_type"] = "qwen3vl_merger"

	// Add vision activation function detection
	hiddenAct := strings.ToLower(m.VisionModel.HiddenAct)
	if strings.Contains(hiddenAct, "gelu") {
		kv["vision.use_gelu"] = true
	} else if hiddenAct == "silu" {
		kv["vision.use_silu"] = true
	}

	kv["vision.shortest_edge"] = m.VisionModel.Size.ShortestEdge
	kv["vision.longest_edge"] = m.VisionModel.Size.LongestEdge

	kv["vision.image_mean"] = m.VisionModel.ImageMean
	kv["vision.image_std"] = m.VisionModel.ImageStd

	return kv
}

func (m *qwen3VLModel) Tensors(ts []Tensor) []*ggml.Tensor {
	var rest []Tensor
	var out []*ggml.Tensor
	for _, t := range ts {
		switch {
		case strings.Contains(t.Name(), "attn_qkv"):
			out = append(out, slices.Collect(splitDim(t, 0,
				split{Replacer: strings.NewReplacer("attn_qkv", "attn_q")},
				split{Replacer: strings.NewReplacer("attn_qkv", "attn_k")},
				split{Replacer: strings.NewReplacer("attn_qkv", "attn_v")},
			))...)
		case strings.Contains(t.Name(), "patch_embed") && strings.HasSuffix(t.Name(), "weight"):
			// Handle Conv3D patch embedding - split temporal dimension for Qwen3VL
			shape := t.Shape()
			if len(shape) == 5 { // Conv3D: [out_channels, in_channels, kt, kh, kw]
				// Split temporal dimension (kt should be 2 for Qwen3VL)
				kt := shape[2]
				if kt == 2 {
					// Create two separate Conv2D weights
					out = append(out, &ggml.Tensor{
						Name:     strings.Replace(t.Name(), "weight", "weight.0", 1),
						Kind:     t.Kind(),
						Shape:    []uint64{shape[0], shape[1], shape[3], shape[4]}, // Remove temporal dim
						WriterTo: temporalSplit{tensor: t, index: 0},
					})
					out = append(out, &ggml.Tensor{
						Name:     strings.Replace(t.Name(), "weight", "weight.1", 1),
						Kind:     t.Kind(),
						Shape:    []uint64{shape[0], shape[1], shape[3], shape[4]}, // Remove temporal dim
						WriterTo: temporalSplit{tensor: t, index: 1},
					})
				} else {
					// Fallback to original behavior for non-standard temporal sizes
					out = append(out, &ggml.Tensor{
						Name:     t.Name(),
						Kind:     t.Kind(),
						Shape:    append([]uint64{shape[0] * shape[1]}, shape[2:]...),
						WriterTo: t,
					})
				}
			} else {
				// Standard 2D conv handling
				out = append(out, &ggml.Tensor{
					Name:     t.Name(),
					Kind:     t.Kind(),
					Shape:    append([]uint64{shape[0] * shape[1]}, shape[2:]...),
					WriterTo: t,
				})
			}
		case strings.Contains(t.Name(), "deepstack_merger") && strings.Contains(t.Name(), "linear_fc"):
			// Handle MoE-style deepstack merger linear layers
			// Map linear_fc1 -> fc.0, linear_fc2 -> fc.2 (matching Qwen2VL indexing)
			newName := t.Name()
			if strings.Contains(newName, "linear_fc1") {
				newName = strings.Replace(newName, "linear_fc1", "fc.0", 1)
			} else if strings.Contains(newName, "linear_fc2") {
				newName = strings.Replace(newName, "linear_fc2", "fc.2", 1)
			}
			out = append(out, &ggml.Tensor{
				Name:     newName,
				Kind:     t.Kind(),
				Shape:    t.Shape(),
				WriterTo: t,
			})
		case strings.Contains(t.Name(), "merger.norm"):
			// Handle merger normalization layers
			out = append(out, &ggml.Tensor{
				Name:     strings.Replace(t.Name(), "merger.norm", "post_norm", 1),
				Kind:     t.Kind(),
				Shape:    t.Shape(),
				WriterTo: t,
			})
		default:
			rest = append(rest, t)
		}
	}

	return append(m.qwen3Model.Tensors(rest), out...)
}

// temporalSplit implements WriterTo for splitting temporal dimension of Conv3D
type temporalSplit struct {
	tensor Tensor
	index  int
}

func (ts temporalSplit) WriteTo(w io.Writer) (int64, error) {
	// This would need to be implemented to actually split the tensor data
	// For now, delegate to the original tensor (this needs proper implementation)
	return ts.tensor.WriteTo(w)
}

func (m *qwen3VLModel) Replacements() []string {
	return append(
		m.qwen3Model.Replacements(),
		"model.language_", "",
		"model.visual", "v",
		"patch_embed.proj", "patch_embed",
		"blocks", "blk",
		"attn.qkv", "attn_qkv", 
		"attn.proj", "attn_out",
		"deepstack_merger_list", "deepstack_merger",
		"merger.linear_fc1", "merger.fc.0",
		"merger.linear_fc2", "merger.fc.2",
		"merger.norm", "post_norm",
		".qkv.", ".attn_qkv.",
		".proj.", ".attn_out.",
	)
}
