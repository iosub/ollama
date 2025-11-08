package model

import (
	"errors"
	"fmt"
	_ "image/jpeg"
	_ "image/png"
	"log/slog"
	"os"
	"reflect"
	"strconv"
	"strings"

	_ "golang.org/x/image/bmp"
	_ "golang.org/x/image/tiff"
	_ "golang.org/x/image/webp"

	"github.com/ollama/ollama/fs"
	fsggml "github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/kvcache"
	"github.com/ollama/ollama/logutil"
	"github.com/ollama/ollama/ml"
	_ "github.com/ollama/ollama/ml/backend"
	backendggml "github.com/ollama/ollama/ml/backend/ggml"
	"github.com/ollama/ollama/ml/nn/pooling"
	"github.com/ollama/ollama/model/input"
)

var (
	ErrNoVisionModel        = errors.New("this model is missing data required for image input")
	ErrUnsupportedModel     = errors.New("model not supported")
	ErrUnsupportedTokenizer = errors.New("tokenizer not supported")
)

func cloneKV(kv fsggml.KV) fsggml.KV {
	cloned := make(fsggml.KV, len(kv))
	for k, v := range kv {
		cloned[k] = v
	}
	return cloned
}

func applyProjectorMetadata(dst fsggml.KV, src fsggml.KV) {
	slog.Debug("applyProjectorMetadata called", "src_keys", len(src), "dst_keys", len(dst))

	// Log ALL source keys to debug
	slog.Debug("=== ALL PROJECTOR KEYS ===")
	for key := range src {
		slog.Debug("projector key", "key", key)
	}
	slog.Debug("=== END PROJECTOR KEYS ===")

	arch := dst.Architecture()
	if arch == "" {
		if a, ok := dst["general.architecture"].(string); ok {
			arch = a
		} else if a, ok := src["general.architecture"].(string); ok {
			arch = a
		}
	}
	if arch == "" {
		slog.Debug("applyProjectorMetadata: no architecture detected; skipping mapping")
		return
	}

	slog.Debug("applying metadata for architecture", "arch", arch)

	clipVisionCount := 0
	for key := range src {
		if strings.HasPrefix(key, "clip.vision.") {
			clipVisionCount++
			slog.Debug("found clip.vision key", "key", key)
		}
	}
	slog.Debug("clip.vision keys found", "count", clipVisionCount)

	var (
		patchSize       uint32
		imageSize       uint32
		spatialMerge    uint32
		temporalPatch   uint32
		deepstackMapped bool
	)

	for key, value := range src {
		switch {
		case strings.HasPrefix(key, "clip.vision.is_deepstack_layers"):
			slog.Debug("attempting to map is_deepstack_layers", "value_type", fmt.Sprintf("%T", value), "value", value)
			if flags, ok := toBoolSlice(value); ok {
				var indexes []int32
				for idx, flag := range flags {
					if flag {
						indexes = append(indexes, int32(idx))
					}
				}
				prefixedKey := arch + ".vision.deepstack_visual_indexes"
				dst[prefixedKey] = indexes
				deepstackMapped = true
				slog.Debug("mapped deepstack_visual_indexes", "key", prefixedKey, "count", len(indexes), "indexes", indexes)
			} else {
				slog.Warn("failed to convert is_deepstack_layers to bool slice", "value_type", fmt.Sprintf("%T", value))
			}
		case strings.HasPrefix(key, "clip.vision."):
			suffix := strings.TrimPrefix(key, "clip.vision.")
			visionKey := arch + ".vision." + suffix
			switch suffix {
			case "image_mean":
				if mean, ok := toFloat32Slice(value); ok {
					dst[visionKey] = mean
					slog.Debug("mapped vision image_mean", "key", visionKey, "len", len(mean))
				}
			case "image_std":
				if std, ok := toFloat32Slice(value); ok {
					dst[visionKey] = std
					slog.Debug("mapped vision image_std", "key", visionKey, "len", len(std))
				}
			case "attention.layer_norm_epsilon", "rope.freq_base":
				if f32, ok := toFloat32(value); ok {
					dst[visionKey] = f32
					slog.Debug("mapped float vision key", "clip_key", key, "vision_key", visionKey, "value", f32)
				}
			case "patch_size":
				if v, ok := toUint32(value); ok {
					dst[visionKey] = v
					patchSize = v
					slog.Debug("mapped patch_size", "value", v)
				}
			case "spatial_merge_size":
				if v, ok := toUint32(value); ok {
					dst[visionKey] = v
					spatialMerge = v
					slog.Debug("mapped spatial_merge_size", "value", v)
				}
			case "temporal_patch_size":
				if v, ok := toUint32(value); ok {
					dst[visionKey] = v
					temporalPatch = v
					slog.Debug("mapped temporal_patch_size", "value", v)
				}
			case "image_size":
				if v, ok := toUint32(value); ok {
					dst[visionKey] = v
					imageSize = v
					slog.Debug("mapped image_size", "value", v)
				}
			case "num_channels", "block_count", "embedding_length", "feed_forward_length", "projection_dim", "attention.head_count":
				if v, ok := toUint32(value); ok {
					dst[visionKey] = v
					slog.Debug("mapped uint vision key", "clip_key", key, "vision_key", visionKey, "value", v)
				}
			default:
				dst[visionKey] = value
				if slog.Default().Enabled(nil, slog.LevelDebug) {
					slog.Debug("mapped vision key", "clip_key", key, "vision_key", visionKey)
				}
			}
		case key == "clip.projector_type":
			dst[arch+".vision.projector_type"] = value
			slog.Debug("mapped projector_type", "value", value)
		case key == "clip.use_gelu":
			if b, ok := value.(bool); ok {
				dst[arch+".vision.use_gelu"] = b
				slog.Debug("mapped use_gelu", "value", b)
			}
		}
	}

	if !deepstackMapped {
		if indexes := dst.Ints(arch + ".vision.deepstack_visual_indexes"); len(indexes) > 0 {
			slog.Debug("deepstack indexes already present", "count", len(indexes))
		}
	}

	if _, ok := dst[arch+".vision.temporal_patch_size"]; !ok && temporalPatch > 0 {
		dst[arch+".vision.temporal_patch_size"] = temporalPatch
	}

	if _, ok := dst[arch+".vision.spatial_merge_size"]; !ok && spatialMerge > 0 {
		dst[arch+".vision.spatial_merge_size"] = spatialMerge
	}

	if _, ok := dst[arch+".vision.patch_size"]; !ok && patchSize > 0 {
		dst[arch+".vision.patch_size"] = patchSize
	}

	if _, ok := dst[arch+".vision.image_size"]; !ok && imageSize > 0 {
		dst[arch+".vision.image_size"] = imageSize
	}

	if _, ok := dst[arch+".vision.num_channels"]; !ok {
		if v, ok := toUint32(src["clip.vision.num_channels"]); ok {
			dst[arch+".vision.num_channels"] = v
			slog.Debug("backfilled num_channels", "value", v)
		}
	}

	if _, ok := dst[arch+".vision.num_positional_embeddings"]; !ok && patchSize > 0 && imageSize > 0 {
		grid := imageSize / patchSize
		if grid > 0 {
			dst[arch+".vision.num_positional_embeddings"] = uint32(grid * grid)
			slog.Debug("derived num_positional_embeddings", "value", grid*grid)
		}
	}

	if _, ok := dst[arch+".vision.spatial_merge_size"]; !ok && spatialMerge == 0 {
		dst[arch+".vision.spatial_merge_size"] = 2
	}
}

func toUint32(v interface{}) (uint32, bool) {
	switch t := v.(type) {
	case uint8:
		return uint32(t), true
	case uint16:
		return uint32(t), true
	case uint32:
		return t, true
	case uint64:
		return uint32(t), true
	case int:
		if t < 0 {
			return 0, false
		}
		return uint32(t), true
	case int32:
		if t < 0 {
			return 0, false
		}
		return uint32(t), true
	case int64:
		if t < 0 {
			return 0, false
		}
		return uint32(t), true
	case float32:
		if t < 0 {
			return 0, false
		}
		return uint32(t), true
	case float64:
		if t < 0 {
			return 0, false
		}
		return uint32(t), true
	case string:
		if u, err := strconv.ParseUint(t, 10, 32); err == nil {
			return uint32(u), true
		}
	}
	return 0, false
}

func toFloat32(v interface{}) (float32, bool) {
	switch t := v.(type) {
	case float32:
		return t, true
	case float64:
		return float32(t), true
	case int:
		return float32(t), true
	case int32:
		return float32(t), true
	case int64:
		return float32(t), true
	case uint32:
		return float32(t), true
	case uint64:
		return float32(t), true
	case string:
		if f, err := strconv.ParseFloat(t, 32); err == nil {
			return float32(f), true
		}
	}
	return 0, false
}

func toFloat32Slice(v interface{}) ([]float32, bool) {
	switch t := v.(type) {
	case []float32:
		return t, true
	case []float64:
		out := make([]float32, len(t))
		for i := range t {
			out[i] = float32(t[i])
		}
		return out, true
	case []interface{}:
		out := make([]float32, 0, len(t))
		for _, item := range t {
			if f, ok := toFloat32(item); ok {
				out = append(out, f)
			} else {
				return nil, false
			}
		}
		return out, true
	}
	return nil, false
}

func toBoolSlice(v interface{}) ([]bool, bool) {
	switch t := v.(type) {
	case []bool:
		return t, true
	case []interface{}:
		out := make([]bool, 0, len(t))
		for _, item := range t {
			if b, ok := item.(bool); ok {
				out = append(out, b)
			} else {
				return nil, false
			}
		}
		return out, true
	case []uint8:
		out := make([]bool, len(t))
		for i, v := range t {
			out[i] = v != 0
		}
		return out, true
	case []int32:
		out := make([]bool, len(t))
		for i, v := range t {
			out[i] = v != 0
		}
		return out, true
	case []uint32:
		out := make([]bool, len(t))
		for i, v := range t {
			out[i] = v != 0
		}
		return out, true
	default:
		// Try reflection for GGML array types like *ggml.array[bool]
		rv := reflect.ValueOf(v)
		if rv.Kind() == reflect.Ptr {
			rv = rv.Elem()
		}
		if rv.Kind() == reflect.Struct {
			// Look for "values" field
			valuesField := rv.FieldByName("values")
			if valuesField.IsValid() && valuesField.Kind() == reflect.Slice {
				// Try to convert to []bool
				if valuesField.Type().Elem().Kind() == reflect.Bool {
					out := make([]bool, valuesField.Len())
					for i := 0; i < valuesField.Len(); i++ {
						out[i] = valuesField.Index(i).Bool()
					}
					return out, true
				}
			}
		}
	}
	return nil, false
}

// Model implements a specific model architecture, defining the forward pass and any model-specific configuration
type Model interface {
	Forward(ml.Context, input.Batch) (ml.Tensor, error)

	Backend() ml.Backend
	Config() config
}

// MultimodalProcessor must be implemented by multimodal models.
type MultimodalProcessor interface {
	// EncodeMultimodal processes a single input (such as an image) and
	// generates an output (typically an embedding) that can be used by the model.
	//
	// The return value is one or more tensors, each with optional model-specific
	// opaque metadata. Typically, the tensors might be views into an embedding
	// with each view representing a chunk of data that can be processed independently
	// in different batches.
	//
	// The result may be cached by the runner.
	EncodeMultimodal(ml.Context, []byte) ([]input.Multimodal, error)

	// PostTokenize is called after tokenization to allow the model to edit the
	// input stream to correctly arrange multimodal elements.
	//
	// The input is a slice of tokens with the results of EncodeMultimodal interleaved
	// in the order that the user provided them. Each element of the slice will be
	// either a single token or single multimodal object.
	//
	// The model must ensure that inputs are stored according to how they will be
	// processed and stored in the cache. For example, Llava-style models should insert
	// placeholder tokens equal to the feature size of the corresponding image with
	// the image itself attached to and split across these tokens. When Forward is called
	// a partial subset of these tokens may be submitted according to the batch size.
	//
	// This function is also responsible for updating MultimodalHash for any Multimodal
	// that is modified to ensure that there is a unique hash value that accurately
	// represents the contents.
	PostTokenize([]*input.Input) ([]*input.Input, error)
}

// Base implements the common fields and methods for all models
type Base struct {
	b ml.Backend
	// projectorBackend is a separate backend for split GGUF models containing vision tensors
	projectorBackend ml.Backend
	config
}

type config struct {
	Cache kvcache.Cache
}

// Backend returns the underlying backend that will run the model
func (m *Base) Backend() ml.Backend {
	return m.b
}

// GetTensor retrieves a tensor from the model backend, falling back to projector backend for split GGUF models
func (m *Base) GetTensor(name string) ml.Tensor {
	// Try main backend first
	if t := m.b.Get(name); t != nil {
		return t
	}
	
	// For split GGUF models, try projector backend
	if m.projectorBackend != nil {
		if t := m.projectorBackend.Get(name); t != nil {
			slog.Debug("split GGUF: loaded tensor from projector", "name", name)
			return t
		}
	}
	
	return nil
}

func (m *Base) Config() config {
	return m.config
}

var models = make(map[string]func(fs.Config) (Model, error))

// Register registers a model constructor for the given architecture
func Register(name string, f func(fs.Config) (Model, error)) {
	if _, ok := models[name]; ok {
		panic("model: model already registered")
	}

	models[name] = f
}

// New initializes a new model instance with the provided configuration based on the metadata in the model file
func New(modelPath string, params ml.BackendParams) (Model, error) {
	b, err := ml.NewBackend(modelPath, params)
	if err != nil {
		return nil, err
	}

	m, err := modelForArch(b.Config())
	if err != nil {
		return nil, err
	}

	base := Base{b: b, config: m.Config()}
	v := reflect.ValueOf(m)
	v.Elem().Set(populateFields(base, v.Elem()))
	return m, nil
}

// NewWithProjector loads a model that requires additional projector GGUFs (e.g. split vision weights).
func NewWithProjector(modelPath string, projectorPaths []string, params ml.BackendParams) (Model, error) {
	if len(projectorPaths) == 0 {
		return New(modelPath, params)
	}

	// Create main model backend
	b, err := ml.NewBackend(modelPath, params)
	if err != nil {
		return nil, err
	}

	ggmlBackend, ok := b.(*backendggml.Backend)
	if !ok {
		return nil, fmt.Errorf("projector loading requires ggml backend")
	}

	// Create separate backend for projector tensors (split GGUF approach)
	projectorBackend, err := ml.NewBackend(projectorPaths[0], params)
	if err != nil {
		return nil, fmt.Errorf("failed to load projector: %w", err)
	}

	slog.Debug("split GGUF: loaded projector backend", "path", projectorPaths[0])

	var merged fsggml.KV
	if baseKV, ok := ggmlBackend.Config().(fsggml.KV); ok {
		merged = cloneKV(baseKV)
	} else {
		merged = make(fsggml.KV)
	}

	// Load projector GGUFs and merge their metadata
	for _, projector := range projectorPaths {
		// Open projector file
		projFile, err := os.Open(projector)
		if err != nil {
			return nil, fmt.Errorf("failed to open projector %s: %w", projector, err)
		}
		defer projFile.Close()
		
		// Parse projector KV metadata (no array size limit for projectors)
		projGGML, err := fsggml.Decode(projFile, -1)
		if err != nil {
			return nil, fmt.Errorf("failed to decode projector %s: %w", projector, err)
		}
		
		// Apply projector metadata to merged config
		applyProjectorMetadata(merged, projGGML.KV())
	}

	// Use merged config for model creation
	m, err := modelForArch(merged)
	if err != nil {
		return nil, err
	}

	base := Base{b: ggmlBackend, projectorBackend: projectorBackend, config: m.Config()}
	v := reflect.ValueOf(m)
	v.Elem().Set(populateFields(base, v.Elem()))
	return m, nil
}

func newTextProcessor(modelPath string, projectorPaths []string) (TextProcessor, error) {
	r, err := os.Open(modelPath)
	if err != nil {
		return nil, err
	}
	defer r.Close()

	meta, err := fsggml.Decode(r, -1)
	if err != nil {
		return nil, err
	}

	merged := meta.KV()
	if len(projectorPaths) > 0 {
		merged = cloneKV(merged)
		for _, projector := range projectorPaths {
			f, err := os.Open(projector)
			if err != nil {
				return nil, err
			}
			projMeta, err := fsggml.Decode(f, -1)
			_ = f.Close()
			if err != nil {
				return nil, err
			}
			applyProjectorMetadata(merged, projMeta.KV())
		}
	}

	m, err := modelForArch(merged)
	if err != nil {
		return nil, err
	}

	tp, ok := m.(TextProcessor)
	if !ok {
		return nil, ErrUnsupportedTokenizer
	}
	return tp, nil
}

func NewTextProcessor(modelPath string) (TextProcessor, error) {
	return newTextProcessor(modelPath, nil)
}

func NewTextProcessorWithProjector(modelPath string, projectorPaths []string) (TextProcessor, error) {
	if len(projectorPaths) == 0 {
		return NewTextProcessor(modelPath)
	}
	return newTextProcessor(modelPath, projectorPaths)
}

func modelForArch(c fs.Config) (Model, error) {
	arch := c.Architecture()
	if pooling.Type(c.Uint("pooling_type")) != pooling.TypeNone {
		arch = arch + "_embed"
	}

	f, ok := models[arch]
	if !ok {
		return nil, ErrUnsupportedModel
	}

	return f(c)
}

func populateFields(base Base, v reflect.Value, tags ...Tag) reflect.Value {
	t := v.Type()

	if t.Kind() == reflect.Struct {
		allNil := true
		for i := range t.NumField() {
			tt := t.Field(i).Type
			vv := v.Field(i)
			if !vv.CanSet() {
				continue
			}

			// make a copy
			tagsCopy := tags
			if tag := t.Field(i).Tag.Get("gguf"); tag != "" {
				tagsCopy = append(tagsCopy, parseTag(tag))
			}

			if tt == reflect.TypeOf((*Base)(nil)).Elem() {
				vv.Set(reflect.ValueOf(base))
			} else if tt == reflect.TypeOf((*ml.Tensor)(nil)).Elem() {
				var fn func([]Tag, string, string) [][]string
				fn = func(tags []Tag, prefix, suffix string) (fullNames [][]string) {
					if len(tags) > 0 {
						var names []string
						if tags[0].name != "" {
							for _, n := range append([]string{tags[0].name}, tags[0].alternatives...) {
								names = append(names, prefix+n+suffix)
							}
						}
						childNames := fn(tags[1:], tags[0].prefix, tags[0].suffix)
						if len(names) == 0 {
							// current tag has no name, use child names only
							fullNames = append(fullNames, childNames...)
						} else if len(childNames) == 0 {
							// current tag has names but no children, create branches for each name
							for _, name := range names {
								fullNames = append(fullNames, []string{name})
							}
						} else {
							// merge each name with each child
							for _, name := range names {
								for _, childName := range childNames {
									fullNames = append(fullNames, append([]string{name}, childName...))
								}
							}
						}
					}

					return fullNames
				}

				names := fn(tagsCopy, "", "")
				for _, name := range names {
					tensorName := strings.Join(name, ".")
					// Use GetTensor to search both main and projector backends for split GGUF
					var tensor ml.Tensor
					if base.projectorBackend != nil {
						tensor = base.Backend().Get(tensorName)
						if tensor == nil {
							tensor = base.projectorBackend.Get(tensorName)
							if tensor != nil {
								slog.Debug("populateFields: loaded tensor from projector", "name", tensorName)
							}
						}
					} else {
						tensor = base.Backend().Get(tensorName)
					}
					if tensor != nil {
						logutil.Trace("found tensor", "", tensor)
						vv.Set(reflect.ValueOf(tensor))
						break
					}
				}
			} else if tt.Kind() == reflect.Pointer || tt.Kind() == reflect.Interface {
				setPointer(base, vv, tagsCopy)
			} else if tt.Kind() == reflect.Slice || tt.Kind() == reflect.Array {
				for i := range vv.Len() {
					vvv := vv.Index(i)
					if vvv.Kind() == reflect.Pointer || vvv.Kind() == reflect.Interface {
						setPointer(base, vvv, append(tagsCopy, Tag{name: strconv.Itoa(i)}))
					} else {
						vvv.Set(populateFields(base, vvv, append(tagsCopy, Tag{name: strconv.Itoa(i)})...))
					}
				}
			}

			if !canNil(tt) || !vv.IsNil() {
				allNil = false
			}
		}

		if allNil {
			return reflect.Zero(t)
		}
	}

	return v
}

func setPointer(base Base, v reflect.Value, tags []Tag) {
	vv := v
	if v.Kind() == reflect.Interface {
		if v.IsNil() {
			return
		}

		vv = vv.Elem()
	}

	vv = reflect.Indirect(vv)
	if v.IsNil() {
		vv = reflect.New(v.Type().Elem()).Elem()
	}

	if f := populateFields(base, vv, tags...); f.CanAddr() {
		v.Set(f.Addr())
	}
}

type Tag struct {
	name,
	// prefix and suffix are applied to child tags
	prefix,
	suffix string
	alternatives []string
}

func parseTag(s string) (tag Tag) {
	parts := strings.Split(s, ",")
	if len(parts) > 0 {
		tag.name = parts[0]

		for _, part := range parts[1:] {
			if value, ok := strings.CutPrefix(part, "alt:"); ok && tag.name == "" {
				// elevate alternative to primary if no primary given
				tag.name = value
				slog.Warn("gguf tag has alt: but no primary name", "tag", s)
			} else if ok {
				tag.alternatives = append(tag.alternatives, value)
			}
			if value, ok := strings.CutPrefix(part, "pre:"); ok {
				tag.prefix = value
			}
			if value, ok := strings.CutPrefix(part, "suf:"); ok {
				tag.suffix = value
			}
		}
	}

	return
}

func canNil(t reflect.Type) bool {
	return t.Kind() == reflect.Chan ||
		t.Kind() == reflect.Func ||
		t.Kind() == reflect.Interface ||
		t.Kind() == reflect.Map ||
		t.Kind() == reflect.Pointer ||
		t.Kind() == reflect.Slice
}

func Forward(ctx ml.Context, m Model, batch input.Batch) (ml.Tensor, error) {
	if len(batch.Positions) != len(batch.Sequences) {
		return nil, fmt.Errorf("length of positions (%v) must match length of seqs (%v)", len(batch.Positions), len(batch.Sequences))
	}

	if len(batch.Positions) < 1 {
		return nil, errors.New("batch size cannot be less than 1")
	}

	cache := m.Config().Cache
	if cache != nil {
		err := cache.StartForward(ctx, batch, false)
		if err != nil {
			return nil, err
		}
	}

	t, err := m.Forward(ctx, batch)
	if err != nil {
		return nil, err
	}

	ctx.Forward(t)

	return t, nil
}
