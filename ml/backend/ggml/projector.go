package ggml

// #cgo CPPFLAGS: -I${SRCDIR}/ggml/include
// #include <stdlib.h>
// #include <stdint.h>
// #include "ggml.h"
// #include "ggml-backend.h"
import "C"

import (
	"fmt"
	"os"
	"unsafe"

	"github.com/ollama/ollama/format"
	fsggml "github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/logutil"
)

// AttachProjector extends the backend with tensors from an external GGUF file (e.g. split vision projectors).
// It returns the projector configuration so callers can merge metadata as needed.
func (b *Backend) AttachProjector(path string) (fsggml.KV, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open projector: %w", err)
	}
	defer f.Close()

	meta, err := fsggml.Decode(f, -1)
	if err != nil {
		return nil, fmt.Errorf("decode projector: %w", err)
	}

	bt, ok := b.deviceBufferTypes[b.output]
	if !ok {
		return nil, fmt.Errorf("no buffer type for projector device")
	}

	ctx := (*C.struct_ggml_context)(nil)
	if b.projectorContexts != nil {
		ctx = b.projectorContexts[bt]
	}

	if ctx == nil {
		tensorCount := len(meta.Tensors().Items()) + 1024 // add substantial headroom for tensor metadata allocations
		memSize := C.ggml_tensor_overhead() * C.size_t(tensorCount)
		if minReserve := C.size_t(64 * 1024 * 1024); memSize < minReserve {
			memSize = minReserve
		}
		ctx = C.ggml_init(C.struct_ggml_init_params{
			mem_size: memSize,
			no_alloc: true,
		})
		if ctx == nil {
			return nil, fmt.Errorf("failed to initialize projector context")
		}

		if b.projectorContexts == nil {
			b.projectorContexts = make(map[C.ggml_backend_buffer_type_t]*C.struct_ggml_context)
		}
		b.projectorContexts[bt] = ctx
	}

	if b.tensorLoadTargets == nil {
		b.tensorLoadTargets = make(map[string][]string)
	}
	if b.tensors == nil {
		b.tensors = make(map[string]*C.struct_ggml_tensor)
	}

	for _, t := range meta.Tensors().Items() {
		if _, exists := b.tensors[t.Name]; exists {
			continue
		}

		var shapePtr *C.int64_t
		if len(t.Shape) > 0 {
			shapePtr = (*C.int64_t)(unsafe.Pointer(&t.Shape[0]))
		}

		cname := C.CString(t.Name)
		tensor := C.ggml_new_tensor(ctx, t.Kind, C.int(len(t.Shape)), shapePtr)
		if tensor == nil {
			C.free(unsafe.Pointer(cname))
			return nil, fmt.Errorf("failed to create tensor %s", t.Name)
		}
		C.ggml_set_name(tensor, cname)
		C.free(unsafe.Pointer(cname))

		b.tensors[t.Name] = tensor
		b.tensorLoadTargets[t.Name] = []string{t.Name}

		size := pad(C.ggml_backend_buft_get_alloc_size(bt, tensor), C.ggml_backend_buft_get_alignment(bt))
		if mem := b.btDeviceMemory[bt]; mem != nil {
			idx := len(mem.Weights) - 1
			if idx < 0 {
				idx = 0
				mem.Weights = append(mem.Weights, 0)
			}
			mem.Weights[idx] += uint64(size)
		}

		logutil.Trace("projector tensor registered", "name", t.Name, "size", format.HumanBytes2(uint64(size)))
	}

	// Allocate buffer AFTER creating all tensors
	if b.allocMemory {
		// Check if buffer already exists for this context
		if b.weightBuffers == nil {
			b.weightBuffers = make(map[*C.struct_ggml_context]C.ggml_backend_buffer_t)
		}
		if _, exists := b.weightBuffers[ctx]; !exists {
			buffer := C.ggml_backend_alloc_ctx_tensors_from_buft(ctx, bt)
			if buffer == nil {
				return nil, fmt.Errorf("failed to allocate projector buffer")
			}
			C.ggml_backend_buffer_set_usage(buffer, C.GGML_BACKEND_BUFFER_USAGE_WEIGHTS)
			b.weightBuffers[ctx] = buffer
		}
	}

	b.extraFiles = append(b.extraFiles, modelFile{path: path, meta: meta})

	return meta.KV(), nil
}

// SetConfig overrides the backend configuration used by downstream model constructors.
func (b *Backend) SetConfig(cfg fsggml.KV) {
	b.config = cfg
}
