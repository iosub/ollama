package llamarunner

import (
	"testing"

	"github.com/ollama/ollama/llama"
)

func TestImageCache(t *testing.T) {
	cache := ImageContext{images: make([]imageCache, 4)}

	valA := []*llama.MtmdChunk{nil, nil}
	valB := []*llama.MtmdChunk{nil}
	valC := []*llama.MtmdChunk{nil}
	valD := []*llama.MtmdChunk{nil}
	valE := []*llama.MtmdChunk{nil}

	// Empty cache
	result, err := cache.findImage(0x5adb61d31933a946)
	if err != errImageNotFound {
		t.Errorf("found result in empty cache: result %v, err %v", result, err)
	}

	// Insert A
	cache.addImage(0x5adb61d31933a946, valA)

	result, err = cache.findImage(0x5adb61d31933a946)
	if len(result) != len(valA) {
		t.Errorf("failed to find expected value length: result %d, want %d err %v", len(result), len(valA), err)
	}

	// Insert B
	cache.addImage(0x011551369a34a901, valB)

	result, err = cache.findImage(0x5adb61d31933a946)
	if len(result) != len(valA) {
		t.Errorf("failed to find expected value length: result %d, want %d err %v", len(result), len(valA), err)
	}
	result, err = cache.findImage(0x011551369a34a901)
	if len(result) != len(valB) {
		t.Errorf("failed to find expected value length: result %d, want %d err %v", len(result), len(valB), err)
	}

	// Replace B with C
	cache.addImage(0x011551369a34a901, valC)

	result, err = cache.findImage(0x5adb61d31933a946)
	if len(result) != len(valA) {
		t.Errorf("failed to find expected value length: result %d, want %d err %v", len(result), len(valA), err)
	}
	result, err = cache.findImage(0x011551369a34a901)
	if len(result) != len(valC) {
		t.Errorf("failed to find expected value length: result %d, want %d err %v", len(result), len(valC), err)
	}

	// Evict A
	cache.addImage(0x756b218a517e7353, valB)
	cache.addImage(0x75e5e8d35d7e3967, valD)
	cache.addImage(0xd96f7f268ca0646e, valE)

	result, err = cache.findImage(0x5adb61d31933a946)
	if len(result) == len(valA) {
		t.Errorf("expected eviction of original entry: result len %d", len(result))
	}
	result, err = cache.findImage(0x756b218a517e7353)
	if len(result) != len(valB) {
		t.Errorf("failed to find expected value length: result %d, want %d err %v", len(result), len(valB), err)
	}
	result, err = cache.findImage(0x011551369a34a901)
	if len(result) != len(valC) {
		t.Errorf("failed to find expected value length: result %d, want %d err %v", len(result), len(valC), err)
	}
	result, err = cache.findImage(0x75e5e8d35d7e3967)
	if len(result) != len(valD) {
		t.Errorf("failed to find expected value length: result %d, want %d err %v", len(result), len(valD), err)
	}
	result, err = cache.findImage(0xd96f7f268ca0646e)
	if len(result) != len(valE) {
		t.Errorf("failed to find expected value length: result %d, want %d err %v", len(result), len(valE), err)
	}
}
