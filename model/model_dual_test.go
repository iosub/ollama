package model

import (
	"reflect"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/ollama/ollama/ml/nn"
)

func TestPopulateFieldsDualBackend(t *testing.T) {
	type fakeLayer struct {
		Query *nn.Linear `gguf:"attn_q"`
		Key   *nn.Linear `gguf:"attn_k"`
	}

	type fakeModel struct {
		Layers [1]fakeLayer `gguf:"blk"`
	}

	var m fakeModel
	v := reflect.ValueOf(&m)

	// Main backend has attn_q
	b := &fakeBackend{
		names: []string{
			"blk.0.attn_q.weight",
		},
	}

	// Projector backend has attn_k
	pb := &fakeBackend{
		names: []string{
			"blk.0.attn_k.weight",
		},
	}

	v.Elem().Set(populateFields(Base{b: b, pb: pb}, v.Elem()))

	if diff := cmp.Diff(fakeModel{
		Layers: [1]fakeLayer{
			{
				Query: &nn.Linear{Weight: &fakeTensor{Name: "blk.0.attn_q.weight"}},
				Key:   &nn.Linear{Weight: &fakeTensor{Name: "blk.0.attn_k.weight"}},
			},
		},
	}, m); diff != "" {
		t.Errorf("populateFields() set incorrect values (-want +got):\n%s", diff)
	}
}
