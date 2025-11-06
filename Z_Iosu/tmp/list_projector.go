package main

import (
	"fmt"
	"log"
	"os"
	"strings"

	fsggml "github.com/ollama/ollama/fs/ggml"
)

func main() {
	if len(os.Args) < 2 {
		log.Fatalf("usage: %s <gguf>", os.Args[0])
	}
	path := os.Args[1]
	var filter string
	if len(os.Args) > 2 {
		filter = strings.ToLower(os.Args[2])
	}
	f, err := os.Open(path)
	if err != nil {
		log.Fatalf("open: %v", err)
	}
	defer f.Close()

	meta, err := fsggml.Decode(f, -1)
	if err != nil {
		log.Fatalf("decode: %v", err)
	}

	fmt.Printf("tensors: %d\n", len(meta.Tensors().Items()))
	for _, t := range meta.Tensors().Items() {
		if filter != "" && !strings.Contains(strings.ToLower(t.Name), filter) {
			continue
		}
		fmt.Printf("%s %v\n", t.Name, t.Shape)
	}
}
