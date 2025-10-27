package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/ollama/ollama/fs/ggml"
)

func main() {
	path := flag.String("file", "", "path to gguf")
	filter := flag.String("filter", "", "substring filter")
	flag.Parse()

	if *path == "" {
		log.Fatal("missing --file")
	}

	f, err := os.Open(*path)
	if err != nil {
		log.Fatalf("open: %v", err)
	}
	defer f.Close()

	meta, err := ggml.Decode(f, -1)
	if err != nil {
		log.Fatalf("decode: %v", err)
	}

	for _, t := range meta.Tensors().Items() {
		if *filter == "" || strings.Contains(t.Name, *filter) {
			fmt.Printf("%s %v\n", t.Name, t.Shape)
		}
	}
}
