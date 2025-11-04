package llamarunner

import (
	"errors"
	"fmt"
	"hash/maphash"
	"log/slog"
	"sync"
	"time"

	"github.com/ollama/ollama/llama"
)

const imageCacheSize = 4

type ImageContext struct {
	// mu is required to be held when generating embeddings or accessing the cache
	mu sync.Mutex

	mtmd *llama.MtmdContext

	// cache of images to embeddings
	images    []imageCache
	imageHash maphash.Hash
}

func NewImageContext(llamaContext *llama.Context, modelPath string) (*ImageContext, error) {
	arch, err := llama.GetModelArch(modelPath)
	if err != nil {
		return nil, fmt.Errorf("unable to determine vision architecture: %w (%s)", err, modelPath)
	}

	var c ImageContext
	if arch == "clip" {
		c.mtmd, err = llama.NewMtmdContext(llamaContext, modelPath)
	} else {
		return nil, fmt.Errorf("unknown vision model architecture: %s", arch)
	}

	if err != nil {
		return nil, err
	}

	c.images = make([]imageCache, imageCacheSize)

	return &c, nil
}

func (c *ImageContext) Free(modelPath string) {
	if c == nil {
		return
	}

	if c.mtmd != nil {
		c.mtmd.Free()
	}

	for i := range c.images {
		freeChunks(c.images[i].val)
		c.images[i].val = nil
	}
}

func (c *ImageContext) MultimodalTokenize(llamaContext *llama.Context, data []byte) ([]*llama.MtmdChunk, error) {
	if c == nil {
		return nil, nil
	}

	if len(data) == 0 {
		return nil, errors.New("received zero length image")
	}

	hash := c.hashImage(data)

	c.mu.Lock()
	defer c.mu.Unlock()

	chunks, err := c.findImage(hash)
	if err != nil {
		if c.mtmd == nil {
			return nil, errors.New("received image but vision model not loaded")
		}

		chunks, err = c.mtmd.MultimodalTokenize(llamaContext, data)
		if err != nil {
			return nil, err
		}

		c.addImage(hash, chunks)
	}

	return chunks, nil
}

func (c *ImageContext) BatchSize(configuredBatchSize int) int {
	// If images are not supported, we don't need to allocate embedding batches
	if c == nil {
		return 0
	}

	return configuredBatchSize
}

func (c *ImageContext) EmbedSize(llamaContext *llama.Context) int {
	return llamaContext.Model().NEmbd()
}

type imageCache struct {
	key      uint64
	val      []*llama.MtmdChunk
	lastUsed time.Time
}

func (c *ImageContext) hashImage(image []byte) uint64 {
	c.imageHash.Reset()
	_, _ = c.imageHash.Write(image)
	return c.imageHash.Sum64()
}

var errImageNotFound = errors.New("image not found in cache")

func (c *ImageContext) findImage(hash uint64) ([]*llama.MtmdChunk, error) {
	for i := range c.images {
		if c.images[i].key == hash {
			slog.Debug("loading image embeddings from cache", "entry", i)
			c.images[i].lastUsed = time.Now()
			return cloneChunks(c.images[i].val)
		}
	}

	return nil, errImageNotFound
}

func (c *ImageContext) addImage(hash uint64, chunks []*llama.MtmdChunk) {
	clones, err := cloneChunks(chunks)
	if err != nil {
		slog.Warn("unable to clone cached image chunks", "err", err)
		return
	}

	best := time.Now()
	var bestImage int

	for i := range c.images {
		if c.images[i].key == hash {
			bestImage = i
			break
		}

		if c.images[i].lastUsed.Compare(best) < 0 {
			best = c.images[i].lastUsed
			bestImage = i
		}
	}

	slog.Debug("storing image embeddings in cache", "entry", bestImage, "used", c.images[bestImage].lastUsed)
	freeChunks(c.images[bestImage].val)
	c.images[bestImage].key = hash
	c.images[bestImage].val = clones
	c.images[bestImage].lastUsed = time.Now()
}

func cloneChunks(chunks []*llama.MtmdChunk) ([]*llama.MtmdChunk, error) {
	if len(chunks) == 0 {
		return nil, nil
	}

	clones := make([]*llama.MtmdChunk, len(chunks))
	for i, chunk := range chunks {
		clone, err := chunk.Clone()
		if err != nil {
			freeChunks(clones[:i])
			return nil, err
		}
		clones[i] = clone
	}

	return clones, nil
}

func freeChunks(chunks []*llama.MtmdChunk) {
	for _, chunk := range chunks {
		if chunk != nil {
			chunk.Free()
		}
	}
}
