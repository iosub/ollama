package llamarunner

import (
	"errors"
	"fmt"
	"log/slog"
	"reflect"
	"time"

	"github.com/ollama/ollama/llama"
)

type InputCache struct {
	// context window size (per slot)
	numCtx int

	// individual KV caches
	slots []InputCacheSlot

	// optimize cache eviction for multiple users
	multiUserCache bool

	lc *llama.Context
}

func NewInputCache(lc *llama.Context, kvSize int, numSlots int, multiUserCache bool) (*InputCache, error) {
	if kvSize/numSlots < 1 {
		return nil, fmt.Errorf("must have at least one kv cache entry per parallel sequence (kv: %v parallel: %v)", kvSize, numSlots)
	}

	slots := make([]InputCacheSlot, numSlots)

	for i := range slots {
		slots[i] = InputCacheSlot{
			Id:     i,
			Inputs: make([]input, 0),
		}
	}

	return &InputCache{
		numCtx:         kvSize / numSlots,
		slots:          slots,
		multiUserCache: multiUserCache,
		lc:             lc,
	}, nil
}

// Locking: Operations on InputCacheSlot (including finding one
// through LoadCacheSlot) require a lock to be held that serializes
// these operations with each other and llama.Decode

type InputCacheSlot struct {
	// Index in the KV cache
	Id int

	// Inputs that are stored in the KV cache
	Inputs []input

	// is this cache actively being processed as part of a sequence?
	InUse bool

	// last time this cache was used (as of start of processing)
	lastUsed time.Time
}

func (c *InputCache) LoadCacheSlot(prompt []input, cachePrompt bool) (*InputCacheSlot, []input, error) {
	var slot *InputCacheSlot
	var numPast int
	var err error

	// In single-user scenarios, the longest cache slot works fine for getting good input
	// cache hit rates and it reuses the same VRAM over and over again, which is good for
	// GPU performance in situations where we miss the input cache.
	// For multiple users, the "best" cache slot produces better input cache hit rates
	// at the cost of worse performance when we miss the input cache (because it causes
	// GPU L2 cache misses due to spreading out accesses across VRAM).
	if !c.multiUserCache {
		slot, numPast, err = c.findLongestCacheSlot(prompt)
	} else {
		slot, numPast, err = c.findBestCacheSlot(prompt)
	}
	if err != nil {
		return nil, nil, err
	}

	if !cachePrompt {
		numPast = 0
	}

	slot.InUse = true
	slot.lastUsed = time.Now()

	if numPast == len(prompt) {
		// Leave one input to sample so we can get a response
		numPast--
	}

	// Convert input index to token position for KV cache operations
	numPastTokens := countInputTokens(prompt[:numPast])
	
	// Always clear discarded portions completely
	if numPastTokens == 0 {
		// No cache reuse - clear everything
		c.lc.KvCacheSeqRm(slot.Id, 0, -1)
		// Free all old image chunks
		for i := range slot.Inputs {
			if slot.Inputs[i].imageChunk != nil {
				slot.Inputs[i].imageChunk.Free()
			}
		}
		slot.Inputs = nil
	} else if !c.lc.KvCacheSeqRm(slot.Id, numPastTokens, -1) {
		// Some models don't support partial erasure
		c.lc.KvCacheSeqRm(slot.Id, 0, -1)
		numPast = 0
		numPastTokens = 0
		// Free discarded chunks
		for i := numPast; i < len(slot.Inputs); i++ {
			if slot.Inputs[i].imageChunk != nil {
				slot.Inputs[i].imageChunk.Free()
			}
		}
	} else {
		// Partial cache reuse - free only discarded chunks
		for i := numPast; i < len(slot.Inputs); i++ {
			if slot.Inputs[i].imageChunk != nil {
				slot.Inputs[i].imageChunk.Free()
			}
		}
	}

	slog.Debug("loading cache slot", "id", slot.Id, "cache", len(slot.Inputs), "prompt", len(prompt),
		"used", numPast, "remaining", len(prompt)-numPast)

	slot.Inputs = prompt[:numPast]
	prompt = prompt[numPast:]

	return slot, prompt, nil
}

func (c *InputCache) findLongestCacheSlot(prompt []input) (*InputCacheSlot, int, error) {
	longest := -1
	var longestSlot *InputCacheSlot

	for i, s := range c.slots {
		if s.InUse {
			continue
		}

		count := countCommonPrefix(s.Inputs, prompt)
		if count > longest {
			longest = count
			longestSlot = &c.slots[i]
		}
	}

	if longestSlot == nil {
		return nil, 0, errors.New("no available cache slots")
	}

	return longestSlot, longest, nil
}

func (c *InputCache) findBestCacheSlot(prompt []input) (*InputCacheSlot, int, error) {
	oldest := time.Now()
	var oldestSlot *InputCacheSlot

	longest := -1
	var longestSlot *InputCacheSlot

	for i, s := range c.slots {
		count := countCommonPrefix(s.Inputs, prompt)
		if count > longest {
			longest = count
			longestSlot = &c.slots[i]
		}

		if s.lastUsed.Compare(oldest) < 0 && !s.InUse {
			oldest = s.lastUsed
			oldestSlot = &c.slots[i]
		}
	}

	if longest == len(longestSlot.Inputs) && !longestSlot.InUse {
		return longestSlot, longest, nil
	}

	if oldestSlot.InUse {
		return nil, 0, errors.New("no available cache slots")
	}

	if len(oldestSlot.Inputs) != 0 {
		slog.Debug("evicting cache slot", "id", oldestSlot.Id, "inputs", len(oldestSlot.Inputs),
			"used", oldestSlot.lastUsed)
	}

	if longest > 0 && longestSlot != oldestSlot {
		longestTokens := countInputTokens(longestSlot.Inputs[:longest])
		slog.Debug("forking cache slot", "src", longestSlot.Id, "dst", oldestSlot.Id, "inputs", longest, "total",
			len(longestSlot.Inputs))
		oldestSlot.Inputs = make([]input, longest)
		copy(oldestSlot.Inputs, longestSlot.Inputs[:longest])
		// This is only nil for unit tests
		if c.lc != nil {
			c.lc.KvCacheSeqRm(oldestSlot.Id, 0, -1)
			c.lc.KvCacheSeqCp(longestSlot.Id, oldestSlot.Id, 0, longestTokens)
		}
	}

	return oldestSlot, longest, nil
}

func countCommonPrefix(a []input, b []input) int {
	var count int

	for i := range a {
		if i >= len(b) {
			break
		}

		if !reflect.DeepEqual(a[i], b[i]) {
			break
		}

		count++
	}

	return count
}

func findInputsUpToToken(inputs []input, targetTokens int) int {
	tokens := 0
	for i, in := range inputs {
		if tokens >= targetTokens {
			return i
		}
		switch {
		case in.imageChunk != nil:
			tokens += in.imageChunk.NumTokens()
		case in.imageID != "" && in.token == -1: // tokenImagePlaceholder
			tokens++
		default:
			tokens++
		}
	}
	return len(inputs)
}

func (c *InputCache) ShiftDiscard(inputTokens int, numKeep int) int {
	targetFree := (c.numCtx - numKeep) / 2
	targetFree = max(targetFree, 1)

	currentFree := c.numCtx - inputTokens

	return max(targetFree-currentFree, 0)
}

type ErrReprocessInputs struct {
	Inputs []input
}

func (e *ErrReprocessInputs) Error() string {
	return fmt.Sprintf("kv cache shift not supported, inputs need reprocessing (input count: %v)", len(e.Inputs))
}

// ShiftCacheSlot frees up space in the KV cache by deleting the oldest half of history
// and shifting the newest half into that space (saving numKeep inputs at the beginning).
//
// Assumes that at least 1 entry can be freed up by shifting (i.e. numKeep < numCtx)
func (c *InputCache) ShiftCacheSlot(slot *InputCacheSlot, numKeep int) error {
	if numKeep >= c.numCtx {
		return fmt.Errorf("unable to shift context - keep exceeds context (keep: %v context: %v)", numKeep, c.numCtx)
	}

	inputTokens := countInputTokens(slot.Inputs)
	discard := c.ShiftDiscard(inputTokens, numKeep)

	if discard <= 0 {
		return nil
	}

	slog.Debug("context limit hit - shifting", "id", slot.Id, "limit", c.numCtx, "input", inputTokens,
		"keep", numKeep, "discard", discard)

	var shiftFailed bool

	if c.lc.KvCacheCanShift() {
		// For models that support shifting, attempt to shift the KV cache
		if !c.lc.KvCacheSeqRm(slot.Id, numKeep, numKeep+discard) {
			shiftFailed = true
			slog.Debug("kv cache removal not supported, clearing cache and returning inputs for reprocessing", "id", slot.Id)
		} else {
			c.lc.KvCacheSeqAdd(slot.Id, numKeep+discard, inputTokens, -discard)
		}
	} else {
		// For models that don't support shifting
		shiftFailed = true
		slog.Debug("kv cache cannot shift, clearing cache and returning inputs for reprocessing", "id", slot.Id)
	}

	if shiftFailed {
		// Create new input slice preserving structure
		// Find input indices corresponding to token positions
		keptInputs := findInputsUpToToken(slot.Inputs, numKeep)
		discardInputs := findInputsUpToToken(slot.Inputs[keptInputs:], discard)
		newInputs := make([]input, keptInputs+len(slot.Inputs)-(keptInputs+discardInputs))
		copy(newInputs[:keptInputs], slot.Inputs[:keptInputs])
		copy(newInputs[keptInputs:], slot.Inputs[keptInputs+discardInputs:])

		// Clear the entire KV cache
		_ = c.lc.KvCacheSeqRm(slot.Id, 0, -1)
		// Reset the slot inputs since we've cleared the cache
		slot.Inputs = []input{}

		// Return error with inputs that need to be reprocessed
		return &ErrReprocessInputs{Inputs: newInputs}
	}

	// Standard shift succeeded - update input array
	keptInputs := findInputsUpToToken(slot.Inputs, numKeep)
	discardInputs := findInputsUpToToken(slot.Inputs[keptInputs:], discard)
	for i := keptInputs + discardInputs; i < len(slot.Inputs); i++ {
		slot.Inputs[i-discardInputs] = slot.Inputs[i]
	}
	slot.Inputs = slot.Inputs[:len(slot.Inputs)-discardInputs]

	return nil
}
