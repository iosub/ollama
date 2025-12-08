# OllamaSplitRunner Branch

## Branch Objective

**Enable Qwen3-VL split GGUF models to work correctly in Ollama's Go runner.**

Split GGUF models use two separate files (text model + vision encoder) instead of a single unified file. While these models work perfectly in llama.cpp's C++ runner (PR #13305), Ollama's new Go runner lacks support for the "deepstack" vision architecture required by Qwen3-VL.

## What This Branch Does

Implements deepstack support for split GGUF models:
- Vision embedding projection layers (FC1/FC2)
- Deepstack feature concatenation (main + 3 layers → 16384 dims)
- LLM integration (adding deepstack to hidden states)

## Current Status

**⚠️ Work in Progress - Blocked**

- ✅ Architecture understood and implemented
- ✅ Concatenation and LLM integration working
- ❌ **Blocker**: FC weights not loading from GGUF into struct array
- ❌ Model crashes with assertion failure at ggml.c:1669

## Documentation

All documentation is in `z_iosu_2/1_split_ollama/`:
- `README.md` - Quick overview
- `QUICK_START.md` - Immediate action guide
- `docs/` - Technical analysis
- `fix/` - Task tracking

## Next Steps

1. Investigate `model/vision_bridge.go` array handling
2. Implement manual FC weight loading if needed
3. Test projection works (4608→4096 dimensions)
4. Validate with actual images

## Base Commit

Based on checkpoint `c44f3de8` where split model loading works (with assertion disabled).
