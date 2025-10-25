# Ollama Custom Patches - Branch 12_07_mio

This directory contains patches and documentation for custom modifications applied to the Ollama codebase on branch `12_07_mio`.

## Applied Patches

### 0032 - Intel GPU Memory Detection (llama.cpp submodule)
**Files**: 
- `PR_12654_Intel_Level_Zero_Memory.md` - Full documentation
- `0032-Add-memory-detection-for-Intel-GPU-using-Level-Zero.patch` - Patch file

**Summary**: Adds Level Zero Sysman API support for accurate VRAM detection on Intel Arc and Flex GPUs in llama.cpp.

**Commit**: Included in `8a3856f41`

**Impact**:
- Improved memory detection for Intel GPUs in llama.cpp backend
- Better resource allocation decisions
- Cross-platform support (Windows/Linux)

---

### 0033 - Vulkan GPU Device Ordering
**Files**:
- `PR_Vulkan_GPU_Ordering.md` - Full documentation
- `0033-Vulkan-GPU-ordering-by-device-ID.patch` - Patch file

**Summary**: Changes GPU selection to respect Vulkan device enumeration order (ID-based) instead of VRAM-based sorting.

**Commit**: `15eab624f`

**Impact**:
- Predictable GPU selection
- Honors VkConfig settings
- Primary GPU (ID 0) prioritized
- Better multi-GPU behavior

---

### 0034 - Qwen2.5 VL Causal Masking Fix
**Files**:
- `PR_16745_Qwen_VL_Causal_Masking.md` - Full documentation
- `0034-Fix-Qwen25-VL-cache-causal-masking.patch` - Patch file

**Summary**: Fixes causal masking for Qwen Vision-Language models by tracking actual KV cache positions for non-consecutive token positions.

**Commit**: `e1a3d8557`

**Impact**:
- Qwen2.5-VL models work correctly
- Vision embeddings processed properly
- M-RoPE positions fixed for images
- Supports multi-image inputs

---

### 0035 - Intel GPU Level Zero Memory Detection (Ollama)
## Patch Application Order

These patches were applied in the following sequence:

1. **0032** - Intel Level Zero (llama.cpp submodule) - Base GPU detection in backend
2. **0033** - Vulkan GPU Ordering - GPU selection by device ID
3. **0034** - Qwen VL Fix - Vision model causal masking
4. **0035** - Intel Level Zero (Ollama) - Full integration in Ollama

**Total Changes**:
- **19 files** modified/created
- **1,054 insertions**
- **110 deletions**

---

## How to Apply Patches

### To a Fresh Ollama Clone

```bash
# From repository root
git apply Z_Iosu/patches/0035-Intel-GPU-Level-Zero-memory-detection.patch
git apply Z_Iosu/patches/0033-Vulkan-GPU-ordering-by-device-ID.patch
```

### To llama.cpp Submodule

```bash
cd llama/llama.cpp
git apply ../../Z_Iosu/patches/0032-Add-memory-detection-for-Intel-GPU-using-Level-Zero.patch
git apply ../../Z_Iosu/patches/0034-Fix-Qwen25-VL-cache-causal-masking.patch
cd ../..
```
### To a Fresh Ollama Clone

```bash
# From repository root
git apply Z_Iosu/patches/PR_12654_Intel_Level_Zero_Memory.patch
git apply Z_Iosu/patches/PR_Vulkan_GPU_Ordering.patch
git apply Z_Iosu/patches/PR_16745_Qwen_VL_Causal_Masking.patch
```

### To llama.cpp Submodule

```bash
cd llama/llama.cpp
git apply ../../Z_Iosu/patches/0032-Add-memory-detection-for-Intel-GPU-using-Level-Zero.patch
cd ../..
```

---

## Verification After Patching

### Check Intel GPU Detection
```powershell
# Build with Vulkan support
powershell -ExecutionPolicy Bypass -File Z_Iosu\scripts\build_windows.ps1 buildVulkan

# Run and check logs
$env:OLLAMA_DEBUG=1
.\dist\windows-amd64\ollama.exe serve
# Look for Level Zero initialization messages
```

### Check GPU Ordering
```powershell
$env:OLLAMA_DEBUG=1
.\dist\windows-amd64\ollama.exe serve
# Check logs for "GPU order after sort" with ID sequence 0, 1, 2...
```

### Check Qwen VL Models
```bash
ollama run qwen2.5-vl:7b "Describe this image" --image test.jpg
# Should work without position assertion errors
```

---

## Compatibility

### Tested Environments
- **Windows 11** with Visual Studio 2022
- **CUDA**: 13.0
- **Vulkan SDK**: 1.4.328.1
- **Intel Level Zero**: Latest runtime
- **Compiler**: MSVC 19.44, llvm-mingw 20240619

### GPU Compatibility
- **NVIDIA**: RTX 20/30/40 series (CUDA/Vulkan)
- **Intel**: Arc A-series, Flex (Vulkan + Level Zero)
- **AMD**: RX 6000/7000 series (Vulkan)

### Model Compatibility
- **All quantized models**: Q4_K_M, Q5_K_M, Q8_0, etc.
- **Vision models**: Qwen2.5-VL, Qwen-VL, LLaVA, etc.
- **Text models**: Llama, Gemma, Mistral, etc.

---

## Related Resources

### Upstream References
- [Ollama PR #12654](https://github.com/ollama/ollama/pull/12654) - Intel GPU Memory
- [llama.cpp PR #16745](https://github.com/ggml-org/llama.cpp/pull/16745) - Qwen VL Fix
- [Qwen VL Discussion](https://github.com/ggml-org/llama.cpp/issues/16207)

### Additional Patches to Consider
- [llama.cpp PR #16764](https://github.com/ggml-org/llama.cpp/pull/16764) - OCR improvements
- [Ollama PR #12665](https://github.com/ollama/ollama/pull/12665) - Related GPU work

---

## Maintenance Notes

### When to Reapply
- After pulling upstream changes
- After rebasing branch
- After submodule updates

### Conflict Resolution
If patches fail to apply:
1. Check if upstream already includes the change
2. Manually apply using documentation as reference
3. Regenerate patch: `git format-patch -1 <commit_hash>`

### Testing Checklist
- [ ] Compilation successful (CPU + Vulkan + CUDA)
- [ ] Intel GPU detected correctly (if available)
- [ ] GPU ordering follows ID sequence (0, 1, 2...)
- [ ] Qwen VL models run without errors
- [ ] Multi-GPU selection predictable
- [ ] No regressions in text-only models

---

## Contact & Support

For issues specific to these patches:
- Check documentation files (*.md) for detailed explanations
- Review patch files (*.patch) for exact changes
- Test with `OLLAMA_DEBUG=1` for verbose logging

For upstream issues:
- Report to respective GitHub repositories
- Reference commit hashes from this branch

---

**Last Updated**: October 25, 2025  
**Branch**: 12_07_mio  
**Base**: upstream/main (ollama/ollama)
