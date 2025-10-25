# PR #12654: Intel GPU Memory Detection using Level Zero Sysman API

## Overview
This PR adds support for detecting Intel GPU memory using the Level Zero System Management API (Sysman). This enhancement improves VRAM detection accuracy for Intel Arc and Flex GPUs when running with Vulkan backend.

## Source
- **Upstream PR**: https://github.com/ollama/ollama/pull/12654
- **Applied**: October 25, 2025
- **Branch**: 12_07_mio
- **Commit**: 8a3856f41

## Problem Statement
Intel GPUs were not reporting accurate VRAM information through Vulkan API alone. The Level Zero Sysman API provides more detailed and accurate memory information for Intel discrete GPUs.

## Changes Made

### 1. Dockerfile
- Added Intel oneAPI Level Zero runtime installation
- Copied Level Zero shared libraries to `/lib/ollama/level_zero/`
- Ensures runtime availability for Level Zero API calls

### 2. ggml CMake Build System
**File**: `ml/backend/ggml/ggml/src/CMakeLists.txt`
- Added `mem_l0_sysman.cpp` to build sources
- Integrated Level Zero memory detection into ggml build

### 3. ggml Implementation Header
**File**: `ml/backend/ggml/ggml/src/ggml-impl.h`
- Added Level Zero Sysman API function declarations
- Defined interface for GPU memory querying:
  - `ggml_l0_sysman_init()` - Initialize Level Zero context
  - `ggml_l0_sysman_get_device_count()` - Get number of Intel GPUs
  - `ggml_l0_sysman_get_total_memory()` - Get total VRAM
  - `ggml_l0_sysman_get_free_memory()` - Get available VRAM

### 4. Level Zero Implementation
**File**: `ml/backend/ggml/ggml/src/mem_l0_sysman.cpp` (NEW - 21KB)
- Complete implementation of Intel GPU memory detection
- Dynamic library loading for Windows and Linux
- Key features:
  - Fallback mechanism if Level Zero unavailable
  - Multiple device support
  - Memory query caching for performance
  - Thread-safe initialization
  - Comprehensive error handling

### 5. Vulkan Backend Integration
**File**: `ml/backend/ggml/ggml/src/ggml-vulkan/ggml-vulkan.cpp`
- Enhanced GPU memory detection in `ggml_vk_init()`
- Prioritizes Level Zero data for Intel GPUs
- Falls back to Vulkan memory queries for non-Intel or when L0 unavailable
- Improves accuracy of VRAM reporting

### 6. Patch Documentation
**File**: `llama/patches/0032-Add-memory-detection-for-Intel-GPU-using-Level-Zero.patch`
- Documents changes for llama.cpp submodule
- Tracks Level Zero integration for future updates

## Technical Details

### Level Zero API Functions Used
- `zeInit()` - Initialize Level Zero driver
- `zeDriverGet()` - Enumerate drivers
- `zesDeviceGet()` - Get device handles
- `zesDeviceEnumMemoryModules()` - Enumerate memory modules
- `zesMemoryGetProperties()` - Get memory properties
- `zesMemoryGetState()` - Get current memory state

### Memory Detection Flow
1. Initialize Level Zero driver context
2. Enumerate Intel GPU devices
3. For each device:
   - Query memory module properties
   - Get total memory capacity
   - Get current free memory
4. Cache results for subsequent queries
5. Fallback to Vulkan queries if Level Zero fails

## Benefits
- **Accurate VRAM Detection**: More reliable than Vulkan-only detection
- **Better Resource Management**: Ollama can make informed decisions about model loading
- **Intel GPU Support**: Improved support for Arc A-series and Flex GPUs
- **Cross-Platform**: Works on both Windows and Linux
- **Graceful Degradation**: Falls back to Vulkan if Level Zero unavailable

## Testing Recommendations
1. Test with Intel Arc A770/A750/A380 GPUs
2. Test with Intel Flex 140/170 GPUs
3. Verify VRAM reporting accuracy: `ollama ps` should show correct memory
4. Test multi-GPU scenarios with mixed Intel/NVIDIA/AMD
5. Verify fallback behavior when Level Zero libraries missing

## Dependencies
- Intel Level Zero runtime libraries (Linux: `level-zero`, Windows: bundled)
- Vulkan SDK (existing dependency)
- Compatible Intel GPU driver with Level Zero support

## Known Limitations
- Only detects discrete Intel GPUs (Arc/Flex series)
- Integrated GPUs (UHD/Iris Xe) may have limited Level Zero support
- Requires recent Intel GPU drivers (2023+)

## Files Modified
```
Dockerfile
ml/backend/ggml/ggml/src/CMakeLists.txt
ml/backend/ggml/ggml/src/ggml-impl.h
ml/backend/ggml/ggml/src/mem_l0_sysman.cpp (NEW)
ml/backend/ggml/ggml/src/ggml-vulkan/ggml-vulkan.cpp
llama/patches/0032-Add-memory-detection-for-Intel-GPU-using-Level-Zero.patch (NEW)
```

## Statistics
- **Files Changed**: 7
- **Insertions**: 893
- **Deletions**: 9
- **New Files**: 2

## Related Issues
- Improves accuracy of GPU memory detection for Ollama scheduler
- Complements PR #12665 (GPU ordering) for better multi-GPU support
