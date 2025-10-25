# Vulkan GPU Device Ordering Fix

## Overview
This change modifies Ollama's GPU selection logic to respect Vulkan device enumeration order (ID-based) instead of sorting by VRAM capacity. This ensures predictable GPU selection that honors driver-level device ordering and VkConfig settings.

## Source
- **Based on commit**: 1f279e404 (referenced in upstream discussions)
- **Applied**: October 25, 2025
- **Branch**: 12_07_mio
- **Commit**: 15eab624f

## Problem Statement
Previous behavior sorted GPUs by free VRAM in descending order, which could lead to:
1. **Unpredictable device selection**: Weaker GPUs with more free VRAM selected over primary GPU
2. **VkConfig ignored**: User-defined Vulkan device ordering not respected
3. **Multi-GPU inconsistency**: Selection varied based on VRAM state rather than hardware preference
4. **Debug challenges**: GPU 0 in driver might not be GPU 0 in Ollama

Example problematic scenario:
- GPU 0: RTX 4090 (20GB VRAM, 10GB free)
- GPU 1: GTX 1660 (6GB VRAM, 6GB free)
- **Old behavior**: Selects GTX 1660 (more free VRAM)
- **New behavior**: Selects RTX 4090 (lower device ID)

## Changes Made

### 1. GPU Memory Allocation Logic
**File**: `llm/memory.go`
- **Function**: `pickBestFullFitByLibrary()`
- **Change**: Replace VRAM-based sorting with ID-based sorting

**Old code**:
```go
sort.Sort(sort.Reverse(ml.ByFreeMemory(sgl)))
```

**New code**:
```go
// Respect Vulkan device enumeration order (ID 0, 1, 2...) instead of reordering by VRAM
// This allows VkConfig and driver-level device ordering to be honored
sort.Slice(sgl, func(i, j int) bool {
    return sgl[i].ID < sgl[j].ID
})

// Debug log to verify GPU ordering
for idx, gpu := range sgl {
    slog.Debug("GPU order after sort", "index", idx, "ID", gpu.ID, "name", gpu.Name, "free", format.HumanBytes2(gpu.FreeMemory))
}
```

### 2. GPU Layout Creation
**File**: `llm/server.go`
- **Function**: `createLayout()`
- **Change**: Added debug logging for GPU ordering verification

**Added code**:
```go
// Sort GPUs by ID to respect Vulkan enumeration order
sort.Slice(estimate.layerSplits[i].GPUs, func(j, k int) bool {
    return estimate.layerSplits[i].GPUs[j].ID < estimate.layerSplits[i].GPUs[k].ID
})

// Debug log GPU ordering
for idx, gpu := range estimate.layerSplits[i].GPUs {
    slog.Debug("createLayout GPU ordering", "parallel", i, "index", idx, "ID", gpu.ID, "name", gpu.Name)
}
```

## Technical Details

### GPU Selection Flow
1. Enumerate available GPUs via Vulkan
2. Group GPUs by library type (Vulkan/CUDA/etc.)
3. **Sort by device ID** (ascending: 0, 1, 2, ...)
4. Try fitting model starting from GPU 0
5. If insufficient, expand to GPU 0+1, then 0+1+2, etc.

### Vulkan Device ID Meaning
- **ID 0**: Primary physical device (typically fastest/preferred GPU)
- **ID 1, 2, ...**: Secondary devices in driver enumeration order
- Driver enumeration respects:
  - PCIe slot order
  - VkConfig layer settings
  - Driver-specific preferences

### VkConfig Integration
Users can now use `vk_loader_settings.json` to control device order:
```json
{
    "VK_LAYER_LUNARG_device_select": {
        "device_index": 0
    }
}
```
This will be honored by Ollama's selection logic.

## Benefits
1. **Predictable Selection**: GPU 0 always considered first
2. **Respects Hardware Config**: Follows PCIe/driver enumeration
3. **VkConfig Support**: User preferences honored
4. **Debugging Clarity**: Device IDs match between driver and Ollama logs
5. **Better Multi-GPU**: Primary GPU used preferentially

## Behavior Changes

### Single GPU Systems
**No change** - Only one GPU available

### Multi-GPU Systems (Homogeneous)
**Example**: 2x RTX 4090
- **Old**: Could pick either based on VRAM state
- **New**: Always tries GPU 0 first

### Multi-GPU Systems (Heterogeneous)
**Example**: RTX 4090 (GPU 0) + GTX 1660 (GPU 1)
- **Old**: Might pick GTX 1660 if RTX 4090 busy
- **New**: Always tries RTX 4090 first, expands to both if needed

### SCHED_SPREAD Environments
When `OLLAMA_SCHED_SPREAD=true`:
- Both old and new behaviors spread across all GPUs
- **New advantage**: Predictable ordering (0, 1, 2) instead of VRAM-based

## Debug Logging
New debug logs added:
```
[DEBUG] GPU order after sort | index=0 ID=0 name="NVIDIA GeForce RTX 4090" free=22.5GB
[DEBUG] GPU order after sort | index=1 ID=1 name="NVIDIA GeForce GTX 1660" free=5.2GB
[DEBUG] createLayout GPU ordering | parallel=0 index=0 ID=0 name="NVIDIA GeForce RTX 4090"
```

Enable with: `OLLAMA_DEBUG=1 ollama serve`

## Testing Recommendations
1. **Multi-GPU Verification**:
   ```bash
   OLLAMA_DEBUG=1 ollama run llama3.2:latest
   # Check logs for GPU ordering
   ```

2. **VkConfig Test**:
   - Set preferred device via VkConfig
   - Verify Ollama respects the setting
   - Check debug logs for ID ordering

3. **Load Distribution**:
   - Load multiple models
   - Verify GPU 0 used first
   - Verify expansion to GPU 1 only when needed

4. **VRAM Stress Test**:
   - Fill GPU 0 with models
   - Load another model
   - Verify it uses GPU 1 (not skips GPU 0 entirely)

## Migration Notes
Users who relied on VRAM-based selection may see different behavior:
- **Before**: Largest free VRAM GPU selected
- **After**: Lowest ID GPU selected

If old behavior desired, can be achieved by:
1. Swapping physical GPU positions
2. Using VkConfig to reorder devices
3. Setting `OLLAMA_SCHED_SPREAD=true` for balanced distribution

## Known Limitations
- ID ordering determined by Vulkan driver
- No user override via environment variable (use VkConfig)
- Assumes lower ID = preferred device (typically true)

## Files Modified
```
llm/memory.go
llm/server.go
```

## Statistics
- **Files Changed**: 3 (including resolve)
- **Insertions**: 50
- **Deletions**: 6

## Related Changes
- Works with PR #12654 (Intel Level Zero) for accurate VRAM detection
- Complements SCHED_SPREAD feature for load balancing
- Integrates with Vulkan backend GPU discovery
