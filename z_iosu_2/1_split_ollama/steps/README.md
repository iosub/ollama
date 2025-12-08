# Steps Documentation

This directory contains step-by-step documentation of the implementation process for Qwen3-VL split GGUF support.

## Reading Order

1. **[01_root_cause_analysis.md](01_root_cause_analysis.md)**
   - Identified tensor name mismatch between vision_bridge and GGUF
   - Array indices (0,1,2) vs layer IDs (8,16,24)

2. **[02_deepstack_visual_indexes.md](02_deepstack_visual_indexes.md)**
   - Updated deepstackVisualIndexes to correct layer IDs
   - Changed `[0,1,2]` → `[8,16,24]`

3. **[03_manual_fc_loading.md](03_manual_fc_loading.md)**
   - Implemented manual FC weight loading
   - Loads 6 weights (3 layers × 2 FCs) using Backend.Get()
   - Result: All weights loaded successfully

4. **[04_patchmerger_fix.md](04_patchmerger_fix.md)**
   - Added FC nil check for PatchMerger
   - Prevents calling Forward when FC weights missing
   - Result: No more "FC layers are nil" warnings

5. **[05_projection_verification.md](05_projection_verification.md)**
   - Added detailed logging to verify projections
   - Confirmed all 3 projections work correctly (4608→4096)
   - Result: All deepstack outputs correct shape [4096 2025]

6. **[06_current_blocker.md](06_current_blocker.md)** ⚠️ **CURRENT STATUS**
   - Main vision output dimension mismatch
   - Error during concatenation (ggml.c:2515)
   - Needs research on main vision projection strategy

## Quick Summary

### ✅ Completed (85%)
- FC weight loading infrastructure
- Deepstack projection (3 layers)
- Dimension verification logging

### ❌ Blocked (15%)
- Main vision output projection
- Concatenation dimension mismatch
- Need to understand split model main projection strategy

## Key Files Modified

### Core Implementation
- `model/models/qwen3vl/model.go`
  - Lines 61-100: Manual FC loading
  - Lines 3-16: Added imports (fmt, ml/nn)

- `model/models/qwen3vl/model_vision.go`
  - Line 406: deepstackVisualIndexes [8,16,24]
  - Line 358: PatchMerger FC nil check
  - Lines 143-152: Detailed projection logging

## Test Logs Reference
- `z_iosu_2/logs/go09.log` - Before any fixes (FC nil errors)
- `z_iosu_2/logs/go10.log` - First FC loading attempt (didn't execute)
- `z_iosu_2/logs/go11.log` - FC loading works, still has error
- `z_iosu_2/logs/go12.log` - PatchMerger fix applied
- `z_iosu_2/logs/go13.log` - Projection verification
- `z_iosu_2/logs/go14.log` - Latest with detailed logging

## Next Session Guide

1. **Read**: `06_current_blocker.md` for current status
2. **Research**: llama.cpp `qwen3vl.cpp` main vision projection
3. **Test**: Try using deepstack merger for main projection
4. **Verify**: Check if concatenation works with corrected main output
