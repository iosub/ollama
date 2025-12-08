# Project Objective

## 🎯 GOAL
Make Qwen3-VL split GGUF models work in Ollama's Go runner.

## ✅ FACTS
- ✅ Model files are valid (work in llama.cpp PR #13305)
- ✅ Architecture understood (deepstack: 1152→4608→4096→16384)
- ❌ **BLOCKED**: vision_bridge doesn't load FC weights into array

## 🚧 STATUS
- Crashes at ggml.c:1669 assertion
- FC1/FC2 = nil (weights exist but don't load)
- 80% implemented, stuck on tensor loading

## 📋 NEXT
1. Investigate vision_bridge array support
2. If not supported → manual loading
3. Test projection works

**Start**: Read `QUICK_START.md` then investigate vision_bridge.go

---

## 📁 DIRECTORY STRUCTURE

```
1_split_ollama/
├── README.md              # This file - Project overview (30 sec read)
├── QUICK_START.md         # Immediate action guide (2 min read)
├── CODE_CHANGES.md        # Exact files and lines modified
│
├── docs/                  # Technical documentation
│   ├── SPLIT_MODEL_ANALYSIS.md   # Deep architecture analysis
│   └── PR_REFERENCE.md           # How PR #13305 worked (llama.cpp)
│
├── fix/                   # Task tracking
│   └── FIX_TRACKING.md           # Completed/failed/remaining tasks
│
└── logsToCompara/         # Reference logs
    └── PR-RunLlama.log           # Working llama.cpp output
```

**Reading order for new session**:
1. `README.md` (this file) → overview
2. `QUICK_START.md` → what to do
3. `CODE_CHANGES.md` → what was changed
4. `docs/` → if need technical details
5. `fix/FIX_TRACKING.md` → if need task breakdown
