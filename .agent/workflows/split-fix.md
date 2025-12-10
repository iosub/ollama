---
description: Mandatory rules for working on the split-qwen3vl project
---

# MANDATORY RULES FOR SPLIT-QWEN3VL

## BEFORE DOING ANYTHING

1. READ z_iosu_2/1_split_ollama/OBJETIVO_SPLIT.md - The goal that NEVER changes
2. READ z_iosu_2/1_split_ollama/seguir.md - Test history and findings
3. READ z_iosu_2/1_split_ollama/HECHOS.md if exists - Confirmed facts

## MAIN OBJECTIVE

Make the split model split-qwen3vl-8b:latest return C25-16499-R using Ollama Engine Go.

IT IS NOT:
- Extracting invoice data that is just the test
- Using llama.cpp runner
- Modifying the GGUF model

IT IS:
- Fixing the Go code in Ollama to properly support split models

## AFTER EACH TEST

1. Update seguir.md with:
   - Test number
   - What was changed
   - Exact result
   - Whether it worked or not

## AFTER EACH IMPORTANT FINDING

1. Add to HECHOS.md:
   - The confirmed fact
   - Evidence log line number
   - Why it matters

## BEFORE MODIFYING CODE

1. CHECK in seguir.md if we already tried that change
2. DO NOT REPEAT changes that already failed
3. EXPLAIN why this change will be different

## IF USER SAYS STOP OR DO NOT MODIFY

1. STOP IMMEDIATELY
2. DO NOT make any more code changes
3. UPDATE documentation with current state

## KEY FILES

- model/models/qwen3vl/model_vision.go - Main vision encoder code
- model/models/qwen3vl/model.go - Model and tensor loading
- model/models/qwen3vl/imageprocessor.go - Image processing
- z_iosu_2/funcionaaaaaaaaa/ollama/ - Reference code that WORKS non-split

## MANDATORY COMPARISON

Non-split qwen3-vl:8b-instruct returns C25-16499-R CORRECT
Split split-qwen3vl-8b:latest must return C25-16499-R TARGET

## REFERENCE LOGS

- z_iosu_2/logs/nosplit14.log - Non-split model working
- z_iosu_2/logs/viejos/resultadoBien.md - Complete correct result
