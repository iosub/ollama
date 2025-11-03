PS C:\IA\tools\ollama\llama> cd .\vendor\
PS C:\IA\tools\ollama\llama\vendor> git status  
HEAD detached from 0de0a0157
nothing to commit, working tree clean
PS C:\IA\tools\ollama\llama\vendor> git show --oneline 070ff4d5356083d60b807bb34d36b31c3653a29e  -1                                       
fatal: bad object 070ff4d5356083d60b807bb34d36b31c3653a29e
PS C:\IA\tools\ollama\llama\vendor> git show --oneline https://github.com/ggml-org/llama.cpp/tree/070ff4d5356083d60b807bb34d36b31c3653a29e  -1
fatal: invalid object name 'https'.
PS C:\IA\tools\ollama\llama\vendor> git show --oneline 070ff4d5356083d60b807bb34d36b31c3653a29e  -1                                           
fatal: bad object 070ff4d5356083d60b807bb34d36b31c3653a29e
PS C:\IA\tools\ollama\llama\vendor> git remote -v
origin  https://github.com/JJJYmmm/llama.cpp.git (fetch)
origin  https://github.com/JJJYmmm/llama.cpp.git (push)
upstream        https://github.com/ggml-org/llama.cpp.git (fetch)
upstream        https://github.com/ggml-org/llama.cpp.git (push)
PS C:\IA\tools\ollama\llama\vendor> git status
HEAD detached from 0de0a0157
nothing to commit, working tree clean
PS C:\IA\tools\ollama\llama\vendor> cd llama/vendor; git status --short -b
cd : Cannot find path 'C:\IA\tools\ollama\llama\vendor\llama\vendor' because it does not exist.
At line:1 char:1
+ cd llama/vendor; git status --short -b
+ ~~~~~~~~~~~~~~~
    + CategoryInfo          : ObjectNotFound: (C:\IA\tools\oll...or\llama\vendor:String) [Set-Location], ItemNotFoundException
    + FullyQualifiedErrorId : PathNotFound,Microsoft.PowerShell.Commands.SetLocationCommand

## HEAD (no branch)
PS C:\IA\tools\ollama\llama\vendor> cd llama\vendor; git status --short -b
cd : Cannot find path 'C:\IA\tools\ollama\llama\vendor\llama\vendor' because it does not exist.
At line:1 char:1
+ cd llama\vendor; git status --short -b
+ ~~~~~~~~~~~~~~~
    + CategoryInfo          : ObjectNotFound: (C:\IA\tools\oll...or\llama\vendor:String) [Set-Location], ItemNotFoundException
    + FullyQualifiedErrorId : PathNotFound,Microsoft.PowerShell.Commands.SetLocationCommand

## HEAD (no branch)
PS C:\IA\tools\ollama\llama\vendor> git status --short -b
## HEAD (no branch)
PS C:\IA\tools\ollama\llama\vendor> git show --oneline -s HEAD
a2d9eed58 (HEAD) interleave multi rope
PS C:\IA\tools\ollama\llama\vendor> git remote -v
origin  https://github.com/JJJYmmm/llama.cpp.git (fetch)
origin  https://github.com/JJJYmmm/llama.cpp.git (push)
upstream        https://github.com/ggml-org/llama.cpp.git (fetch)
upstream        https://github.com/ggml-org/llama.cpp.git (push)
PS C:\IA\tools\ollama\llama\vendor> git fetch --all
Fetching origin
Fetching upstream
remote: Enumerating objects: 267, done.
remote: Counting objects: 100% (193/193), done.
remote: Compressing objects: 100% (75/75), done.
remote: Total 267 (delta 154), reused 118 (delta 118), pack-reused 74 (from 3)
Receiving objects: 100% (267/267), 2.65 MiB | 6.91 MiB/s, done.
Resolving deltas: 100% (159/159), completed with 66 local objects.
From https://github.com/ggml-org/llama.cpp
 * [new branch]          cisc/support-extended-embd -> upstream/cisc/support-extended-embd
 * [new branch]          gg/benches          -> upstream/gg/benches
   a4b54f269..d441c31b1  gg/clip-fa          -> upstream/gg/clip-fa
 * [new branch]          gg/context-skip-compute -> upstream/gg/context-skip-compute
 * [new branch]          gg/server-use-alias -> upstream/gg/server-use-alias
   2f68ce7cf..070ff4d53  master              -> upstream/master
   70cc3300a..79b98dbf9  xsn/mtmd_custom_min_max_tokens -> upstream/xsn/mtmd_custom_min_max_tokens
 * [new tag]             b6919               -> b6919
 * [new tag]             b6920               -> b6920
 * [new tag]             b6922               -> b6922
 * [new tag]             b6923               -> b6923
 * [new tag]             b6924               -> b6924
 * [new tag]             b6927               -> b6927
 * [new tag]             b6929               -> b6929
 * [new tag]             b6931               -> b6931
 * [new tag]             b6932               -> b6932
 * [new tag]             b6933               -> b6933
 * [new tag]             b6934               -> b6934
PS C:\IA\tools\ollama\llama\vendor> git branch -a
* (HEAD detached from 0de0a0157)
  master
  pr16780
  remotes/origin/HEAD -> origin/master
  remotes/origin/add_qwen3vl
  remotes/origin/kvshift_imrope
  remotes/origin/master
  remotes/origin/qwen3vl-1022
  remotes/origin/qwen3vl_tmp
  remotes/upstream/0cc4m/vulkan-coopmat-amd-windows
  remotes/upstream/0cc4m/vulkan-device-architecture
  remotes/upstream/0cc4m/vulkan-enable-conv2d-on-apple
  remotes/upstream/0cc4m/vulkan-fix-host-memory-max-size
  remotes/upstream/0cc4m/vulkan-fix-mm-tests
  remotes/upstream/0cc4m/vulkan-fprintf-fix
  remotes/upstream/0cc4m/vulkan-instance-cleanup
  remotes/upstream/0cc4m/vulkan-mmq-dp4a-vec-k-quants
  remotes/upstream/0cc4m/vulkan-op-opt-step-sgd
  remotes/upstream/0cc4m/vulkan-print-coopmat-shapes
  remotes/upstream/0cc4m/vulkan-renderdoc
  remotes/upstream/0cc4m/vulkan-suballoc-1gb
  remotes/upstream/0cc4m/vulkan-subgroup-size-control-amd
  remotes/upstream/7507-main-intel-dockerfile
  remotes/upstream/HEAD -> upstream/master
  remotes/upstream/SVE-vector-length-agnostic-VLA-gg
  remotes/upstream/add-gemma2-soft-capping
  remotes/upstream/add-pidfile
  remotes/upstream/alloc-assert-fix
  remotes/upstream/apply-3585
  remotes/upstream/assert-restore-abort
  remotes/upstream/avoid-gnu-source
  remotes/upstream/batched-bench
  remotes/upstream/build-metal-default
  remotes/upstream/cam-simple-fix
  remotes/upstream/ceb/bert
  remotes/upstream/ceb/bert-tokenizer-fixes
  remotes/upstream/ceb/convert-hf-refactor
  remotes/upstream/ceb/convert-vocab-fallback
  remotes/upstream/ceb/fix-badspecial-silentfail
  remotes/upstream/ceb/fix-cmake-typo
  remotes/upstream/ceb/fix-cuda-warning-flags
  remotes/upstream/ceb/fix-draft-model-default
  remotes/upstream/ceb/fix-logit-check
  remotes/upstream/ceb/fix-msvc-build
  remotes/upstream/ceb/fix-win-unicode-fpaths
  remotes/upstream/ceb/fix-yarn-neox
  remotes/upstream/ceb/libstdcpp-assertions
  remotes/upstream/ceb/nomic-bert
  remotes/upstream/ceb/nomic-vulkan-fix-add
  remotes/upstream/ceb/perf-faster-multigpu
  remotes/upstream/ceb/restore-convert
  remotes/upstream/ceb/wpm-portable-tolower
  remotes/upstream/cedo/add-outetts-v0.3
  remotes/upstream/cedo/fix-q25vl
  remotes/upstream/chunks
  remotes/upstream/ci-android
  remotes/upstream/ci/server/fix-slow-test
  remotes/upstream/ci_cublas
  remotes/upstream/cisc/support-extended-embd
  remotes/upstream/cisc/test-tokenizers-remote
  remotes/upstream/codeplay/dequant_q4_K_improvements
  remotes/upstream/codeplay/fix-matmul-arith
  remotes/upstream/codeplay/revert-host-alloc
  remotes/upstream/codeplay/sycl-main
  remotes/upstream/codeplay/tg-warmup
  remotes/upstream/compilade/batch-splits
  remotes/upstream/compilade/bitnet-ternary
  remotes/upstream/compilade/convert-hf-refactor
  remotes/upstream/compilade/convert-prequant
  remotes/upstream/compilade/convert-reflinks
  remotes/upstream/compilade/convert-safetensors-parse
  remotes/upstream/compilade/cuda-falcon-h1
  remotes/upstream/compilade/cuda-tq2_0
  remotes/upstream/compilade/faster-lazy-safetensors
  remotes/upstream/compilade/faster-session-sizes
  remotes/upstream/compilade/fix-batch-reserve-rwkv
  remotes/upstream/compilade/fix-command-r
  remotes/upstream/compilade/fix-convert-gemma-1-instruct
  remotes/upstream/compilade/fix-metadata-name-extraction
  remotes/upstream/compilade/fix-mpt-pretok
  remotes/upstream/compilade/fix-output-overflow
  remotes/upstream/compilade/fix-prequant-mxfp4-gpt-oss
  remotes/upstream/compilade/fix-pydantic-example
  remotes/upstream/compilade/fix-qp-iq1-problems
  remotes/upstream/compilade/fix-recurrent-batch-init
  remotes/upstream/compilade/fix-server-long-system-prompt
  remotes/upstream/compilade/fix-server-tests-penalty
  remotes/upstream/compilade/fix-ssm-scan-groups
  remotes/upstream/compilade/gguf-py-dequant
  remotes/upstream/compilade/gguf-py-fix-old-numpy
  remotes/upstream/compilade/gguf-py-fix-q-shape
  remotes/upstream/compilade/gguf-py-mxfp4
  remotes/upstream/compilade/gguf-py-quants-class
  remotes/upstream/compilade/hybrid-kv_unified
  remotes/upstream/compilade/imatrix-batched-chunks
  remotes/upstream/compilade/imatrix-gguf-default
  remotes/upstream/compilade/imatrix-gguf-warning
  remotes/upstream/compilade/imatrix-mxfp4
  remotes/upstream/compilade/imatrix-neutral-prior
  remotes/upstream/compilade/imatrix-saner-3d
  remotes/upstream/compilade/lazier-moe-convert-hf
  remotes/upstream/compilade/lazy-bfloat16-convert-hf
  remotes/upstream/compilade/lazy-convert-hf
  remotes/upstream/compilade/lazy-tuples
  remotes/upstream/compilade/mamba2
  remotes/upstream/compilade/nul-str-token
  remotes/upstream/compilade/optimal-rounding
  remotes/upstream/compilade/output-reorder-lazy-sort
  remotes/upstream/compilade/parallel-convert
  remotes/upstream/compilade/pyright-fix-ignores
  remotes/upstream/compilade/pyright-tests
  remotes/upstream/compilade/q8_0-convert-hf
  remotes/upstream/compilade/readonly-recurrent-inputs
  remotes/upstream/compilade/refactor-kv-cache
  remotes/upstream/compilade/refactor-kv-cache-gg
  remotes/upstream/compilade/refactor-session-files
  remotes/upstream/compilade/requirements-cpu-torch
  remotes/upstream/compilade/superbpe
  remotes/upstream/compilade/test-model-random
  remotes/upstream/compilade/tokenize-example-parse-special
  remotes/upstream/cuda-batched-gemm
  remotes/upstream/cuda-batched-gemm-deq
  remotes/upstream/cuda-cublas-opts
  remotes/upstream/cuda-multi-gpu
  remotes/upstream/cuda-quantum-batch
  remotes/upstream/custom-attention-mask
  remotes/upstream/custom-attention-mask-no-roped-cache
  remotes/upstream/deploy
  remotes/upstream/dequantize-matmul-3-gg
  remotes/upstream/dev
  remotes/upstream/f16c
  remotes/upstream/fairydreaming/t5-clean-3-gg
  remotes/upstream/feat/s390x-cpu-feats
  remotes/upstream/fix-convert-modelname
  remotes/upstream/fix-eos
  remotes/upstream/fix-kv-cache-access
  remotes/upstream/fix-ninja-metallib-build
  remotes/upstream/fix-refact
  remotes/upstream/fix-sessions
  remotes/upstream/fix-tensor-split-zero
  remotes/upstream/fix_clblast
  remotes/upstream/fix_cmd_name
  remotes/upstream/fix_ctx_default
  remotes/upstream/fix_q_xxs_mul_mat
  remotes/upstream/fix_sycl_ci
  remotes/upstream/flash-attn
  remotes/upstream/flash-attn-cuda
  remotes/upstream/gabe-l-hart/HybridRecurrentCache
  remotes/upstream/gg/add-phi-3-support
  remotes/upstream/gg/add-phixtral
  remotes/upstream/gg/allow-kv-overrides
  remotes/upstream/gg/arm-try-fix-msvc
  remotes/upstream/gg/authors
  remotes/upstream/gg/avoid-mutex
  remotes/upstream/gg/batch-simplify-output
  remotes/upstream/gg/bench-handle-decode-errors
  remotes/upstream/gg/benches
  remotes/upstream/gg/bert-f16
  remotes/upstream/gg/bitnet
  remotes/upstream/gg/bpe-preprocess
  remotes/upstream/gg/build-linux-static
  remotes/upstream/gg/build-pack-lib-include
  remotes/upstream/gg/cache-token-to-piece
  remotes/upstream/gg/cb-naming
  remotes/upstream/gg/check-python-version
  remotes/upstream/gg/ci-add-arm-msvc-toolchain
  remotes/upstream/gg/ci-fix-save-load
  remotes/upstream/gg/ci-loongson
  remotes/upstream/gg/ci-python
  remotes/upstream/gg/ci-rename-job
  remotes/upstream/gg/clang-tidy-disable-bugprone
  remotes/upstream/gg/clip-fa
  remotes/upstream/gg/cmake-dedup-link
  remotes/upstream/gg/compare-change-path
  remotes/upstream/gg/compare-mlx
  remotes/upstream/gg/context-fix-enc-attn-type
  remotes/upstream/gg/context-remove-logits-all
  remotes/upstream/gg/context-skip-compute
  remotes/upstream/gg/context-sync-upon-output-reorder
  remotes/upstream/gg/convert-fix-byte-tokens
  remotes/upstream/gg/cpu-fix-cpy-iq
  remotes/upstream/gg/cublas-f32
  remotes/upstream/gg/disable-sgemm
  remotes/upstream/gg/encode-pad-equal
  remotes/upstream/gg/fa-no-kq-pad
  remotes/upstream/gg/fa-req-kq-hs
  remotes/upstream/gg/fix-amx
  remotes/upstream/gg/fix-android
  remotes/upstream/gg/fix-build-gf
  remotes/upstream/gg/fix-cpu-blas
  remotes/upstream/gg/fix-devops
  remotes/upstream/gg/fix-embeddings-wip
  remotes/upstream/gg/fix-fa-q-non-cont
  remotes/upstream/gg/fix-min-max
  remotes/upstream/gg/fix-python-names
  remotes/upstream/gg/fix-spm-added-tokens-dict-4958
  remotes/upstream/gg/fix-starcoder2
  remotes/upstream/gg/fix-sve
  remotes/upstream/gg/fix-vld1q_s8_x4-4872
  remotes/upstream/gg/flash-attn
  remotes/upstream/gg/flash-attn-32x8
  remotes/upstream/gg/flash-attn-a
  remotes/upstream/gg/flash-attn-cuda
  remotes/upstream/gg/flash-attn-interleave-cc
  remotes/upstream/gg/flash-attn-mask-f16
  remotes/upstream/gg/flash-attn-online
  remotes/upstream/gg/flash-attn-rebase
  remotes/upstream/gg/flash-attn-simd
  remotes/upstream/gg/flash-attn-sync
  remotes/upstream/gg/flash-attn-wip
  remotes/upstream/gg/flash-attn-wip2
  remotes/upstream/gg/flash-attn-wip3
  remotes/upstream/gg/flash-attn-wip4
  remotes/upstream/gg/float-pos
  remotes/upstream/gg/ggml-atomic-int
  remotes/upstream/gg/ggml-cont
  remotes/upstream/gg/ggml-fix-zero-blocks
  remotes/upstream/gg/ggml-rework-cgraph
  remotes/upstream/gg/ggml_scale
  remotes/upstream/gg/gguf-fix-null-defer
  remotes/upstream/gg/gguf-py-0.11.0
  remotes/upstream/gg/gpu-prec-tests
  remotes/upstream/gg/grammar-refactor
  remotes/upstream/gg/graph-mamba-reuse
  remotes/upstream/gg/graph-prec
  remotes/upstream/gg/graph-reuse-reset-fix
  remotes/upstream/gg/hf
  remotes/upstream/gg/hf-args
  remotes/upstream/gg/hf-auto-dl
  remotes/upstream/gg/hf-test
  remotes/upstream/gg/hparams-swa-rope
  remotes/upstream/gg/http-threads
  remotes/upstream/gg/imatrix-gpu-4931
  remotes/upstream/gg/imatrix-remove-assert
  remotes/upstream/gg/indent
  remotes/upstream/gg/infill-better-stop
  remotes/upstream/gg/iq2-refactor-and-tests
  remotes/upstream/gg/kv-cache-prepare-separation
  remotes/upstream/gg/kv-compress
  remotes/upstream/gg/kv-determinism
  remotes/upstream/gg/kv-fix-shift
  remotes/upstream/gg/lfs
  remotes/upstream/gg/llama-add-log
  remotes/upstream/gg/llama-disambiguate
  remotes/upstream/gg/llama-high-throughput-rebase
  remotes/upstream/gg/llama-high-throughput-save
  remotes/upstream/gg/llama-high-throughput-save2
  remotes/upstream/gg/llama-kv-cache
  remotes/upstream/gg/llama-refactor-sampling
  remotes/upstream/gg/llama-reorganize
  remotes/upstream/gg/llama-shadow-on
  remotes/upstream/gg/llama3-support
  remotes/upstream/gg/logits-slowdown
  remotes/upstream/gg/mamba-fix-squeeze
  remotes/upstream/gg/media-add-svg-logo
  remotes/upstream/gg/metal-alloc-size
  remotes/upstream/gg/metal-async
  remotes/upstream/gg/metal-async-save-global-queue
  remotes/upstream/gg/metal-batched
  remotes/upstream/gg/metal-dequant-align
  remotes/upstream/gg/metal-disable-fa-256
  remotes/upstream/gg/metal-embed
  remotes/upstream/gg/metal-f16
  remotes/upstream/gg/metal-fa-f16
  remotes/upstream/gg/metal-fa-f16-save
  remotes/upstream/gg/metal-fa-vec-bs20
  remotes/upstream/gg/metal-fattn-reqs
  remotes/upstream/gg/metal-fix-build
  remotes/upstream/gg/metal-fix-fa
  remotes/upstream/gg/metal-fix-fa-2
  remotes/upstream/gg/metal-fix-thread-safety
  remotes/upstream/gg/metal-fuse-add-rms
  remotes/upstream/gg/metal-mm-pad
  remotes/upstream/gg/metal-mmid-max-rows
  remotes/upstream/gg/metal-mul-mat-f16
  remotes/upstream/gg/metal-mul-mat-write-opt
  remotes/upstream/gg/metal-mul-mm-rework
  remotes/upstream/gg/metal-mul-mv-new
  remotes/upstream/gg/metal-mul-mv-new-save2
  remotes/upstream/gg/metal-mul-mv-new-save3
  remotes/upstream/gg/metal-mul-mv-opt-2
  remotes/upstream/gg/metal-opt-mul-mat-id
  remotes/upstream/gg/metal-q4_0-opt
  remotes/upstream/gg/metal-refactor-mv-2
  remotes/upstream/gg/metal-reuse-graphs
  remotes/upstream/gg/metal-use-virtual-gpu-address
  remotes/upstream/gg/min-p-fix
  remotes/upstream/gg/model-cards
  remotes/upstream/gg/nix-remove-opencl
  remotes/upstream/gg/no-char32_t
  remotes/upstream/gg/ops-update-blas
  remotes/upstream/gg/ot-cpu-repack
  remotes/upstream/gg/pad-kv-cache
  remotes/upstream/gg/per-layer-kv
  remotes/upstream/gg/phi-2
  remotes/upstream/gg/phi-2-2
  remotes/upstream/gg/plamo-test
  remotes/upstream/gg/py-minor-fixes
  remotes/upstream/gg/quantize-fallback
  remotes/upstream/gg/quantum-k-cache
  remotes/upstream/gg/refactor-alibi-2
  remotes/upstream/gg/remove-gqa-check-4657
  remotes/upstream/gg/remove-instruct
  remotes/upstream/gg/remove-k-quants-per-iter
  remotes/upstream/gg/rename-n_ctx
  remotes/upstream/gg/repack-fix-mul-mat-id
  remotes/upstream/gg/repack-fix-wsize
  remotes/upstream/gg/repack-opt-mm-id
  remotes/upstream/gg/repeng
  remotes/upstream/gg/replace-all
  remotes/upstream/gg/rmse_quantization
  remotes/upstream/gg/rpc-fix-misaligned
  remotes/upstream/gg/script-bench-models
  remotes/upstream/gg/server-chunked-prefill
  remotes/upstream/gg/server-debug-win
  remotes/upstream/gg/server-eos-pre-calc
  remotes/upstream/gg/server-fix-ignore-eos
  remotes/upstream/gg/server-fix-infill
  remotes/upstream/gg/server-fix-prompt
  remotes/upstream/gg/server-fix-spec
  remotes/upstream/gg/server-fix-spec-ctx-shift
  remotes/upstream/gg/server-fix-vision-tests
  remotes/upstream/gg/server-infill-empty-prompt-4027
  remotes/upstream/gg/server-infill-end-on-nl
  remotes/upstream/gg/server-logs
  remotes/upstream/gg/server-models-loading
  remotes/upstream/gg/server-reenable-swa-spec
  remotes/upstream/gg/server-test-lru
  remotes/upstream/gg/server-unified-slots
  remotes/upstream/gg/server-update-js
  remotes/upstream/gg/server-use-alias
  remotes/upstream/gg/server-v1-completion
  remotes/upstream/gg/soft-max-ext
  remotes/upstream/gg/speculative-experiments
  remotes/upstream/gg/speculative-fix-oob
  remotes/upstream/gg/speculative-infill
  remotes/upstream/gg/speculative-update
  remotes/upstream/gg/survey-nvidia
  remotes/upstream/gg/swa-fix-kv-shift
  remotes/upstream/gg/swiftui-bench
  remotes/upstream/gg/system-info-llamafile
  remotes/upstream/gg/test-arm
  remotes/upstream/gg/test-bench
  remotes/upstream/gg/test-embd
  remotes/upstream/gg/test-fp16
  remotes/upstream/gg/tfs-ob1
  remotes/upstream/gg/tmp-ci
  remotes/upstream/gg/tokenizer-cleanup
  remotes/upstream/gg/try-fix-sycl-iq1_s
  remotes/upstream/gg/ttfb
  remotes/upstream/gg/tts-fix-ubatch
  remotes/upstream/gg/unary-non-cont
  remotes/upstream/gg/unicode-refactor
  remotes/upstream/gg/unified-fix-k-shift
  remotes/upstream/gg/update-phi2-convert
  remotes/upstream/gg/vocab-fix-no-vocab
  remotes/upstream/ggml-backends
  remotes/upstream/ggml-backends-metal
  remotes/upstream/ggml-impl
  remotes/upstream/ggml-quants
  remotes/upstream/gguf
  remotes/upstream/gguf-64bit
  remotes/upstream/gguf-fix-publish
  remotes/upstream/gguf-pip
  remotes/upstream/gguf-publish-ci
  remotes/upstream/gguf-python
  remotes/upstream/gguf-write-single-pass
  remotes/upstream/gguf-write-tensor
  remotes/upstream/graph-profiler
  remotes/upstream/gritlm-pr
  remotes/upstream/hp/tmp/kv-cache-defrag
  remotes/upstream/ik/better_q2_k_s
  remotes/upstream/ik/even_better_iq1s
  remotes/upstream/ik/faster_hellaswag
  remotes/upstream/ik/fix_hellaswag
  remotes/upstream/ik/fix_iq3xxs_metal
  remotes/upstream/ik/fix_k_cache_backend_tests
  remotes/upstream/ik/fix_warnings
  remotes/upstream/ik/ggml-quants-cpp
  remotes/upstream/ik/i-quants-64
  remotes/upstream/ik/imatrix_legacy_quants
  remotes/upstream/ik/iq1_s
  remotes/upstream/ik/iq2_2.31bpw
  remotes/upstream/ik/iq3_s_faster
  remotes/upstream/ik/iq3_s_multiplier
  remotes/upstream/ik/quantize_not_repeating
  remotes/upstream/ik/quantize_with_kv_overrides
  remotes/upstream/ik/test_quantize_fns
  remotes/upstream/ik/try_fix_iq1s_sycl
  remotes/upstream/ik/try_fix_rocm_k_cache
  remotes/upstream/jared/permit-causal-encode
  remotes/upstream/jed/spm-clblast
  remotes/upstream/jg/cuda-fa-np-runtime
  remotes/upstream/jg/gguf-refactor
  remotes/upstream/jg/llama-opt-3
  remotes/upstream/jg/llama-sanitize
  remotes/upstream/kv-cache-opts
  remotes/upstream/llama-metadata
  remotes/upstream/llama-pull
  remotes/upstream/llama-refactor
  remotes/upstream/llama-refactor-norm
  remotes/upstream/llama_server_completions
  remotes/upstream/llama_server_timings
  remotes/upstream/llava-fix-offloading
  remotes/upstream/llm-build-context
  remotes/upstream/llm-reuse-constants
  remotes/upstream/lookahead
  remotes/upstream/lto
  remotes/upstream/master
  remotes/upstream/maxk/sched-prio-updates
  remotes/upstream/metal-cont-bug
  remotes/upstream/metal-fix-norm
  remotes/upstream/metal-improve-batching
  remotes/upstream/metal-soft-max
  remotes/upstream/mixtral
  remotes/upstream/mlx-challenge
  remotes/upstream/mmap
  remotes/upstream/mmap-pages-stats
  remotes/upstream/mul-mat-pad
  remotes/upstream/norm-quants
  remotes/upstream/norm-quants-rebase
  remotes/upstream/opencl-add-mul-mat-f16-f32-image
  remotes/upstream/passkey
  remotes/upstream/patch-1
  remotes/upstream/perf-study
  remotes/upstream/podman
  remotes/upstream/pr/4484
  remotes/upstream/pr_add_intel_amx_support
  remotes/upstream/prepare-PR-of-minicpm-v2.5-gg
  remotes/upstream/q4_0-q4_2-range-fix
  remotes/upstream/q4_1_more_accel
  remotes/upstream/q4_1_more_accel_kahan
  remotes/upstream/q4_1_more_accel_loopsplit
  remotes/upstream/q4_3-range-fix
  remotes/upstream/quant-attn
  remotes/upstream/refactor-mpi
  remotes/upstream/remove-vzip
  remotes/upstream/rev-sampling
  remotes/upstream/revert-11820-vers_fix
  remotes/upstream/revert-12734-fix_code_in_ggmlsycl
  remotes/upstream/revert-5901-fix_set_gpu
  remotes/upstream/revert-7777-host-usm-context-fix
  remotes/upstream/revert-pool
  remotes/upstream/rewrite-llama-run-to-be-llama-server-based
  remotes/upstream/rpc-hash-readme
  remotes/upstream/sampling-greedy-with-probs
  remotes/upstream/sampling-refactor
  remotes/upstream/scratch
  remotes/upstream/server-cfg
  remotes/upstream/server-oai-compat
  remotes/upstream/server-parallel
  remotes/upstream/server-rev
  remotes/upstream/shards-lang/gio/visionos-ci
  remotes/upstream/sl/aligned-alloc-no-abort
  remotes/upstream/sl/async-weight-copy
  remotes/upstream/sl/auto-flash-attn
  remotes/upstream/sl/cuda-f16-fix3
  remotes/upstream/sl/cuda-fattn-par-test
  remotes/upstream/sl/cuda-uma
  remotes/upstream/sl/detect-imatrix-nan
  remotes/upstream/sl/dio-test
  remotes/upstream/sl/disable-pp-nkvo
  remotes/upstream/sl/dump-allocs
  remotes/upstream/sl/fix-docker-main-server-build
  remotes/upstream/sl/fix-docker-omp
  remotes/upstream/sl/fix-omp-one-thread
  remotes/upstream/sl/fix-ppl-seq-max
  remotes/upstream/sl/fix-quant-kv-shift
  remotes/upstream/sl/fix-rpc-nkvo
  remotes/upstream/sl/fix-sched-reserve
  remotes/upstream/sl/llama-bench-headers
  remotes/upstream/sl/pr-releases
  remotes/upstream/sl/prepare-next-graph
  remotes/upstream/sl/rpc-backend-cpy
  remotes/upstream/sl/sched-copy-incr-fix
  remotes/upstream/sl/test-mul-mat-backend
  remotes/upstream/sl/zero-max-size
  remotes/upstream/speculative
  remotes/upstream/speculative-grammar
  remotes/upstream/speculative-tree
  remotes/upstream/steering
  remotes/upstream/support-starcoder-fix
  remotes/upstream/support_device_reg
  remotes/upstream/sycl-cmake-append
  remotes/upstream/sycl-conv-op
  remotes/upstream/sycl-global-variables
  remotes/upstream/sycl-mul-mat-id
  remotes/upstream/sycl-onednn-convolution
  remotes/upstream/sycl/disable_reorder_opt
  remotes/upstream/sycl_q3s_q1s
  remotes/upstream/sync-ggml-25-04-03-try-fix
  remotes/upstream/sync-ggml-25-05-01
  remotes/upstream/sync-ggml-25-07-19
  remotes/upstream/sync-ggml-25-07-25
  remotes/upstream/tcp_server
  remotes/upstream/test-bench
  remotes/upstream/test-mac-os-ci
  remotes/upstream/test-mmv
  remotes/upstream/try-fix-metal
  remotes/upstream/upd-issue-templates
  remotes/upstream/update_sycl_doc
  remotes/upstream/vb/add-smollm3
  remotes/upstream/xd/ops-musa
  remotes/upstream/xsn/ci_legacy_gg
  remotes/upstream/xsn/ggml_scale_bias
  remotes/upstream/xsn/mtmd_custom_min_max_tokens
  remotes/upstream/xsn/private_batch_api
  remotes/upstream/xsn/private_batch_api_pooling_none
  remotes/upstream/xsn/qwen3next_experiment
  remotes/upstream/xsn/tmp_jinja_safer
PS C:\IA\tools\ollama\llama\vendor> git show --oneline -s HEAD^C
PS C:\IA\tools\ollama\llama\vendor> git show --oneline -s upstream/master
070ff4d53 (upstream/master, upstream/HEAD) mtmd: add --image-min/max-tokens (#16921)
PS C:\IA\tools\ollama\llama\vendor> git show --oneline -s upstream/master
070ff4d53 (upstream/master, upstream/HEAD) mtmd: add --image-min/max-tokens (#16921)
PS C:\IA\tools\ollama\llama\vendor> git show --oneline -s master
0aca00577 (master) clip-unicode
PS C:\IA\tools\ollama\llama\vendor> git show --oneline -s origin/master
19a5a3edf (origin/master, origin/HEAD) ggml : Leverage the existing GGML_F32_VEC helpers to vectorize ggml_vec_set_f32 for faster fills (#16522)
PS C:\IA\tools\ollama\llama\vendor> git checkout master
Warning: you are leaving 29 commits behind, not connected to
any of your branches:

  a2d9eed58 interleave multi rope
  7d0f7c806 report LoadLibrary failures
  fb99e4e30 NVML fallback for unified memory GPUs
  04fb4ee92 GPU discovery enhancements
 ... and 25 more.

If you want to keep them by creating a new branch, this may be a good time
to do so with:

 git branch <new-branch-name> a2d9eed58

Switched to branch 'master'
Your branch and 'upstream/master' have diverged,
and have 3 and 23 different commits each, respectively.
  (use "git pull" if you want to integrate the remote branch with yours)
PS C:\IA\tools\ollama\llama\vendor> git reset --hard 070ff4d5356083d60b807bb34d36b31c3653a29e
HEAD is now at 070ff4d53 mtmd: add --image-min/max-tokens (#16921)
PS C:\IA\tools\ollama\llama\vendor> git status --short -b
## master...upstream/master
PS C:\IA\tools\ollama\llama\vendor> git show --oneline -s HEAD
070ff4d53 (HEAD -> master, upstream/master, upstream/HEAD) mtmd: add --image-min/max-tokens (#16921)
PS C:\IA\tools\ollama\llama\vendor> 

============================
hparams : add n_embd_full to support extended embed
https://github.com/ggml-org/llama.cpp/pull/16928