https://github.com/ggml-org/llama.cpp/pull/16764 -un ocr
https://github.com/ggml-org/llama.cpp/issues/16207#issuecomment-3443868720 qwen3vl discusion

llama.cpp
https://github.com/ggml-org/llama.cpp/issues/16207#issuecomment-3443868720
https://github.com/LETS-BEE/llama.cpp/commit/99719122bf16db5db85f0c2d37c059a3aefd3eca
llamapr
https://github.com/ggml-org/llama.cpp/pull/16745 hecho.
https://github.com/LETS-BEE/llama.cpp/commit/99719122bf16db5db85f0c2d37c059a3aefd3eca
https://github.com/LETS-BEE/llama.cpp/commit/b913e895a2189b9792da7709b36a36a1aed2feb9
https://github.com/LETS-BEE/llama.cpp/commit/de0e3d3c3ce4b394746ade9263736c8edb40260e
https://github.com/LETS-BEE/llama.cpp/commit/e45aecb7b051d3c0fea968d64aadbeb0b777e4a1

https://github.com/LETS-BEE/llama.cpp/commits/qwen3vl/



https://github.com/LETS-BEE/llama.cpp/commits/qwen3vl/

ollama
https://github.com/ollama/ollama/pull/12665 

time=2025-10-25T21:59:34.181-05:00 level=TRACE source=model.go:215 msg="found tensor" name=mm.mm_input_projection.weight type=f16 shape="[2560 1152]"
ggml.c:1921: GGML_ASSERT(ggml_can_repeat(b, a)) failed
time=2025-10-25T21:59:34.396-05:00 level=INFO source=sched.go:446 msg="Load failed" model=C:\Users\iosuc\.ollama\models\blobs\sha256-aeda25e63ebd698fab8638ffb778e68bed908b960d39d0becc650fa981609d25 error="do load request: Post \"http://127.0.0.1:62439/load\": read tcp 127.0.0.1:62444->127.0.0.1:62439: wsarecv: An existing connection was forcibly closed by the remote host."
time=2025-10-25T21:59:34.396-05:00 level=DEBUG source=server.go:1699 msg="stopping llama server" pid=12616
[GIN] 2025/10/25 - 21:59:34 | 500 |    1.5691132s |       127.0.0.1 | POST     "/api/generate"
time=2025-10-25T21:59:34.411-05:00 level=ERROR source=server.go:273 msg="llama runner terminated" error="exit status 0xc0000409"



================
do the patch files on Z_Iosu\patches and llama\patches
https://github.com/ollama/ollama/pull/12665/files
https://github.com/ollama/ollama/commit/d45f0045cef4ab70d6c5577745e1b866d9006837
