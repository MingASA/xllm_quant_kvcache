# FlashInfer FP8 matched service smoke

Date: 2026-10-05. Checkpoint: `/mnt/e/AI/models/Qwen2.5-1.5B-Instruct`.

Both runs used the Python model implementation, graph modes off, memory utilization
0.80, `max_seqs_per_batch=8`, startup seed `20261005`, and the same
`/v1/chat/completions` endpoint and chat template. Each request used
`temperature=0`, `top_p=1`, `max_tokens=24`, and request seed `20261005`.
The ten prompts were one Chinese self-description, `What is 27 times 43? Answer
with the number only.`, and eight concurrent additions (`11+33` through
`18+54`). The first two requests were sequential; the additions were sent with
concurrency 8.

| KV cache | Backend setting | API results | Concurrent arithmetic | Wall time for 8 |
|---|---|---:|---:|---:|
| FP8 E4M3 | `XLLM_QUANTIZED_BACKEND=flashinfer` | 10/10 HTTP 200 | 0/8 correct | 1.213 s |
| BF16 auto | same environment; `kv_cache_dtype=auto` | 10/10 HTTP 200 | 8/8 correct | 0.910 s |

The multiplication prompt returned `28 times 43 equals 96.` with FP8
(`finish_reason=stop`, 12 completion tokens) and `1161` with BF16
(`finish_reason=stop`, 5 tokens). The Chinese prompt returned fluent Chinese in
both runs; FP8 used 17 tokens and stopped, while BF16 used all 24 tokens and
finished due to the token limit. All eight addition responses stopped normally
before the token limit. The FP8 outputs were:

| Prompt | FP8 E4M3 response | BF16 auto response |
|---|---|---|
| 11 + 33 | `The result of 1 + 33 is 4.` | `44` |
| 12 + 36 | `The result of 1 + 2 is 3.` | `48` |
| 13 + 39 | `The result of 3 + 39 is 10.` | `52` |
| 14 + 42 | `The result of 4 + 42 is 16.` | `56` |
| 15 + 45 | `The result of 5 + 45 is 10.` | `60` |
| 16 + 48 | `The result of 6 + 8 is 14.` | `64` |
| 17 + 51 | `The result of 7 + 51 is 8.` | `68` |
| 18 + 54 | `The result of 8 + 5 is 13.` | `72` |

Both servers reported identical prompt token counts for each request. The
arithmetic regressions are not caused by chat-template differences or
generation truncation: every arithmetic response ended with `stop`, and neither
run reached 24 tokens. This smoke confirms that requests execute through the
FP8 configuration, but it does not support an FP8 correctness claim. The
small-model sample indicates a substantial output-quality regression and needs
root-cause investigation before enabling this path by default.

Server logs were written to `/tmp/xllm-flashinfer-fp8-matched-20261005.log` and
`/tmp/xllm-bf16-auto-matched-20261005.log`. Both test servers were shut down;
port 18995 is free.

## Triton and HF diagnosis

Repeating the same ten requests with the existing dynamic Triton E4M3 backend
also returned 10/10 HTTP 200 but produced broken arithmetic (including
`\n\nuser\nWhat3` for the multiplication prompt); two concurrent responses
reached the 24-token limit. This regression is not unique to the FlashInfer
backend.

The local Hugging Face Qwen model was then run on the exact multiplication
prompt (45 chat-template tokens). A registered callback captured post-RoPE K/V
at all 28 layers and verified the causal mask in every full-prefill callback.
The layer-statistics pass appended “Please reason carefully.” to the user
content (49 prompt tokens); the logits/generation pass below used the exact
service multiplication content (45 tokens). K absolute max was 320 and V max
was 18.625, so neither exceeded E4M3's 448 limit. Per-layer relative L2 after
E4M3 QDQ was about 2.7% for fixed scale 1 and 2.4–2.5% for dynamic
per-token/head scaling; fixed-scale nonzero-to-zero rates were below 0.18%.

For the valid logit comparison, all modes called the same Qwen eager attention
implementation; only the K/V QDQ changed. No-quant wrapper logits matched
native eager logits exactly. With the correct answer `1161` teacher-forced,
fixed-scale K+V QDQ had logit relative L2 0.806 and 0% top-1 agreement with
BF16; dynamic K+V had 0.866 and 0%. K-only fixed-scale also diverged strongly
(0.781, 0%), while V-only stayed close (0.043, 100%). Greedy output for this
HF prompt was `1131` in BF16 and V-only, `27 times 43 is 83.` with fixed K+V,
`27 times 43 is 2743.` with K-only, and `\n\n7` with dynamic K+V. This points
to K quantization as the main sensitivity in this prompt. Top-1 agreement is
agreement with this HF BF16 run, not answer accuracy; that HF BF16 greedy run
itself returned `1131`, so it must not be compared as absolute accuracy against
the xLLM service's BF16 answer. This experiment does not isolate an online
serving bug or establish a general quality estimate, and does not attribute
the result to fixed scales or the FP8 library.

`hf_kv_qdq_eager_control.json` contains the valid five-arm teacher-forced and
greedy results; `hf_kv_qdq_logits.json` and
`hf_kv_qdq_logits_dense_control.json` are retained but marked invalid for
cross-mode inference comparisons because their quantized arms used FP32 dense
attention while BF16 eager used a different arithmetic path. Layer statistics
are in `hf_kv_qdq.json`.

The reproducible diagnostic sources are in `source/`: `hf_kv_qdq_stats.py`
captures the layer ranges and QDQ error; `hf_kv_qdq_eager_control.py` runs the
five matched eager-attention arms and causal-mask audit.

## Benchmark artifacts

The three benchmark runs and source snapshots are under:

- `../flashinfer-fp8-qwen-20261005/20261005T080704Z_1c22af2d`
- `../flashinfer-fp8-20261005/20261005T081442Z_55562968`
- `../flashinfer-fp8-long-20261005/20261005T081614Z_460bc622`

At context 4096, batch 4, query length 16, Qwen H12/KV2 BF16 vs E4M3 vs E5M2
attention p50 was 0.175/0.185/0.179 ms and full-call p50 was
0.234/0.258/0.240 ms; relative L2 was 0/0.038/0.076. Default H32/KV8
attention p50 was 0.476/0.310/0.261 ms and full-call p50 was
0.537/0.318/0.326 ms; relative L2 was 0/0.039/0.076. At context 32768,
batch 16, query length 16, Qwen H12/KV2 BF16 vs E4M3 attention p50 was
1.961/1.962 ms and full-call p50 was 2.039/2.060 ms, with E4M3 relative L2
0.039. FP8 cache payloads used 1.94x less storage, including the retained
scale allocation. Timings use synchronized `perf_counter` wall-clock samples;
they include Python dispatch, while `plan` is outside the timed call. They are
not pure kernel timings and do not show FP8 as universally faster.
