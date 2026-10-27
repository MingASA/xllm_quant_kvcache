# Native low-precision KV on hybrid models

The native backend keeps Full Attention pages separate from Gated DeltaNet
convolution/recurrent state. Quantizing Full Attention does not quantize that
state. Cache rebinding checks payload, scales and state storage identities.

Select `--kv_cache_backend=flashinfer` with `--kv_cache_dtype=auto`,
`fp8_e4m3` or `nvfp4`. The typed configuration propagates through the service,
master, worker and Python executor; the Python LLM API exposes the same options.
`--kv_cache_k_scale` and `--kv_cache_v_scale` are positive normal FP32 global
decode scales. They need calibration for the actual model and request domain.

`--kv_cache_prefill_backend=auto|fa2|fa3` and
`--kv_cache_decode_backend=auto|flashinfer|xqa` select kernels explicitly.
Auto prefill selects FA3 for BF16/FP16 on SM90 and FA2 otherwise. Auto decode
preserves FlashInfer paged decode. XQA is opt-in and must pass the actual
hardware/head-dimension/GQA/length and graph checks before deployment.
Unsupported explicit selections fail before cache execution; they do not retry
using another kernel. Legacy research environment profiles remain available
only when the typed native backend is not selected.

FP8 E4M3 stores resident FP8 K/V pages. NVFP4 stores packed E2M1 pairs with one
E4M3 block scale per 16 components, plus global K/V scales. FlashInfer consumes
these compressed pages directly. BF16 queries require FA2 for both low-precision
formats; NVFP4 also requires FlashInfer tensor-core paged decode rather than
XQA. SM90 compressed FP4 storage does not imply native FP4 matrix multiplication.

`tools/validate_native_kv_pages.py` checks the writer, prefill/chunked/decode,
changed graph inputs and resident storage against an independent CPU byte
decoder. This operator check does not establish model quality or service
performance. C++ allocation/page-copy tests and complete model service checks
are separate gates. Quality and performance require traceable request outputs,
natural stopping, actual graph replay and production workloads.

The optional BF16 range observer is documented by its configuration and control
files in `xllm/python/attention/kv_observation.py`. It records maxima and finite
counts without storing activations. Its observations produce candidate scales,
not a quality certificate; keep it disabled for formal performance measurements.
