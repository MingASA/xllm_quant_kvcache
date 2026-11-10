# Native NVFP4 development branch

This branch descends from the FP8 PR implementation and adds only NVFP4.
MXFP4 is not admitted here. It preserves the previously implemented SM90
native FlashInfer FA2 writer/Prefill/Decode path from `3b49556a`.

The resident format is E2M1 packed payload plus E4M3 block-16 scales: 144 bytes
per D256 K or V head. Both planes alias the same owning byte page; whole-page
copies preserve payload and scales. GDN state retains the official dtype.
There is no full-cache BF16 shadow. SM90 uses tile-local conversion for MMA;
this branch makes no Blackwell FP4 Tensor Core claim.

Use `kv_cache_dtype=nvfp4`, FA2 Prefill and FlashInfer Decode, with the original
unit global scales. Qwen3.8-27B/H800 D256/Q24/KV4/page128 is the historical
model workload. Native VMM, host offload, external stores and PD transfer are
not admitted. Packed-page VMM geometry and its host tests are retained as
research infrastructure; those tests do not qualify CUDA VMM residency.

The scale-prevalidated writer (`nvfp4_writer.py`) and scheduler-host metadata
helper (`flashinfer_metadata.py`), with their boundary tests, are preserved
from `e710e6f2` as **inactive candidates**. The serving backend still uses the
historically measured native writer and ordinary planner. Do not infer a
candidate performance win or production qualification from CPU checks.
The full prior candidate harness remains on
`research/kv-p1-integration-20261109`.

The standalone backend import retained here supports independent operator
qualification. It is absent from the FP8 PR's common BF16 backend, which
matches the official source.

Historical quality, service data and failures remain in the
[frozen report](https://github.com/MingASA/kv_research_agent/tree/research/mxfp4-fp8-20261105/reports/experiments/mxfp4-fp8-20261105).
Rebranching is not a new GPU or end-to-end measurement. The FP8 PR is reviewed
independently of NVFP4. No old branch or result was deleted.
