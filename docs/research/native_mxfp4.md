# OCP MXFP4 development branch

This branch descends from the FP8 PR implementation and adds only standard
OCP MXFP4. NVFP4 is not admitted here. Source modules, packed page geometry,
Writer, the private FlashInfer FA2 adapter and the Graph padding-mask repair
are retained from `6e99ad95` with the final **full planner** active. The rejected
fast-replanning candidate remains inactive; its historical branch is preserved.
The FP8 parent supplies the retained Query kernel and first-use XQA workspace
initialization for FP8 mode.

The actual format is E2M1 with one E8M0 scale per 32 values, not block-16 E4M3.
At D256 a K or V head occupies 128 payload bytes plus 8 scale bytes. Both
planes alias one 136-byte-per-head owning page. Whole-page copies preserve
both; GDN state retains its original dtype. The fused Writer and attention's
local tile unpack do not maintain a full BF16 history cache.

Supported research scope is non-VMM SM90 BF16, D256, Q24/KV4, page128, full
attention with unit global scales. Use `kv_cache_dtype=mxfp4`, FA2 Prefill and
FlashInfer Decode. Sliding windows, host offload, external stores, PD transfer
and VMM are excluded. Generic packed-page geometry remains research material;
it is not a CUDA VMM qualification claim.

The pinned FlashInfer source adapter and Graph source repair are deliberately
isolated to this development branch. Their private-API dependency and tested
shape restrictions make them unsuitable for the FP8 production PR.
The original qualification tool and its tests are retained under `tools/`
and `tests/python/`. Source imports were ordered for linting; the qualification
tool remains an unchanged archival harness. It is not run by a service launch.

MXFP4 remains experimental: the completed 1,720-core-generation quality matrix,
32 tool tasks and three service workloads did not establish production
acceptance. Detailed results, graphs, raw evidence and preserved failures are
in the [completed report](https://github.com/MingASA/kv_research_agent/blob/bc260191a6f36f0a1517d4a2d1313856f2e58dab/reports/experiments/mxfp4-p0-20261109/REPORT.md).
Rebranching and CPU tests do not replace those results or claim a new GPU run.
