# Mixed-V KV-cache experiment snapshot, 2026-10-11

The completed service experiments use Qwen2.5-1.5B-Instruct on an RTX 5060 Ti
16GB with GDDR7 VRAM. K remains BF16; V is BF16, INT8, FP8 E4M3, or packed
INT4. All arms use the custom FlashInfer mixed-V backend in Python eager mode.
The measured implementation commit is `f944f205409db0dab1751abcf8962adad42caebb`.
This snapshot records results and diagnostics, without changing the backend or
formal experiment configuration.

Start with the [final service and quality summary](2026-10-11-final-mixed-v-cache-summary.md),
then the [Nsight Compute analysis](2026-10-11-nsight-compute-mixed-v-wall-analysis.md)
and [research versus implementation judgment](2026-10-11-literature-vs-implementation-performance-judgment.md).
The remaining reports preserve comparisons and the progression of the analysis.

Completed coverage:

- GSM8K: four arms at client concurrency 128/160/208, 1,319 prompts replayed
  twice per cell; 31,656 responses. Concurrency 256 was conditionally skipped
  because c208 did not show an overall service benefit.
- GovReport: four arms at concurrency 8/16/32/64/128, 200 prompts replayed twice
  per cell; 8,000 responses. This uses the documented 1,024-token output-cap
  extension and head/tail handling for three inputs. Reused BF16 runs have
  sequence cap 128, versus 256 for quantized runs; the final report explains
  this comparison limitation.
- Nsight Compute: six individual diagnostic attention kernels, outside formal
  service runs. Their profiled times must not replace service measurements.
  The split-KV cases profile the main attention kernel and exclude its merge.

Full raw responses, environment records, telemetry and profiler exports remain
in the E-drive archive:

- `/mnt/xllm-e/data/home/mingasa/projects/xllm_quant_kvcache/results/flashinfer-service-seq256-20261011/gsm8k-default-token-budget-seq256-c128-160-208-256`
- `/mnt/xllm-e/data/home/mingasa/projects/xllm_quant_kvcache/results/flashinfer-govreport-seq256-20261011/formal-two-passes`
- `/mnt/e/AI/xllm-storage/reports/2026-10-11-ncu-wall-*.ncu-rep`

`diagnostics/` preserves the scripts actually used, including machine-specific
paths and frozen correctness-gate checks. They are reproducibility snapshots,
not portable CLI tools. The NCU runner covers the five comparisons after the
initial BF16 b36 permission probe. For that initial probe use the same invocation
with mode 0, batch 36, no fixed split, and a fresh output/export path. Do not
execute the scripts while timing a formal service run. The summarizer reads
wide NCU CSV exports, where the first data row contains units.

`snapshot-manifest.json` records hashes of the copied evidence and scripts.
Large raw datasets, profiler binaries, build output and unrelated submodule
changes are outside this snapshot.
