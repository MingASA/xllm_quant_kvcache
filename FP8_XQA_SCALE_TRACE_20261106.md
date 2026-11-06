<!-- Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0. -->

# XQA internal Query conversion and the frozen V-scale change

**The V=.5 candidate does not shrink Q before its FP8 conversion.** In the
inspected FlashInfer 0.6.18.post1 SM90 path, BF16 API Q is converted directly to
E4M3 with unit scale. The wrapper's `q_scale` compensates the shared KV scale
after QK accumulation; it is not the divisor used to quantize Q.

## Source trace

1. `xllm/python/attention/xqa_decode.py:99` passes
   `bmm1_scale=softmax_scale*key_scale`, `bmm2_scale=value_scale`. The current
   contract has D=256, softmax_scale=1/16, K scale=.125, BF16 output, and the
   default `o_scale=1`.
2. FlashInfer `decode.py:3770` sets `kv_scale=V` and
   `q_scale=(softmax_scale*K/V)*sqrt(D)`. It reshapes Q without scaling its
   elements. `xqa.py:385` selects the SM90 FP8 route for FP8 cache and D<=256.
3. `jit/xqa.py` selects `INPUT_FP16=0`, `CACHE_ELEM_ENUM=2`, `SPEC_DEC=0`,
   `LOW_PREC_OUTPUT=0`, `USE_INPUT_KV=0`, and includes `mha_sm90.cu` when an
   SM90 build target exists. `xqa_wrapper.cu:102` forwards the Q pointer and
   scalar scales to `launchHopperF8MHAFlashInfer`.
4. `mha_sm90.cu:1610` loads BF16 Q without scaling. At line 1679,
   `F16QToF8Converter::store` performs `CacheElem{fp16Data[i]}` directly;
   `CacheElem` is `__nv_fp8_e4m3`. There is no multiplication/division by
   `q_scale`, K scale, or V scale before this conversion. The apparent
   `1/kvCacheScale` conversions elsewhere belong to the disabled
   `USE_INPUT_KV` branch, not this resident-cache path.
5. At `mha_sm90.cu:781`,
   `qkScale=q_scale*V*rsqrt(D)`, applied to QK accumulators at line 862.
   The two fixed configurations give the identical FP32 product
   `q_scale*V=.125` before the common reciprocal square root.
6. CUDA's E4M3 BF16 constructor uses saturating RNE conversion
   (`cuda_fp8.hpp:1616`, conversion at line 442). Thus the effective Q operand
   is `E4M3(Q)`, not `E4M3(Q/q_scale)*q_scale`.

For finite input, writing `K8=E4M3(K/.125)` and `V8=E4M3(V/sV)`, the intended
logit expression is `E4M3(Q) @ K8.T * (.125/16)` in both configurations.
The intended reconstructed V operand is `sV*V8`.

| Quantity | Current V=.0625 | Frozen V=.5 |
|---|---:|---:|
| Wrapper `q_scale` | 2 | .25 |
| Actual Q conversion scale | 1 | 1 |
| Effective Q finite range | ±448 | ±448 |
| Smallest positive Q FP8 value | 2^-9 = .001953125 | same |
| Nonzero Q rounds to zero when abs(Q) <= | 2^-10 = .0009765625 | same |
| Intended QK multiplier | 1/128 | 1/128 |
| Reconstructed V finite range | ±28 | ±224 |
| Smallest positive reconstructed V | 2^-13 = .0001220703125 | 2^-10 = .0009765625 |
| Nonzero V rounds to zero when abs(V) <= | 2^-14 = .00006103515625 | 2^-11 = .00048828125 |

The zero boundaries include the midpoint, because RNE ties select zero.
The scale change shifts V's exponent window by three bits. It extends the
large-value range and raises the small-value zero threshold eightfold; it
does not increase normal-range relative precision. Interpreting wrapper
`q_scale=.25` as pre-quantization attenuation of Q would be incorrect.

## Existing independent statistics and their limits

The BF16 observer retained no activation tensors and did not feed quantized
Q/K/V back into the model. Its `query_decode_FP8_error_is_hypothetical=true`
remains valid: it measures hypothetical rounding on the BF16 trajectory,
not XQA's actually loaded Q bytes or quantized-model activations.

| Existing observation | Prefill | Pure decode |
|---|---:|---:|
| Q maximum absolute value | 19.75 | 16.875 |
| Q clipped at unit scale | 0 | 0 |
| Q nonzero-to-zero / finite values | 28,474,619 / 42,376,790,016 | 7,066,693 / 9,712,140,288 |
| Q nonzero-to-zero fraction | .0671939% | .0727614% |
| V current MSE | .023419840514798163 | .015137892733736823 |
| V candidate MSE | .0024483582606331 | .0023153906705902366 |
| V candidate MSE decrease | 89.5458% | 84.7047% |

Across phases, V clipping falls from 749,348 to zero and MSE from
.021875646033588377 to .0024235660469954477; V nonzero-to-zero rises from
581,418 to 4,657,459 out of 8,681,488,384 observations (.0066972% to .0536482%).
These are derived from the existing independent snapshot, without reading
formal answers or changing a scale candidate.

The source trace preserves the original V gate's interpretation: **better
pointwise V reconstruction on this independent BF16 cohort**, with extra
small-value loss. It does not prove better attention outputs or model quality.
Q rounding, K rounding, softmax, and probability rounding remain separate
errors. In particular, this SM90 kernel computes the softmax sum before
mapping exponentials to E4M3 (`mha_sm90.cu:898–906`) and applies V/448 in its
output scaling (`:965–971`); Q/K/V MSE alone does not measure that path.
The increased V scale introduces no additional Q underflow in this source
path, but quantized-model feedback can change the later Q/K/V distribution.

## Evidence boundary and minimal remaining checks

The local `decode.py` hash matches the recorded H800 package. Other inspected
source/header hashes below have not been newly matched to the remote compiled
module in this review. This is source-derived behavior, not new GPU or SASS
evidence. Before attributing actual dispatch precision, retain a read-only
source/build receipt for the imported XQA files and the relevant JIT flags.
No extra numerical search is required.

For the isolated final candidate, the already planned original-fixture writer
byte checks and direct FA3/XQA checks at V=.0625/.5 are sufficient additional
operator scope. Keep the frozen formal model quality and service gates.
No need to rerun independent calibration solely because of this source trace.

Inspected dependency SHA256 values:

| File relative to the FlashInfer package | SHA256 |
|---|---|
| `decode.py` | `d82d107a644596a9349780b839b34e690c50169ea4cea3d0ee02b78e9dc88c1a` |
| `xqa.py` | `6a3f7330980b976201cff0fcdf51e2958e454c4c5951ad2b30c1f2aa13722d33` |
| `jit/xqa.py` | `b47625e738bcef4324e874a20fa4c3aee91c477dafd72b24480e3d235c3d20cc` |
| `data/csrc/xqa/xqa_wrapper.cu` | `31fbba7d0e912091b9aa16a1fd28338372be56b4543edcf556e21e33924a34aa` |
| `data/csrc/xqa/mha_sm90.cu` | `1edd5708fafae8667c9b6ad3119c00d373ad53104bb4f696cb5409423cc7b068` |
| `data/csrc/xqa/defines.h` | `722dd1cba129d23ba532bdab69c15c7255d6870af52d0d3e1d5667734e752973` |

The inspected local CUDA header is `nvidia/cu13/include/cuda_fp8.hpp`, SHA256
`6dcc7d10e4ed4d3458429f0da99837ce338ef1b417034422a3f2d36c2a184c7f`.
