# inkling engine: brief

a from-scratch decode engine for inkling-2-mini on this m2 air (10-core GPU,
24GB, ~90GB/s usable). llama.cpp (`./build`, `llama/PLAN.md`) stays the daily
driver and the reference until milestone 2 passes.

## goal

UD-IQ3_XXS (`env`: `$M`, 11.7GB) at ~24-26 t/s serial decode (38-42 ms per
token) against 15 t/s now, same output, <= ~13GB in use at 100k context.
then dspark on top.

## why not more llama.cpp patches

measured 2026-11-21 on IQ3_XXS, tg32, this build:

- `GGML_METAL_PROF=1`: 10,982 graph nodes, 1 command buffer, GPU 64 ms,
  wall 70 ms, CPU encode 11-15 ms per token (not fully hidden).
- `GGML_METAL_PROF=2` (each node in its own command buffer, so times are
  inflated, ~103 ms total; read them as shares): 6,244 dispatches.

| op | dispatches | ms (isolated) | note |
| --- | ---: | ---: | --- |
| MUL_MAT q8_0 | 234 | 16.0 | attention, ~0.95GB/token: ~59GB/s |
| MUL_MAT_ID iq3_xxs / nvfp4 / iq2_s / iq4_nl | 94 | 32.4 | experts, ~1.3GB: ~40GB/s |
| MUL_MAT f32 | 624 | 9.0 | loop controllers, messages, small mats |
| MUL_MAT q5_K | 14 | 7.0 | head, 0.32GB: ~45GB/s |
| ~5k elementwise, norm, concat, conv, cpy | ~5,200 | ~35 | |

bytes per token ~2.8GB, so the bandwidth floor is ~31 ms. the gap is
expert mat-vec at half bandwidth, ~5k small kernels, f32 mat-muls and
unhidden encode. only the first is fixable inside ggml without rewriting
the graph; the rest is structural, and every rebase re-ports our patches.

## the model, in one place

26.4B params, 24 physical layers, layers 6-17 run 3x: 48 visits per token.
hidden 2304, 24 q heads / 8 kv heads x 128, no RoPE: a relative-position
bias (d_rel 16, extent 1024) and per-query sinks. 20 sliding-window layers
(window 1024) and 4 full-attention layers (log-scaled positions; 8 full
visits read the whole cache). short causal convolutions (kernel 4, state
kept per visit slot), qk/rel L2 clipping, block attention residuals over
depth (6 layers per block), cross-pass messages into looped layers,
128 experts top-8 with gate bias, softcapped gate/up, vocab 201,024
(200,058 real).

references, read in this order:

1. `llama/src-mix/src/models/inkling2.cpp`, `inkling2-loop.cpp`: the graph
   llama.cpp runs; ground truth for our numbers.
2. lookatmyquant `runtime_mlx/inkling/model.py` (MIT,
   github.com/ShinMegamiBoson/lookatmyquant): the same model as readable
   python, with fused decode kernels in `fast_kernels.py`.
3. `llama/patches/0001` (banded flash attention with bias and sink),
   `llama/patches/0002` (NVFP4 mat-vec lane fix).

## design

- rust host, metal shading language kernels, no ggml, mlx or build step
  beyond cargo. dependencies: metal bindings (`objc2-metal`) and little
  else.
- mmap the GGUF; wrap the pages in MTLBuffers with
  `newBufferWithBytesNoCopy` (page-aligned); one residency set.
- weight formats in the file: NVFP4, IQ3_XXS, IQ2_S, IQ2_XS, IQ2_XXS,
  IQ4_NL (experts); Q8_0, Q6_K, Q5_K, Q4_K (attention, head, embedding);
  F32 (controllers). port each dequant from ggml-metal (MIT); write the
  mat-vec loops ourselves. formats behind one trait, so a second file
  (requantized attention, or lookatmyquant codes) is an addition.
- a token is a fixed schedule: encode it once, replay it, change only
  position and cache length through one small buffer. no CPU sync inside
  a token; router top-8 runs on the GPU and the expert kernels read the
  ids from a buffer.
- per visit, ~6-8 kernels: norm + q/k/v/gate/rel projections (+ conv,
  clip); attention (banded or full, bias and sink inside); o-proj +
  residual + block-residual mix; router + top-8; expert gate/up + act;
  expert down + weighted sum + residual. ~400 dispatches per token.
- KV: f16 first, q8_0 at milestone 3. a fixed-size ring per SWA slot.

## milestones

each has a pass test. do not start the next before it passes.

0. reference dumps. with `llama/src-mix` `eval-callback` (rebuild with
   `-DLLAMA_BUILD_EXAMPLES=ON` in a scratch build dir), dump per-visit
   hidden states and final logits for 3 fixed prompts (short, 2k, 8k) to
   `ref/` (gitignored). pass: dumps reload and match a second run.
1. correct. load, run decode one token at a time, no fusion needed.
   pass: per-visit hidden states within 1e-2 relative of the dumps, same
   greedy tokens as `llama-cli --temp 0` for 2k generated tokens, KL of
   final logits <= 1e-3 mean.
2. fast decode. fuse to the schedule above; print GB/s per kernel.
   pass: <= 45 ms per token at short context, each weight kernel >= 75%
   of measured `memcpy` bandwidth, milestone 1 tests still pass.
3. depth. banded attention for one query, q8_0 KV.
   pass: <= 50 ms per token at 8k deep, needle at 31k found at 100k ctx.
4. prefill. simdgroup-matrix tiles, dequant once per tile.
   pass: pp512 >= 120 t/s.
5. serve. openai-compatible `/v1/chat/completions` (streaming) on its own
   port, chat template from the GGUF, a hax preset beside `inkling`.
   then dspark (`unsloth/Inkling-2-Mini-GGUF` `dspark-Inkling-2-Mini-Q8_0.gguf`) with
   the verify pass fused; pick block size by measurement.

## side tasks

- attention requant: q8_0 attention is ~40% of bytes per token. requantize
  attention to Q5_K/Q6_K from the BF16 original (~53GB; disk has room)
  with `llama-quantize --tensor-type`, keep everything else. pass: KL vs
  UD-Q4_K_XL logits not worse than IQ3_XXS's by more than 10%; then ~6 ms
  less per token in either engine.
- lookatmyquant kernel: an m2 mat-vec for its `rtr1`/`ht2` expert codes,
  timed alone. if one token's experts fit in <= 30 ms, its 8.9GB file
  becomes a second format.

## rules

- every experiment through `./wd` (see `llama/PLAN.md`, hazards); unload the
  app's model first. one model on the GPU at a time.
- numbers in commits are measured, one line each: what, before, after.
