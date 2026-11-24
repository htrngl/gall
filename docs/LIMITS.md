# limits and next steps

the limit model and the outcome of each proposal from the review of 23
November 2026, measured to 24 November 2026 on the 24 GB M2 MacBook Air
(warm, interleaved medians: rates are about 20% under the cool figures
in ENGINE.md).

## the limit model

a token multiplies 4.8 G weights: 3.06 G expert (63.7 M per visit),
1.29 G attention, 0.46 G head. it reads 2.9 GB. at 87 GB/s that is 33
ms, so serial decode cannot pass 30 tokens/s while a token reads every
byte once. it now takes 39.5-39.8 ms of GPU time (about 25 tokens/s).

the earlier "150 G weights/s cap" was not a cap. it was conversion and
sign work per weight: bytes read as subnormal halves (byte b is b 2^-24)
convert for free, the power of two goes into the block scale, and the
results are bit-identical. from a cached 64 KB slab Q8_0 runs 195-199 G
weights/s.

where a token goes now, against the floor of its bytes at 87 GB/s:

| part | ms | floor, ms |
| --- | ---: | ---: |
| IQ3_XXS experts | 5.95 | 5.4 (arithmetic-bound) |
| NVFP4 experts | 5.86 | 6.0 |
| IQ2_S experts | 3.37 | 2.7 |
| IQ4_NL, IQ2_XS, IQ2_XXS experts | 0.92 | 0.82 |
| experts, total | ~16.1 | 14.9 |
| Q8_0 attention | 12.9 | 12.8 |
| Q5_K head and attention | 4.26 | 4.2 |
| Q6_K, Q4_K attention | 1.68 | 1.6 |
| small-kernel chain | ~4.7 | ~1.6 |

decode rates by format, G weights/s before and after: IQ3_XXS 142 ->
204, IQ2_S 147 -> 194, IQ2_XXS 166 -> 241, IQ2_XS 141 -> 183, NVFP4 141
-> 157, IQ4_NL 143 -> 158, Q4_K 110 -> 133.

## measured here

- Q8_0 attention is incompressible. over blocks 0, 7, 12, 20: entropy
  7.56-7.68 bits per weight, 29.5 distinct values per block of 32, every
  block's maximum 127. a lossless smaller format saves at most 4%, so
  the 1.11 GB stays.
- expert overlap, 96 greedy tokens after `tests/short` and after
  `ref/2k`, all 48 visits:

  | rows in a batch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | distinct experts per visit | 8 | 12.6 | 16.3 | 19.4 | 22.2 | 24.8 | 27.2 | 29.5 |

  consecutive tokens share 42-44% (22% in visit 0, 73% in visit 17);
  tokens 4 apart share 35%.
- the bus: no `bw` shape (4-1024 words per thread, lanes adjacent or
  not) and no anonymous buffer beat the Q8_0 mat-vec, so 87 GB/s stays
  the floor. one 7.5 MB tensor read again runs 117-119 GB/s against 88
  cold: that is the system cache, not the bus.
- barriers: removing all of them saves 3.4-4 ms. the cost is each small
  kernel's own latency, not the barrier (a barrier and a launch cost 1-2
  us). ms per token by kind: dots 0.38, elem 0.56, low 0.84, ctl 0.79,
  rows 0.54, top8 0.04, post 0.51, score 0.41, gather 0.50, merge 0.05,
  act 0.21.
- the model is sensitive to single-ulp changes through near-tied
  experts, so kernel work is held to bit-identity: `gall mvtest` and
  `gall dr --drift` print hashes of the output bits.

## the proposals and their outcomes

1. batched verify on tiles: done. slabs of 3+ uses and dense matrices
   in passes of 3+ rows go through `simdgroup_matrix` tiles; 1-2 uses
   stay dots. pass ms for 1-8 rows: 47 67 93 114 150 181 235 244 -> 41.5
   61 84 96 105 115 125 135 (the estimate was 47-140).
2. drafts that cost nothing: copy drafts done (96-97% of copied rows
   kept on a code edit, about 6 rows a pass); drafting now runs under
   `response_format`. forced tokens: ruled out (below).
3. a cheap head for the drafter: done. a shortlist of 32,768 tokens; a
   7-token draft 23.7 -> 8 ms, tokens a pass at block 7 code 5.51 ->
   5.23, maths 3.71 -> 3.70, prose 2.52 -> 2.51.
4. draft while verifying: ruled out (below).
5. a tree, not a chain: ruled out (below).
6. sixteen rows: ruled out (below).
7. fewer dispatches: ruled out (below). the dispatches are not where
   the time is; see barriers above.
8. what capped decode at 150 G weights/s: found. it was conversion and
   sign work (the limit model above). decode 45.2 -> 39.5-39.8 ms.
9. the CPU beside the expert kernels: ruled out (below).
10. commit the next token before the wait: done (`Model.greedy`), 0.4-0.5
    ms a token, in `gall run` and in the server for temperature 0
    requests without `response_format` and without the drafter.
11. the true bus ceiling: measured, 87 GB/s holds (above).
12. cooling: open.

## still open

- drafted prose is about level with serial decode (0.97-1.05 of its
  rate). the controller must keep it at or above serial; that work is
  not done.
- a grammar for tool calls built from the request's `tools` with the
  existing schema matcher, for validity. forced runs would also need a
  canonical form.
- IQ3_XXS and IQ2_S run about 1.2 ms over their floors together.
- the small-kernel chain runs about 3 ms over its floor.
- cooling: throughput falls 10% after 2.5 minutes; a thermal pad to the
  case or a cooled stand keeps the cool rate.
- long copies at prefill cost: verify 32-64 copied tokens in one
  prefill-shaped batch (9.25 ms a row against 16.9 in an 8-row pass). on
  a code edit 82.6% of output tokens are in copy runs of 16+ and 59.4%
  in runs of 64+ (`tools/runs.py`); on a quote rewrite 22.5% and none.
- drafts under sampling: proposals are argmax, so a sampled request
  keeps few. gumbel noise shared by the sampler and the drafter would
  keep them and leave the sample the model's own. hax sends no sampler
  fields (the model file's temperature 1.0 applies), so its provider is
  set to temperature 0 in `~/.config/hax/config.json`.
- `tools/long.py` (10 tool turns to 52.6k tokens) has run on the dummy
  server only. the hold after `tool_calls` and the first step after a
  restore (463-533 ms against 52-60, the int8 rebuild under a quarter of
  it) are not measured on it.
- drafter context is not saved, so acceptance just after a restore is
  that of a short context; acceptance against context length is not
  measured.
- passes run far over their byte floors (8 rows 141 ms against 67:
  IQ2_S 25.9 against 9.1, IQ3_XXS 31.1 against 17.0, small ops 19.8),
  and prefill is arithmetic-bound (experts 4.65 ms a token against 1.71).
  64-column prefill tiles were faster by kernel (Q8_0 1,129 -> 1,197 G
  MAC/s, IQ3_XXS 688 -> 827) and not shown faster end to end on a loaded
  mac; dense Q8_0 and Q6_K as dots at 3 rows (branch `dots3`, 86.0 ->
  83.5 ms in one process, not shown between processes) likewise. both
  want a quiet machine.
- long context: a step gains 19 ms by 29.9k and is unmeasured at 60k;
  an 8-row pass costs 4.3 steps at 30k against 3.3 at short context.
- lossy and untried: dropping chosen experts of small gate weight,
  attention below Q8_0, a two-stage head.

## ruled out

- a smaller exact Q8_0 (entropy above).
- IQ3_XXS as 4-bit codes, a sub-block per thread, int8 activations.
- tables of partial dots for the grid formats: 16 KB per input
  position, and a thread has registers only.
- exact pruning of the head: a norm bound on 2,304 dimensions is about
  45 times the typical logit, so it excludes nothing.
- reuse across the three passes of the core layers: each pass needs the
  one before.
- forced tokens: 0 of 201 constrained positions allow exactly one token
  (whitespace and alternative tokenisations always fit). tool-call
  wrappers and the chat template have no grammar.
- draft while verifying: with its context 4 tokens late, first-slot
  acceptance falls 0.89 -> 0.48 code, 0.77 -> 0.34 maths, 0.61 -> 0.22
  prose.
- a tree on the second choice: right at the break 0.33 code, 0.38
  maths, 0.24 prose; a branch row returns at most 0.27 tokens against a
  cost of 0.23-0.55, under 0.5% net.
- snapshot rungs every 4,096 tokens (a rung is 188 MB against 134 MB of
  rows) and saved int8 rows (553 MB more to read at 30k against a 90 ms
  rebuild).
- a cost table by context length: 17.3 tokens/s against 17.7 with the
  fixed table at 31.5k, one run each.
- prefill: batches of 256 and 1,024 against 512 (no difference), and
  the last visit for the last 19 rows only (115.6 against 125.1
  tokens/s).
- Q8_0 at 2 rows is at 1.15 reads already (14.4 ms against 12.5).
- more rows: the conv ring caps a pass at 13 rows; a 13-row build gave
  18.6 ms a token against 19.0 at 8 (2%), dropped.
- fewer dispatches: fusing dots and elem, rows and top8, or score and
  gather was each slower. resource barriers, gate rows in barrier
  shadows and a split post gave no gain or were slower. act inside `up`
  cannot be bit-identical under fast math (about half the outputs 1 ulp
  off); `contract(off)` moves ref/short kl to 1.31e-3 and gives 2
  mismatches on ref/2k.
- the CPU beside the GPU: 96 encoder splits cost 3.7 ms, 19 ms with the
  event waits.
- Q8_0 staged through threadgroup memory in verify: 12 ms slower.
  threadgroup tables pay at 1-2 KB, not at IQ2_S's 8 KB.
