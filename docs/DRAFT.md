# dspark drafter: what it is and whether it pays here

source: `llama/src-mix/src/models/dspark.cpp` (`graph_inkling2`,
`build_inkling2_head`), `llama/src-mix/common/speculative.cpp`
(`common_speculative_impl_draft_dspark`), and the file
`dspark-Inkling-2-Mini-Q8_0.gguf` (402MB, arch `dflash`, 54 tensors, in the
hf cache beside the model).

## the model

inkling's own block drafter (`mtp.safetensors`): 3 transformer layers,
hidden 2304, 24 q heads / 8 kv heads x 128, ffn 9216 (silu(gate) * up, then
down), RoPE (neox pairs, base 1e6, all 128 dims), block size 8, non-causal
inside the block. layers 0 and 2 see a window of 1024 context positions,
layer 1 sees all of them. projections are Q8_0 (85MB per layer, 255MB in
all). it has no embeddings and no head: it borrows `token_embd`,
`token_embd_norm` and `output.weight` from the target.

### inputs from the target, per committed token

`dflash.target_layers = [5, 23, 41, 48]` index `t_layer_inp` in visit
order:

| feature | tensor | in the engine |
| --- | --- | --- |
| f0 | input of visit 5 (physical 5) = `l_out` of slot 4 | `H` after the first `elem` of visit 5 |
| f1 | input of visit 23 (pass 1, physical 11) = `l_out` of slot 28 | `H`, visit 23 |
| f2 | input of visit 41 (pass 2, physical 17) = `l_out` of slot 46 | `H`, visit 41 |
| f3 | `result_norm` (output mix, output norm) | `X * rsqrt(mean(VR^2) + 1e-6)` after the last `elem` |

for a token at position p, with `w_r = softmax(dspark_fuse[r])` (4 rows of
4):

    c_l   = rms(sum_k w_l[k] f_k) * dflare_norm_l          l = 0, 1, 2
    K_l[p] = rope(rms_head(k_ctx_l c_l) * k_ctx_norm_l, p)
    V_l[p] = v_ctx_l c_l
    f_rnn  = rms(sum_k w_3[k] f_k) * dspark_norm           kept for p = n - 1

so the drafter keeps its own K/V per layer (2 rings of 1024 and one full
cache: 4KB per token in f16, 410MB at 100k). cost per committed token: six
4.7MB mat-vecs, ~0.4 ms.

### a draft, after the anchor token `a` at position n (not yet in the target)

    x_0 = rms(token_embd[a]) * token_embd_norm;  x_1..7 = mask_embd
    3 layers over the 8 rows, position n + t:
      h = rms(x) * attn_norm
      q = rope(rms_head(wq h) * q_norm), k = rope(rms_head(wk h) * k_norm)
      keys: context K_l[max(0, n - 1024) .. n) (layer 1: [0, n)) + 8 rows
      x += wo softmax(q k / sqrt(128)) v
      g = rms(x) * ffn_norm;  x += down(silu(gate g) * (up g))
    cur = rms(x) * output_norm
    s = 0 (256); prev = a
    for i = 1..7:
      j = joint_proj [s, markov_w1[prev], cur_i, f_rnn] + b     5120 -> 768
      s = c + sigmoid(j[0:256]) * (s - c),  c = tanh(j[256:512])
      y = cur_i + hidden_out tanh(j[512:768]) + b
      prev = d_i = first argmax of output.weight y              (softcap is monotone)

the block is always 8 rows (masks see each other); `--spec-draft-n-max B`
only chooses how many of the 7 drafts are used. no confidence head. the
target then evaluates `[a, d_1..d_B]` in one batch: logits at row k check
d_{k+1}; the first mismatch is replaced by the target's own token, so a
pass yields 1..B+1 tokens. at temp 0 acceptance is equality with the
target argmax; under sampling llama.cpp samples the target at each row and
accepts while the sample equals the draft (output distribution unchanged,
acceptance lower).

### cost of a draft on this machine

- body: 255MB of q8_0 read once for 8 rows: ~3 ms of bytes at 88GB/s plus
  8x the dot products (0.25G weights x 8 = 2G MAC), est. 5-8 ms.
- head: as written, one 0.32GB output mat-vec per drafted token (5.4 ms
  each, sequential because of `prev`). two exact rewrites remove that:
  `output (cur_i + corr_i) = output cur_i + (output hidden_out) latent_i +
  output b`. the first term is one batched head pass for all rows
  (~6-7 ms); `P = output hidden_out` is 201024 x 256, precomputed once
  (256 head mat-vecs, 1.4 s; 103MB as f16), so each step is a 103MB
  mat-vec (~1.2 ms).
- total est.: 12-15 ms + 1.2 ms per used draft.

## does it beat serial decode here

per pass: draft(B) + verify t(B + 1) for E = (1 - a^(B+1)) / (1 - a)
tokens, a = per-position acceptance.

t(T) from kern's rates (weights 43 ms at T = 1: attention 14 byte-bound,
experts 23 element-bound, head 5.4) plus ~5 ms of small ops per row:

- attention and head weights: read once; each extra row adds dots only
  (~5% each).
- experts: half the token, and not shared. consecutive tokens share 3.5 of
  8 experts per visit (ref/short s0000 vs s0001, 48 visits), so two rows
  need ~1.56x the expert work, and it grows almost linearly after that
  (est. u(T) = 1 + 0.6 (T - 1)).

| B | t(B+1) ms | draft ms | a = 0.52: ms/tok | a = 0.7 | a = 0.85 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 (serial) | 47 | 0 | 47 | 47 | 47 |
| 1 | 68 | 14 | 54 | 48 | 44 |
| 2 | 88 | 15 | 57 | 47 | 40 |
| 3 | 108 | 16 | 65 | 49 | 39 |
| 7 | 190 | 21 | 103 | 67 | 44 |

plainly: at the 52% acceptance PLAN.md records, no block size beats serial
decode on this machine; the experts do not batch. it breaks even near
a = 0.7 at B = 2 and wins ~15% at a = 0.85 (B = 2-3), which is what the
mlx-vlm numbers show (prose slower, math and code 3-14% faster). so the
drafter is worth running only where acceptance is high, and the block size
should follow a running acceptance estimate (B = 0 below ~0.7).

what would change the answer: an expert kernel whose cost is shared across
rows (it is not, the ids differ), or verify rows that cost less than a
decode row. measured numbers below replace these estimates.

## measured

`gall dr $M $D ref/short` (teacher-forced on llama.cpp's greedy text) and
`gall run $M --tokens F --draft $D --block B -n 120` (free-running; the
verify is still serial, so the text equals `gall run` without a drafter:
checked on 120 tokens).

- features against the ref dump: rel 1.5e-4, 2.6e-4, 9.5e-3, 9.6e-3.
- hit rate per slot, prose, teacher-forced: 0.61, 0.35, 0.18, 0.14, 0.10,
  0.06, 0.08 (unconditional).
- tokens per pass (1 + drafts kept):

| prompt | block 1 | block 2 | block 3 | block 7 |
| --- | ---: | ---: | ---: | ---: |
| prose (ref/short) | 1.61 | 1.96 | 2.12 | 2.31 |
| math (induction proof) | | | 2.96 (from the block 7 run) | 4.36 |
| code (rust file head) | | | 3.51 | 5.52 |

- draft cost, gpu: 7 drafts 31 ms, 3 drafts 22 ms. shares for 7: output
  head for 7 rows 17 ms (q5_k, one read), `P latent` 7 x 1.2 ms, body
  5.9 ms (21 q8_0 passes of 8 rows, 255MB), joint + argmax 3 ms.
- kernel finding, for the batched verify too: a weight pass over 8 inputs
  costs 1.3x one input for q8_0 when the unit is dequantized once in the
  kernel text and the input count is a compile-time constant. through
  mv.metal's `u_<fmt>` per input it is 5x slower (the dequant is repeated),
  and with a run-time count 2x slower. q5_k: 7 inputs cost 3.1x one.

with t(T) still the estimate above (serial is 52 ms now) and the measured
draft cost: prose ties at block 1 (83 ms for 1.61 tokens) and loses at
block 2; math gains ~15% and code ~35% at block 3 (132 ms for 2.96 and
3.51 tokens). the real t(T) is below.

## batched verify

`Dr::pass` runs `[anchor, drafts]` on 1 to 8 activation rows and
`Dr::commit` keeps the accepted rows (`Model::truncate`) and injects their
context. three parts:

- weights (`drm.metal` `x_<fmt>_<C>`): a unit of 8 weights is dequantized
  once and dotted with C inputs, C a compile-time constant. a dense matrix
  has C = rows. the expert slabs of a visit are sorted on the gpu by how
  many rows use them (`dr.metal` `uni`), and each use count 1..rows is one
  dispatch whose grid comes from a buffer that `uni` wrote (indirect
  dispatch): a slab is read once and costs one dot per row that uses it.
- small ops (`ops.metal` under `ZR`, `Model::sched_z`): one dispatch for
  all rows, the row on grid z. a conv input goes into the ring before any
  row reads it (`stor`, and `post` runs twice, the first run only stores).
- row 0 is always kept, so its drafter context is in the same command
  buffer; a serial step through `Dr` costs what `Model::step` does.

measured 2026-11-22, `gall dr $M $D ref/short --tv 1,2,3,4,5,6,7,8`: a
pass and its commit on the reference text, best of 6, machine cool.

| rows | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ms | 51.1 | 72.8 | 95.7 | 122.9 | 158.6 | 197.3 | 255.0 | 265.6 |
| ms before | 51.8 | 90-97 | | 223-253 | | | | 738 |

"before" is the build this started from, run back to back with the first
build of the new kernels (which gave 82 / 149 / 316 at 2 / 4 / 8 rows).

- shares at 4 rows (`PROF=1`, every dispatch in its own command buffer):
  expert slabs 70.6 ms (iq3_xxs 25.9, nvfp4 22.6, iq2_s 18.0, the rest
  4.1), q8_0 18.5, q5_k 12.5, q6_k and q4_k 3.6, small ops 13.4, `uni`
  0.6, context of row 0 0.3. four serial tokens: slabs ~92, q8_0 ~52.
- iq3_xxs gate-up at 4 rows by use count: 1 use 10.3 ms, 2 uses 5.6,
  3 uses 4.5, 4 uses 5.5. consecutive rows share more slabs than the
  estimate above assumed, and a second use of a slab is not free.
- weight rows per threadgroup decide the speed and the best count falls
  with the use count (registers). shares in ms at 8 / 4 / 2 rows per
  threadgroup, pass of 4: iq3_xxs 1 use 10.3 / 10.5 / 11.9, 2 uses 7.0 /
  5.6 / 6.3, 3 uses 7.1 / 5.2 / 4.5; nvfp4 3 uses 7.5 / 4.8 / 3.2; q8_0
  4 uses 20.4 / 17.9 / 20.3. used: the serial count for 1 use, 4 for 2, 2
  for 3 and more (q8_0: 4 up to 6 uses).
- from 7 uses the rows are held and the inputs loaded in turn (`XW`): q8_0
  with 8 uses in a pass of 8, 47 ms against 55 and 61 with the inputs
  held (4 and 2 rows per threadgroup). 7 and 8 rows are still poor.
- the scale of a unit inside the dequantized vectors or after the dot: no
  difference (89 ms against 92 for the slabs, before the row counts).
- small ops as one dispatch for all rows, back to back: 2 / 4 / 8 rows
  86.5 / 156.6 / 322.8 -> 80.0 / 143.8 / 302.6 ms. their work still grows
  with the rows (5 ms serial, 13.4 at 4 rows).
- `uni` on one thread 2.7 ms per pass of 4 rows (48 dispatches), 6.3 at 8;
  as one group of 64 threads 0.6.
- 2331 tokens deep (ref/2k, 4 rows): score 5.9 and gather 4.4 ms against
  0.6 and 0.9 shallow; the rest as above.

## when to draft

`Dr::next` and `Spec`. the drafter gives a probability for each draft
(`conf`: softmax of its logits after the softcap), and that predicts
acceptance. teacher-forced on ref/short, 121 drafts of 7, each draft
counted while those before it hold:

| drafter's probability | < 0.3 | 0.3-0.5 | 0.5-0.7 | 0.7-0.9 | > 0.9 |
| --- | ---: | ---: | ---: | ---: | ---: |
| drafts | 25 | 43 | 46 | 46 | 123 |
| accepted | 0.12 | 0.28 | 0.43 | 0.50 | 0.87 |

- `Spec.q` keeps the acceptance by probability (12 bins of the logit) as a
  running mean over verified drafts.
- a round verifies the drafts in order while a draft's chance to be kept
  (the product of q so far) is worth more than the row it adds:
  `p > rate * (tv[j + 2] - tv[j + 1])`, rate the better of serial and
  drafted decode. one more row costs 22-27 ms, so the bar is near 0.45.
- the draft after the last verified one is checked for free against the
  new pending token; if it holds, the rest of the draft is kept for the
  next round and no new draft is made.
- drafting is on while a new draft is expected to pay: tokens and seconds
  of each new draft over its rounds, from q and the measured `tv`, `td`,
  as a running mean against `1 / tv[1]`. off, a draft is made 8 tokens
  after one that would have paid, else after 32, 64, 128.
- the draft length is 2 more than the mean of drafts that held, 3 to 7.

free-running, 128 tokens each, `gall run $M --tokens F -n 128` against the
same with `--draft $D`, back to back in one `./wd` run, machine cool. the
text is equal to plain greedy in every row.

| prompt | plain t/s | drafted t/s | passes by drafts kept (0..) | drafts |
| --- | ---: | ---: | --- | ---: |
| prose (ref/short) | 19.5 | 19.1 | none: off | 4 |
| math (induction proof) | 19.4 | 22.6 | 6, 19, 7, 13, 0, 1 | 36 |
| code (rust file head) | 19.4 | 23.7 | 7, 9, 3, 13, 2, 5 | 33 |

- a fixed block of 3 (`--block 3`), no gate: prose 14.5 t/s, math 22.4;
  code 16.5 against 17.1 plain on a warm machine.
- earlier forms, each against its own plain run: the gate with the switch
  on measured round rates, prose 17.0 (it kept turning on), math 21.8,
  code 21.3; with the kept drafts as well, 17.7, 19.3 (plain 17.4, warm),
  21.5.
- a serial step through `Dr` was 55 ms while its context ran in a second
  command buffer; 51 with it in the pass.
- 2331-token prompt (ref/2k, 48 tokens): text equal at a fixed block of 3
  and gated; gated stays off (17.0 against 18.1 t/s, the plain run first
  and the machine warmer for the second).

- 3556-token code prompt (the head of `src/tok.rs`), 96 tokens, text
  equal: plain then drafted 18.1, 17.7 t/s; drafted then plain 19.0, 16.2.
  the second run of a pair is on a warmer machine (its serial step 60-62
  ms against 55), so the gain is between the two: about 7%.

## prompt context

`Dr::feed` is `Model::feed` with the drafter's features: the batched
prefill (`prefill_tap`) hands back, per token, the residual entering
visits 5, 23, 41 and the normed output mix, and `feed` injects 8 tokens
per command buffer.

- slot hits after a prefilled prompt against a stepped one, ref/short,
  121 drafts: 0.64, 0.34, 0.21 against 0.64, 0.35, 0.21. after the
  2331-token prompt of ref/2k (17 drafts): 0.65, 0.29, 0.24.
- the drafter attends only to the positions injected in an unbroken run
  (`span`), so a prefix restored from a snapshot is left out, not read as
  zeros.
- the drafter's attention had one thread chain per 32 keys of the whole
  context: a draft of 3 took 38 ms at 2331 tokens against 21 shallow. in
  chunks of 256 keys with one chain: draft of 7 43 ms. with 16 simdgroups
  of 16 keys per chunk and a merge: 31.5 ms, 30.3 shallow.

## in the server

`gall serve` opens the drafter when `dspark-Inkling-2-Mini-Q8_0.gguf` lies
beside the model (`gall up` fetches it; `--no-draft` leaves it out).
behind `Lm::step` is `Dr::next(m, st, tok)`: it feeds one token and names
the activation row whose logits follow it.

- a pass leaves `m.pos` after its first token; the rows verified ahead lie
  past it. a later `next(tok)` takes the next row only if tok is that
  row's draft (its context is injected then and `m.pos` moves), else the
  rows are dropped and a new pass starts at tok. so the server samples
  the model's own logits at every step, with any sampler, and a draft is
  kept only while the sample equals it. stop strings, tool calls,
  max_tokens and a hang-up need nothing: the model never holds more than
  was asked for.
- `Spec.q` then learns the acceptance under the sampler in use, so the
  gate and the switch hold there as well.
- under `response_format` drafting is off for the request.
- `truncate`, `reset`, `prime` and `feed` drop the rows ahead;
  `Model::top` says how far a pass wrote the conv rings, `truncate`
  counts from it, and a snapshot stores it (tag `gall snap 2`).
- what `prime` restores or feeds (the system region from 512 tokens) has
  no drafter context: `Dr::forget`.
- the model keeps the three residual features itself (`keep`, 3
  dispatches a token), so a serial step with a drafter is `Model::step`
  with its pre-encoded token, and the context goes out in a command
  buffer that nobody waits for.
- from 4096 tokens the int8 score, gather and fix kernels take the rows on
  grid z as the f16 ones do; a pass claims the int8 copy of its first row
  only, later rows are packed again when they are kept.

measured 2026-11-22 on the build merged with main 2bba3ed, machine cool.

- `gall check ref/short -n 64`, back to back with main's binary: 48.2
  against 48.1 gpu ms/token, 0 greedy mismatches, kl mean 1.05e-4.
- a pass by rows 1..8: 49.5, 70.3, 95.7, 117.0, 149.9, 181.4, 235.1,
  243.9 ms.
- `gall run`, 128 tokens (code 96), plain against `--draft`, text equal:
  prose 20.5, 20.2 t/s; math 20.4, 21.6; code 20.7, 23.3.
- a 5406-token prompt (the head of `src/serve.rs`), 48 tokens, text equal:
  plain 19.0, a fixed block of 3 21.6, gated 20.6 t/s.
- one timing sample with a kernel compile in it (the first pass of 3
  rows, 280 ms) had closed that block for the run: the row kernels now
  compile at `Dr::open` and a sample counts for at most 1.5 times its
  mean.
- a side server on port 9941, 120 tokens a request, `--no-draft` then
  drafting in one `./wd` run. the whole message is equal in every pair,
  under sampling with the same seed too. t/s is the server's
  `predicted_per_second`:

| request | temp 0, off | temp 0, on | default sampling, off | on |
| --- | ---: | ---: | ---: | ---: |
| prose (how a transformer predicts) | 20.5 | 20.7 | 20.2 | 20.5 |
| math (induction proof) | 20.4 | 23.4 | 20.0 | 21.8 |
| code (rust key-value store) | 20.4 | 21.4 | 20.2 | 20.4 |

- rows ahead kept of rows verified, on: temp 0 8 of 10, 47 of 51, 51 of
  67; sampling 13 of 15, 47 of 53, 33 of 48. with 200 tokens: temp 0 math
  20.3 -> 21.9, code 20.3 -> 20.9; sampling math 19.6 -> 21.0, code 19.7
  -> 20.0.
- the answers are the model's thinking text, which reads as prose, so the
  gains are below those of `gall run` on raw math and code.
- two turns with prefix reuse, the same prompt again (checkpoint), an
  830-token system prompt (snapshot written and read), an idle unload and
  an image request give the same messages with and without the drafter.
- that test found `Model::feed` stepping an image row that ends the
  prompt through the token table (the image as the last content part):
  the row now takes the general path.

- a reload after an idle unload took 0.71-0.75 s with the drafter against
  0.06-0.08 without: `Dr::open` ran 33 head passes for `P` each time. `P`
  and `wb` are now a file in `~/.cache/gall` (104 MB, named by the sizes
  of the two models and the drafter's date) read into the gpu buffers,
  and the drafter's map is kept as the model's is: 0.07-0.10 s. footprint
  while unloaded, `--ctx 102400`: 224 MB without the drafter, 231 before,
  229-231 after.
- `sample()` at the default settings (temp 1, no filter) built 200k pairs
  and exps: 0.80-1.58 ms of host time a token. `pick()` skips the terms
  that cannot move a float32 sum or the draw: 0.21-0.37 ms, and the
  messages of 6 requests of 80 tokens (seeds 7 and 9) are the same.

## left

- sampling: drafts are the drafter's argmax; a draft sampled from the
  drafter with the matching accept rule would keep more at high
  temperature.
- the draft is 15-31 ms per round and half of it is the output head for
  the drafter's rows (q5_k, 5.4 ms for one row); math makes a draft per
  3.6 tokens. a coarser copy of the head for drafting would cut it.
- small ops at 4 rows are 13.4 ms against 5 serial: `ctl`, `low`, `post`,
  `rows`, `msg` do work per row that could be shared or made wider.
- 7 and 8 rows (255, 266 ms): the kernels for 7 and 8 uses.
- snapshots do not store the drafter's context, and `prime` feeds the
  system region without it; after either the drafter starts from the
  tokens fed since.
- at the 4096-token line a pass uses f16 for all its rows where serial
  steps would switch to the int8 copy mid-pass.
- the drafter's context cache is 4KB per token (428MB at 102400), and its
  full-attention layer reads all of it per draft.
- `KV=q8` with a batched pass is untested.
