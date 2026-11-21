# dspark drafter: what it is and whether it pays here

source: `src-mix/src/models/dspark.cpp` (`graph_inkling2`,
`build_inkling2_head`), `src-mix/common/speculative.cpp`
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

(filled in by `ink dr`; see git log.)
