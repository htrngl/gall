# reference dumps (milestone 0)

`refdump.cpp` is a standalone tool on the llama.h C API. it links the
libraries that `./build` leaves in `src-mix/build/bin` and does not change
`src-mix`. it feeds a plain-text prompt (no chat template), then decodes
greedily, and writes tokens, logits and graph nodes to `ref/` (gitignored).

```sh
B=src-mix/build/bin; mkdir -p scratch
clang++ -std=c++17 -O2 -w -Isrc-mix/include -Isrc-mix/ggml/include \
  tools/refdump.cpp -o scratch/refdump -L$B -lllama -lggml -lggml-base \
  -Wl,-rpath,$PWD/$B
. ./env
./wd scratch/refdump $M PROMPT.txt OUTDIR [n_gen=64] [n_batch=256] \
  [n_dump=2] [n_logits=65]
```

settings: all layers on metal, flash attention on, f16 KV, `n_ubatch =
n_batch`, `n_ctx` = prompt + n_gen + 256 rounded up to 256. `ALL=1` also
dumps the nodes that have only scheduler names (`node_N`, `x (view)`...).

## directories

| dir | prompt | prompt fed as | n_gen |
| --- | --- | --- | --- |
| `ref/short` | 34 tokens | one token per graph | 2000 |
| `ref/short.b256` | same | one batch | 2000 |
| `ref/2k` | 2331 tokens, head of `docs/build.md` | one token per graph | 256 |
| `ref/2k.b256` | same | batches of 256 | 256 |
| `ref/8k.b256` | 10504 tokens, head of `tools/server/README.md` | batches of 256 | 64 |

the engine decodes token by token, so compare with the directories without
a suffix. the `.b256` directories show what batched prefill changes. there
is no token-by-token `8k` (about 14 minutes under the lock); run
`refdump $M ref/8k.b256/prompt.txt ref/8k 64 1` when the engine needs it.
`tools/cmp.py A B` compares two step directories or two `logits.f32`.

## files

- `prompt.txt`: the text.
- `tokens.txt`: prompt token ids, one per line, as `llama_tokenize(...,
  add_special = true, parse_special = false)` returns them. this vocabulary
  has `add_bos_token = false`: there is no BOS. the first id of `short` is
  371 (`The`).
- `gen.txt`: `n_gen` greedy ids. `gen[0]` is the argmax after the prompt.
- `logits.f32`: little-endian f32, rows of 200058 (the unpadded vocabulary,
  after the tanh softcap). row 0 is the output after the last prompt token.
  row `i + 1` is the output after `gen[i]` is fed. `n_logits` rows at most.
- `sNNNN/`: every named node of decode step N. step N feeds `gen[N]` at
  position `len(tokens) + N` in a one-token graph.
- `sNNNN/index.tsv`: one line per file in execution order:
  `file, ggml op, ne0, ne1, ne2, ne3`.
- `sNNNN/<name>.<type>`: raw little-endian data in ggml order (ne0 is the
  fastest index). type is `f32`, `i32` or `f16`. a name that occurs again
  in one graph gets `#1`, `#2`.

## names

a name is `<cb name>-<slot>`. the slot is the cache slot 0..47, not the
physical layer.

visit order for this file (arch `inkling2`, legacy looped layout): slots
0..17 (physical 0..17, pass 0), then 24..35 (physical 6..17, pass 1), then
36..47 (physical 6..17, pass 2), then 18..23 (physical 18..23).
`index.tsv` has this order. physical layer of slot s >= 24: `6 + (s - 24)
% 12`. the block position `ip` of a slot is its physical layer.

per slot, in order (shapes for one token):

| file | shape | what |
| --- | --- | --- |
| `norm-S.f32` | 2304 | attention input norm. full-attention slots (physical 5, 11, 17, 23) have `norm-S`, `#1`, `#2` for Q, K, V. pass 1 and 2 SWA slots also have three (the messages make Q, K, V differ). the last `norm-S#k` is the `ffn_norm` output |
| `Qcur-S.f32` | 128 x 24 | after clip (and tau on full layers) |
| `Kcur-S.f32`, `Vcur-S.f32` | 128 x 8 | after conv, clip / rms norm; the values stored in the KV cache (the cache rounds them to f16) |
| `Rcur-S.f32` | 16 x 24 | after clip (and tau) |
| `kq_b-S.f32` | n_kv x 1 x 24 | dense bias minus the sink per head. n_kv is the padded cache view (256 here), not the number of valid keys |
| `kq_mask_rel_pos-S.f16` | n_kv x 1 x 24 | `kq_b` plus the causal/window mask (-inf on invalid cells), as the attention reads it |
| `kqv_out-S.f32` | 3072 | attention output before the gate and `wo` |
| `attn_out-S.f32` | 2304 | after `wo` and the post conv. absent on the 8 slots that start a block (physical 0, 6, 12, 18): there the same tensor is `ffn_inp-S` |
| `ffn_inp-S.f32` | 2304 | residual stream before the MoE |
| `ffn_moe_logits-S.f32` | 128 | router output |
| `ffn_moe_probs-S.f32` | 128 | sigmoid |
| `ffn_moe_probs_biased-S.f32` | 128 | + `exp_probs_b`, used for selection only |
| `ffn_moe_argsort-S.i32` | 128 | descending order |
| `ffn_moe_topk-S.i32` | 8 | selected experts |
| `ffn_moe_weights-S.f32` | 8 | unbiased probs of the selected |
| `ffn_moe_weights_sum-S`, `_sum_clamped-S` | 1 | sum, sum + 1e-20 |
| `ffn_moe_weights_norm-S.f32` | 8 | weights / sum |
| `ffn_moe_weights_scaled-S.f32` | 8 | x `expert_weights_scale` (8.0) |
| `ffn_moe_gate_up_scaled-S.f32` | 2304 x 8 | gate (first 1152) and up (last 1152) per expert, after the per-expert `.scale`. physical 20..23 have no `.scale` tensor: the file is `ffn_moe_gate_up-S` |
| `ffn_moe_situ-S.f32` | 1152 x 8 | `beta*tanh(gate/beta)*sigmoid(gate) * lb*tanh(up/lb)`, beta 4, lb 25 |
| `ffn_moe_down_scaled-S.f32` | 2304 x 8 | down output per expert after `.scale` (`ffn_moe_down-S` on physical 20..23) |
| `ffn_moe_weighted-S.f32` | 2304 x 8 | x scaled weights |
| `ffn_moe_out-S.f32` | 2304 | sum over the 8 experts |
| `ffn_out-S.f32` | 2304 | after the post conv |
| `l_out-S.f32` | 2304 | slot output |

without a slot: `embd` (token embedding), `norm` (after `token_embd_norm`),
`norm#1` (after `output_norm`), `result_norm` (the same values),
`result_output` (200058 logits).

`cb()` renames a tensor when it is called again on it. so there is no
`ffn_moe_gate_up-S` beside `ffn_moe_gate_up_scaled-S`, no
`ffn_moe_gate-S` / `ffn_moe_up-S` (views; they keep scheduler names, use
`ALL=1`), and `ffn_moe_weights-S` has the name of the first `cb()` only
when a later one works on a new tensor. the mix (block residual attention),
control and convolve helpers have no `cb()` names.

## cb() names in build_moe_ffn (SITU, merged gate_up)

from `src-mix/src/llama-graph.cpp`, in graph order: `ffn_moe_logits`,
`ffn_moe_probs`, `ffn_moe_probs_biased`, `ffn_moe_argsort` (on
`topk->src[0]`), `ffn_moe_topk`, `ffn_moe_weights`, `ffn_moe_weights_sum`,
`ffn_moe_weights_sum_clamped`, `ffn_moe_weights_norm`,
`ffn_moe_weights_scaled`, `ffn_moe_gate_up` then `ffn_moe_gate_up_scaled`
(same tensor; the scale is applied in `build_lora_mm_id` as
`mul_mat_id(...) * scale[id]`), `ffn_moe_gate`, `ffn_moe_up` (views),
`ffn_moe_situ`, `ffn_moe_down` then `ffn_moe_down_scaled`,
`ffn_moe_weighted`, `ffn_moe_out`.

## what matches

measured 2026-11-21 on `short`, `scratch/cmp.py`-style comparison:

- a second run with the same arguments is bitwise equal: 1397 of 1397
  files in `s0000` and in `s0001`, and all 65 logit rows.
- the dump itself changes the result a little. the scheduler computes a
  dumped step node by node, so the metal backend does not fuse ops. against
  a run with `n_dump = 0`: logits of row 0 are equal, row 1 differs by
  1.3e-3 (max abs), and the difference grows through the KV cache to 2.3
  max abs (0.67 mean over 65 rows). the 65 argmax tokens are equal.
- batched prefill (one batch of 34) against token by token: row 0 differs
  by 0.15 max abs, the worst of 65 rows by 1.17. the 65 argmax tokens are
  equal. only 67 of 1397 node files of `s0000` are bitwise equal. the two
  2000-token greedy texts are equal for 186 tokens and differ from token
  187.
- `2k` against `2k.b256`: row 0 differs by 0.52 max abs, the worst of 65
  rows by 2.81 (0.59 mean). the 65 argmax tokens and all 256 greedy tokens
  are equal.

so small rounding differences grow (top-8 expert selection is not
continuous). compare a new engine step by step with teacher-forced tokens
from `gen.txt`, and judge it against these noise levels, not against 1e-6.
llama.cpp does not give the same 2000 greedy tokens for its own two prefill
modes, so the milestone 1 test "same greedy tokens for 2k tokens" cannot be
passed as written.
