# Reference dumps

The reference build is **patched llama.cpp b12600 + Unsloth pins**, with
our local Metal patches. The recorded dumps date from 21 November 2026.

`refdump.cpp` uses the llama.h C API to produce reference data for Gall.
It links the libraries that `llama/build` leaves in
`llama/src-mix/build/bin` without changing that source tree. It feeds a
plain-text prompt without a chat template, decodes greedily, and writes
tokens, logits and graph nodes to `ref/` (gitignored).

```sh
B=llama/src-mix/build/bin
mkdir -p scratch
clang++ -std=c++17 -O2 -w -Illama/src-mix/include \
  -Illama/src-mix/ggml/include tools/refdump.cpp -o scratch/refdump \
  -L$B -lllama -lggml -lggml-base -Wl,-rpath,@executable_path/../$B
M=$(gall model)
lockf /tmp/gall.lock scratch/refdump "$M" PROMPT.txt OUTDIR \
  [n_gen=64] [n_batch=256] [n_dump=2] [n_logits=65]
```

Settings: all layers on Metal, flash attention on, f16 KV, `n_ubatch =
n_batch`, `n_ctx` = prompt + n_gen + 256 rounded up to 256. `ALL=1` also
dumps the nodes that have only scheduler names (`node_N`, `x (view)`...).

`lockf` holds the GPU lock that Gall takes, so only one model loads at a
time. The prompt paths below describe the original reference
inputs; use each dump’s `prompt.txt` to reproduce it.

## Reference sets

| dir | prompt | prompt fed as | n_gen |
| --- | --- | --- | --- |
| `ref/short` | 34 tokens | one token per graph | 2000 |
| `ref/short.b256` | same | one batch | 2000 |
| `ref/short.400` | same | one token per graph, 401 logit rows | 400 |
| `ref/2k` | 2331 tokens, head of `docs/build.md` | one token per graph | 256 |
| `ref/2k.b256` | same | batches of 256 | 256 |
| `ref/8k.b256` | 10504 tokens, head of `tools/server/README.md` | batches of 256 | 64 |

For serial decode checks, compare with the directories without a suffix.
Use the `.b256` directories to check batched prefill. There is no
token-by-token `8k` dump; creating one takes about 14 minutes under the
lock:

```sh
lockf /tmp/gall.lock \
  scratch/refdump "$M" ref/8k.b256/prompt.txt ref/8k 64 1
```

`tools/cmp.py A B` compares two step directories or two `logits.f32`.

## Files

- `prompt.txt`: the text.
- `tokens.txt`: prompt token ids, one per line, as `llama_tokenize(...,
  add_special = true, parse_special = false)` returns them. This vocabulary
  has `add_bos_token = false`: there is no BOS. The first ID of `short` is
  371 (`The`).
- `gen.txt`: `n_gen` greedy ids. `gen[0]` is the argmax after the prompt.
- `logits.f32`: little-endian f32, rows of 200058 (the unpadded vocabulary,
  after the tanh softcap). Row 0 is the output after the last prompt token.
  Row `i + 1` is the output after `gen[i]` is fed. `n_logits` rows at most.
- `sNNNN/`: every named node of decode step N. Step N feeds `gen[N]` at
  position `len(tokens) + N` in a one-token graph.
- `sNNNN/index.tsv`: one line per file in execution order:
  `file, ggml op, ne0, ne1, ne2, ne3`.
- `sNNNN/<name>.<type>`: raw little-endian data in ggml order (ne0 is the
  fastest index). Type is `f32`, `i32` or `f16`. A name that occurs again
  in one graph gets `#1`, `#2`.

## Tensor names

A name is `<cb name>-<slot>`. The slot is the cache slot 0..47, not the
physical layer.

Visit order for this file (arch `inkling2`, legacy looped layout): slots
0..17 (physical 0..17, pass 0), then 24..35 (physical 6..17, pass 1), then
36..47 (physical 6..17, pass 2), then 18..23 (physical 18..23).
`index.tsv` has this order. Physical layer of slot s >= 24: `6 + (s - 24)
% 12`. The block position `ip` of a slot is its physical layer.

Per slot, in order (shapes for one token):

| file | shape | what |
| --- | --- | --- |
| `norm-S.f32` | 2304 | attention input norm. Full-attention slots (physical 5, 11, 17, 23) have `norm-S`, `#1`, `#2` for Q, K, V. Pass 1 and 2 SWA slots also have three (the messages make Q, K, V differ). The last `norm-S#k` is the `ffn_norm` output |
| `Qcur-S.f32` | 128 x 24 | after clip (and tau on full layers) |
| `Kcur-S.f32`, `Vcur-S.f32` | 128 x 8 | after conv, clip / rms norm; the values stored in the KV cache (the cache rounds them to f16) |
| `Rcur-S.f32` | 16 x 24 | after clip (and tau) |
| `kq_b-S.f32` | n_kv x 1 x 24 | dense bias minus the sink per head. n_kv is the padded cache view (256 here), not the number of valid keys |
| `kq_mask_rel_pos-S.f16` | n_kv x 1 x 24 | `kq_b` plus the causal/window mask (-inf on invalid cells), as the attention reads it |
| `kqv_out-S.f32` | 3072 | attention output before the gate and `wo` |
| `attn_out-S.f32` | 2304 | after `wo` and the post conv. Absent on the 8 slots that start a block (physical 0, 6, 12, 18): there the same tensor is `ffn_inp-S` |
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
| `ffn_moe_gate_up_scaled-S.f32` | 2304 x 8 | gate (first 1152) and up (last 1152) per expert, after the per-expert `.scale`. Physical 20..23 have no `.scale` tensor: the file is `ffn_moe_gate_up-S` |
| `ffn_moe_situ-S.f32` | 1152 x 8 | `beta*tanh(gate/beta)*sigmoid(gate) * lb*tanh(up/lb)`, beta 4, lb 25 |
| `ffn_moe_down_scaled-S.f32` | 2304 x 8 | down output per expert after `.scale` (`ffn_moe_down-S` on physical 20..23) |
| `ffn_moe_weighted-S.f32` | 2304 x 8 | x scaled weights |
| `ffn_moe_out-S.f32` | 2304 | sum over the 8 experts |
| `ffn_out-S.f32` | 2304 | after the post conv |
| `l_out-S.f32` | 2304 | slot output |

Without a slot: `embd` (token embedding), `norm` (after `token_embd_norm`),
`norm#1` (after `output_norm`), `result_norm` (the same values),
`result_output` (200058 logits).

`cb()` renames a tensor when it is called again on it. There is no
`ffn_moe_gate_up-S` beside `ffn_moe_gate_up_scaled-S`, no
`ffn_moe_gate-S` / `ffn_moe_up-S` (views; they keep scheduler names, use
`ALL=1`), and `ffn_moe_weights-S` has the name of the first `cb()` only
when a later one works on a new tensor. The mix (block residual attention),
control and convolve helpers have no `cb()` names.

## Callback names in build_moe_ffn

From `llama/src-mix/src/llama-graph.cpp`, in graph order: `ffn_moe_logits`,
`ffn_moe_probs`, `ffn_moe_probs_biased`, `ffn_moe_argsort` (on
`topk->src[0]`), `ffn_moe_topk`, `ffn_moe_weights`, `ffn_moe_weights_sum`,
`ffn_moe_weights_sum_clamped`, `ffn_moe_weights_norm`,
`ffn_moe_weights_scaled`, `ffn_moe_gate_up` then `ffn_moe_gate_up_scaled`
(same tensor; the scale is applied in `build_lora_mm_id` as
`mul_mat_id(...) * scale[id]`), `ffn_moe_gate`, `ffn_moe_up` (views),
`ffn_moe_situ`, `ffn_moe_down` then `ffn_moe_down_scaled`,
`ffn_moe_weighted`, `ffn_moe_out`.

## Numerical agreement

Measured 2026-11-21 on `short`, `scratch/cmp.py`-style comparison:

- A second run with the same arguments is bitwise equal: 1397 of 1397
  files in `s0000` and in `s0001`, and all 65 logit rows.
- The dump itself changes the result a little. The scheduler computes a
  dumped step node by node, so the metal backend does not fuse ops. Against
  a run with `n_dump = 0`: logits of row 0 are equal, row 1 differs by
  1.3e-3 (max abs), and the difference grows through the KV cache to 2.3
  max abs (0.67 mean over 65 rows). The 65 argmax tokens are equal.
- Batched prefill (one batch of 34) against token by token: row 0 differs
  by 0.15 max abs, the worst of 65 rows by 1.17. The 65 argmax tokens are
  equal. Only 67 of 1397 node files of `s0000` are bitwise equal. The two
  2000-token greedy texts are equal for 186 tokens and differ from token
  187.
- `2k` against `2k.b256`: row 0 differs by 0.52 max abs, the worst of 65
  rows by 2.81 (0.59 mean). The 65 argmax tokens and all 256 greedy tokens
  are equal.

Small rounding differences grow (top-8 expert selection is not
continuous). Compare a new engine step by step with teacher-forced tokens
from `gen.txt`, and judge it against these noise levels, not against 1e-6.
Even llama.cpp produces different long greedy continuations with its two
prefill modes. Use teacher-forced comparisons to separate numerical error
from the effect of an earlier token choice.

## HTTP smoke test

The smoke test uses the model’s tokenizer with a dummy answer. It does not
load GPU weights or acquire the model lock. It checks authentication,
streaming, repeated load/unload cycles, parking and graceful shutdown.

```sh
python3 tools/smoke.py target/release/gall "$(gall model)"
```

## Memory measurement

`memory.py` measures physical footprint and resident size before first
use, during prompt processing and generation, and after unloading. It
uses the real model, a temporary snapshot cache and an 805-token prompt.
The server it starts takes the GPU lock:

```sh
python3 tools/memory.py target/release/gall "$(gall model)"
```

The reported process footprint excludes some file-backed weight memory;
do not treat it as total RAM required by the model. See the
[engine notes](../docs/ENGINE.md) for the measured workload and limitations.
